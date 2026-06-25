"""
Error analysis utilities for classification and ranking evaluation.

Used by the /error-analysis skill. Can also be imported directly.

Usage:
    from scripts.error_analysis import ErrorAnalyzer
    analyzer = ErrorAnalyzer(predictions, ground_truth, label_names)
    report = analyzer.full_analysis()
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


@dataclass
class PatternStat:
    name: str
    error_freq: float
    correct_freq: float
    enrichment: float
    p_value: float
    n_errors: int
    n_correct: int


@dataclass
class ErrorCluster:
    name: str
    pattern: str
    count: int
    examples: List[dict]
    stage_attribution: str = ""
    recommended_fix: str = ""


@dataclass
class CellAnalysis:
    predicted: str
    actual: str
    n_errors: int
    pct_of_total: float
    patterns: List[PatternStat]
    clusters: List[ErrorCluster]


def extract_text_features(text: str) -> dict:
    """Extract interpretable features from a text input."""
    words = text.lower().split()
    return {
        "char_length": len(text),
        "word_count": len(words),
        "unique_words": len(set(words)),
        "has_age": any(w in words for w in ["age", "year", "years", "yo", "old"]),
        "has_gender": any(w in words for w in ["male", "female", "man", "woman", "boy", "girl"]),
        "has_lab": any(w in words for w in ["hemoglobin", "creatinine", "platelet", "wbc", "hgb", "bilirubin", "alt", "ast"]),
        "has_cancer": any(w in words for w in ["cancer", "carcinoma", "tumor", "tumour", "malignant", "oncology", "lymphoma", "leukemia"]),
        "has_cardiac": any(w in words for w in ["cardiac", "heart", "coronary", "arrhythmia", "hypertension", "cad"]),
        "has_neuro": any(w in words for w in ["neurological", "brain", "stroke", "seizure", "epilepsy"]),
        "has_diabetes": any(w in words for w in ["diabetes", "diabetic", "insulin", "glucose", "hba1c", "a1c"]),
        "has_exclusion_lang": any(w in words for w in ["no", "not", "exclude", "without", "unable"]),
        "has_inclusion_lang": any(w in words for w in ["must", "required", "confirmed", "proven", "eligible"]),
    }


def compute_contrastive_stats(
    error_features: List[dict],
    correct_features: List[dict],
    feature_name: str,
) -> Optional[PatternStat]:
    """
    Compare a binary feature's frequency in error vs correct sets.
    Returns None if the feature is not discriminative.
    """
    n_err = len(error_features)
    n_cor = len(correct_features)
    if n_err == 0 or n_cor == 0:
        return None

    err_count = sum(1 for f in error_features if f.get(feature_name))
    cor_count = sum(1 for f in correct_features if f.get(feature_name))

    err_freq = err_count / n_err
    cor_freq = cor_count / n_cor if n_cor > 0 else 0

    if cor_freq == 0:
        enrichment = float('inf') if err_freq > 0 else 1.0
    else:
        enrichment = err_freq / cor_freq

    # Fisher's exact test for 2x2 contingency
    try:
        from scipy.stats import fisher_exact
        table = [
            [err_count, n_err - err_count],
            [cor_count, n_cor - cor_count],
        ]
        _, p_value = fisher_exact(table)
    except ImportError:
        p_value = -1.0

    return PatternStat(
        name=feature_name,
        error_freq=err_freq,
        correct_freq=cor_freq,
        enrichment=enrichment,
        p_value=p_value,
        n_errors=err_count,
        n_correct=cor_count,
    )


def compute_continuous_stats(
    error_values: List[float],
    correct_values: List[float],
    feature_name: str,
) -> Optional[PatternStat]:
    """Compare a continuous feature between error and correct sets."""
    if len(error_values) < 5 or len(correct_values) < 5:
        return None

    err_mean = np.mean(error_values)
    cor_mean = np.mean(correct_values)

    try:
        from scipy.stats import mannwhitneyu
        _, p_value = mannwhitneyu(error_values, correct_values, alternative='two-sided')
    except ImportError:
        p_value = -1.0

    enrichment = err_mean / cor_mean if cor_mean != 0 else float('inf')

    return PatternStat(
        name=feature_name,
        error_freq=err_mean,
        correct_freq=cor_mean,
        enrichment=enrichment,
        p_value=p_value,
        n_errors=len(error_values),
        n_correct=len(correct_values),
    )


class ErrorAnalyzer:
    """
    Systematic contrastive error analysis for classification tasks.

    Analyzes each off-diagonal cell of the confusion matrix independently,
    finding features that are over-represented in errors relative to correct
    predictions with the same ground truth label.
    """

    def __init__(
        self,
        examples: List[dict],
        label_names: Optional[List[str]] = None,
        input_field: str = "input",
        predicted_field: str = "predicted",
        actual_field: str = "actual",
    ):
        self.examples = examples
        self.input_field = input_field
        self.predicted_field = predicted_field
        self.actual_field = actual_field

        labels = sorted(set(ex[actual_field] for ex in examples) | set(ex[predicted_field] for ex in examples))
        self.label_names = label_names or [str(l) for l in labels]
        self.labels = labels

    def confusion_matrix(self) -> np.ndarray:
        y_true = [ex[self.actual_field] for ex in self.examples]
        y_pred = [ex[self.predicted_field] for ex in self.examples]
        return confusion_matrix(y_true, y_pred, labels=self.labels)

    def analyze_cell(self, predicted_label, actual_label, max_examples: int = 3) -> CellAnalysis:
        """Analyze one off-diagonal cell: predicted=X, actual=Y."""
        errors = [ex for ex in self.examples
                  if ex[self.predicted_field] == predicted_label and ex[self.actual_field] == actual_label]

        # correct baseline: same actual label, correctly predicted
        correct = [ex for ex in self.examples
                   if ex[self.actual_field] == actual_label and ex[self.predicted_field] == actual_label]

        total_errors = sum(1 for ex in self.examples if ex[self.predicted_field] != ex[self.actual_field])
        pct = (len(errors) / total_errors * 100) if total_errors > 0 else 0

        error_features = [extract_text_features(ex[self.input_field]) for ex in errors]
        correct_features = [extract_text_features(ex[self.input_field]) for ex in correct]

        # contrastive analysis on binary features
        binary_feats = [k for k in error_features[0] if isinstance(error_features[0][k], bool)] if error_features else []
        patterns = []
        for feat in binary_feats:
            stat = compute_contrastive_stats(error_features, correct_features, feat)
            if stat and stat.enrichment > 1.5 and stat.p_value < 0.1:
                patterns.append(stat)

        # contrastive on continuous features
        continuous_feats = [k for k in error_features[0] if isinstance(error_features[0][k], (int, float))] if error_features else []
        for feat in continuous_feats:
            err_vals = [f[feat] for f in error_features]
            cor_vals = [f[feat] for f in correct_features]
            stat = compute_continuous_stats(err_vals, cor_vals, feat)
            if stat and stat.p_value < 0.1:
                patterns.append(stat)

        patterns.sort(key=lambda p: p.enrichment, reverse=True)

        # representative examples
        cluster = ErrorCluster(
            name=f"pred={predicted_label}, actual={actual_label}",
            pattern=patterns[0].name if patterns else "no dominant pattern",
            count=len(errors),
            examples=[{self.input_field: ex[self.input_field][:200]} for ex in errors[:max_examples]],
        )

        return CellAnalysis(
            predicted=str(predicted_label),
            actual=str(actual_label),
            n_errors=len(errors),
            pct_of_total=pct,
            patterns=patterns,
            clusters=[cluster],
        )

    def full_analysis(self) -> dict:
        """Run contrastive analysis on every off-diagonal cell."""
        cm = self.confusion_matrix()
        cells = []
        for i, actual in enumerate(self.labels):
            for j, predicted in enumerate(self.labels):
                if i == j:
                    continue
                if cm[i][j] == 0:
                    continue
                cell = self.analyze_cell(predicted, actual)
                cells.append(cell)

        cells.sort(key=lambda c: c.n_errors, reverse=True)

        return {
            "confusion_matrix": cm.tolist(),
            "label_names": self.label_names,
            "total_examples": len(self.examples),
            "total_errors": sum(c.n_errors for c in cells),
            "cells": cells,
        }

    def print_report(self) -> None:
        """Print a formatted error analysis report."""
        result = self.full_analysis()
        cm = np.array(result["confusion_matrix"])
        names = result["label_names"]

        print("=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)
        header = "actual\\pred".ljust(20) + "".join(n[:12].ljust(14) for n in names)
        print(header)
        for i, name in enumerate(names):
            row = name[:18].ljust(20) + "".join(str(cm[i][j]).ljust(14) for j in range(len(names)))
            print(row)

        print(f"\nTotal: {result['total_examples']} examples, {result['total_errors']} errors")

        for cell in result["cells"]:
            print(f"\n{'=' * 60}")
            print(f"CELL: predicted={cell.predicted}, actual={cell.actual}")
            print(f"  {cell.n_errors} errors ({cell.pct_of_total:.1f}% of all errors)")
            print(f"{'=' * 60}")

            if cell.patterns:
                print(f"\n  {'Pattern':<25} {'Err%':>8} {'Cor%':>8} {'Enrich':>8} {'p':>8}")
                print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
                for p in cell.patterns[:10]:
                    print(f"  {p.name:<25} {p.error_freq:>7.1%} {p.correct_freq:>7.1%} {p.enrichment:>7.1f}x {p.p_value:>8.4f}")
            else:
                print("  No significantly enriched patterns found")

            for cluster in cell.clusters:
                print(f"\n  Examples ({cluster.count} total):")
                for ex in cluster.examples:
                    text = ex.get("input", "")[:120]
                    print(f"    - {text}...")
