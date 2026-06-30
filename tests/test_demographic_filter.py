"""
Tests for the age/demographic pre-filter pure functions.

Run locally:  pytest tests/test_demographic_filter.py -v

All tests are pure (no model loading). They cover:
- _extract_patient_age: format variants, first-match anchoring
- _trial_age_range: inclusion min, inclusion max, exclusion min/max,
                    gestational-age masking, sanity check
- _trial_excludes_age: end-to-end decisions
- Known error-analysis cases: topics 20155 (31yo vs age≥50), 45 (infant vs adults),
  201516 (4yo vs age 12+)
"""

import pytest
from ctmatch.matching.demographic import (
    extract_patient_age as _extract_patient_age,
    trial_age_range as _trial_age_range,
    trial_excludes_age as _trial_excludes_age,
    mask_nonchrono_age as _mask_nonchrono_age,
)


# ── _extract_patient_age ──────────────────────────────────────────────────────

class TestExtractPatientAge:

    def test_year_old(self):
        assert _extract_patient_age("A 58-year-old male presents...") == 58.0

    def test_yo_abbreviation(self):
        assert _extract_patient_age("A 31 yo female with no PMH") == 31.0

    def test_yo_abbreviation_period(self):
        assert _extract_patient_age("A 44 y.o. male is brought to the ER") == 44.0

    def test_week_old_infant(self):
        age = _extract_patient_age("A 15-week-old infant brought to the clinic")
        assert abs(age - 15 / 52.0) < 1e-9

    def test_month_old_infant(self):
        age = _extract_patient_age("A 7-month-old boy is brought to emergency")
        assert abs(age - 7 / 12.0) < 1e-9

    def test_day_old_neonate(self):
        age = _extract_patient_age("A 3-day-old female infant with jaundice")
        assert abs(age - 3 / 365.0) < 1e-9

    def test_no_age_returns_none(self):
        assert _extract_patient_age("A group of workers preparing for a trip.") is None

    def test_first_match_anchors_to_patient_not_family_member(self):
        # Vignette opens with "A 31-year-old woman"; later mentions "70-year-old mother"
        text = (
            "A 31-year-old woman with no previous medical problems presents. "
            "Her 70-year-old mother has hypertension."
        )
        assert _extract_patient_age(text) == 31.0

    def test_integer_age_without_hyphen(self):
        assert _extract_patient_age("A 4 year old boy presents to the ER") == 4.0

    def test_six_month_old(self):
        age = _extract_patient_age(
            "A 6-month-old male infant has a urine output of less than 0.2 mL/kg/hr"
        )
        assert abs(age - 6 / 12.0) < 1e-9


# ── _mask_nonchrono_age ───────────────────────────────────────────────────────

class TestMaskNonchronoAge:

    def test_gestational_age_masked(self):
        assert 'age' not in _mask_nonchrono_age(
            "gestational age ≥ 33 weeks"
        ).lower().replace('nonchrono', '')

    def test_postnatal_age_masked(self):
        result = _mask_nonchrono_age("Postnatal age ≥ 30 weeks Birth weight > 1800 g")
        assert "NONCHRONO" in result

    def test_n_weeks_gestational_age_masked(self):
        result = _mask_nonchrono_age("30 weeks gestational age and birth weight")
        assert "30" not in result or "NONCHRONO" in result

    def test_chronological_age_not_affected(self):
        text = "age ≥ 18 years"
        assert _mask_nonchrono_age(text) == text


# ── _trial_age_range ──────────────────────────────────────────────────────────

class TestTrialAgeRange:

    # ── Inclusion minimum ──────────────────────────────────────────────────────

    def test_inc_age_ge_18(self):
        assert _trial_age_range("Inclusion Criteria: age ≥ 18 years") == (18.0, None)

    def test_inc_age_ge_symbol(self):
        assert _trial_age_range("Inclusion Criteria: Age >= 18 years") == (18.0, None)

    def test_inc_age_plus(self):
        assert _trial_age_range("Inclusion Criteria: Allergic asthma age 12+") == (12.0, None)

    def test_inc_at_least(self):
        assert _trial_age_range("Inclusion Criteria: at least 18 years of age") == (18.0, None)

    def test_inc_n_years_or_older(self):
        assert _trial_age_range("Inclusion Criteria: 65 years or older") == (65.0, None)

    def test_inc_n_years_of_age_or_older(self):
        assert _trial_age_range("Inclusion Criteria: 18 years of age or older") == (18.0, None)

    def test_inc_age_gt_40_years(self):
        assert _trial_age_range(
            "Inclusion Criteria: Age > 40 years Syncope within the last 12 hours"
        ) == (40.0, None)

    def test_inc_age_gt_with_or_clause_skipped(self):
        # "Age > 16, or < 16 and accompanied" → no unambiguous minimum
        min_a, max_a = _trial_age_range(
            "Inclusion Criteria: Age > 16, or < 16 and accompanied by a legal guardian"
        )
        assert min_a is None

    # ── Inclusion maximum ──────────────────────────────────────────────────────

    def test_inc_age_lt_16_years(self):
        assert _trial_age_range("Inclusion Criteria: age < 16 years") == (None, 16.0)

    def test_inc_age_le_65_years(self):
        assert _trial_age_range("Inclusion Criteria: age ≤ 65 years") == (None, 65.0)

    # ── Exclusion minimum ──────────────────────────────────────────────────────

    def test_exc_under_18_years_old(self):
        doc = "Inclusion Criteria: Signed consent, Exclusion Criteria: patients under 18 years old"
        assert _trial_age_range(doc) == (18.0, None)

    def test_exc_less_than_50_years_old(self):
        # Topic 20155: 31yo vs trial excluding <50
        doc = (
            "Inclusion Criteria: Patients signing Informed Consent, "
            "Exclusion Criteria: Patients less than 50 years old"
        )
        assert _trial_age_range(doc) == (50.0, None)

    def test_exc_age_lt_18_years(self):
        doc = (
            "Inclusion Criteria: acute pulmonary embolism, "
            "Exclusion Criteria: Age < 18 years"
        )
        assert _trial_age_range(doc) == (18.0, None)

    # ── Exclusion maximum ──────────────────────────────────────────────────────

    def test_exc_over_21_years(self):
        doc = (
            "Inclusion Criteria: Patients under 21 years of age at risk of bone fragility, "
            "Exclusion Criteria: Age > 16 years"
        )
        _, max_a = _trial_age_range(doc)
        assert max_a == 16.0

    # ── Gestational age not parsed ─────────────────────────────────────────────

    def test_gestational_age_not_parsed_as_patient_min(self):
        # "gestational age ≥ 33 weeks" must NOT become min_age=33 years
        doc = (
            "Inclusion Criteria: Newborns with gestational age ≥ 33 weeks and "
            "neonatal hyperbilirubinemia requiring phototherapy"
        )
        min_a, max_a = _trial_age_range(doc)
        assert min_a is None or min_a < 2.0  # ≤ ~33 weeks in years is fine

    def test_at_least_n_weeks_gestational_age_not_min(self):
        doc = "Inclusion Criteria: at least 30 weeks gestational age and birth weight > 1200 g"
        min_a, _ = _trial_age_range(doc)
        assert min_a is None or min_a < 1.0

    # ── Symptoms duration not parsed ───────────────────────────────────────────

    def test_greater_than_7_days_symptoms_not_max_age(self):
        # "greater than 7 days" must not be parsed as max age
        doc = (
            "Inclusion Criteria: Chief complaint of acute appendicitis, "
            "Exclusion Criteria: Patients with symptoms greater than 7 days"
        )
        _, max_a = _trial_age_range(doc)
        assert max_a is None

    # ── Sanity check: min > max → (None, None) ─────────────────────────────────

    def test_min_gt_max_contradiction_returns_none_none(self):
        # Pediatric trial (max=20) + exclusion section accidentally sets min=40 → contradiction
        doc = (
            "Inclusion Criteria: age < 20 years (pediatric only), "
            "Exclusion Criteria: age < 40 years"
        )
        # inc gives max=20; exc gives min=40; min(40) > max(20) → discard both
        assert _trial_age_range(doc) == (None, None)

    def test_no_age_info_returns_none_none(self):
        doc = "Inclusion Criteria: Informed consent, Exclusion Criteria: Pregnancy"
        assert _trial_age_range(doc) == (None, None)

    # ── Months correctly converted ─────────────────────────────────────────────

    def test_age_ge_3_months_converts_to_fractional_years(self):
        doc = "Inclusion Criteria: age ≥ 3 months"
        min_a, _ = _trial_age_range(doc)
        assert min_a is not None
        assert abs(min_a - 3 / 12.0) < 1e-9


# ── _trial_excludes_age ────────────────────────────────────────────────────────

class TestTrialExcludesAge:

    # ── Below minimum ──────────────────────────────────────────────────────────

    def test_child_excluded_from_adult_trial(self):
        # Topic 201516: 4yo, trial requires age 12+
        doc = "Inclusion Criteria: Allergic asthma or allergic rhinitis age 12+, Exclusion Criteria: smokers"
        assert _trial_excludes_age(doc, patient_age=4.0) is True

    def test_teen_excluded_from_age_50_plus_trial(self):
        # Topic 20155: 31yo, trial excludes <50
        doc = (
            "Inclusion Criteria: Patients signing Informed Consent, "
            "Exclusion Criteria: Patients less than 50 years old"
        )
        assert _trial_excludes_age(doc, patient_age=31.0) is True

    def test_child_excluded_from_age_18_plus_trial(self):
        doc = "Inclusion Criteria: age ≥ 18 years infiltrate on chest X-ray"
        assert _trial_excludes_age(doc, patient_age=8.0) is True

    # ── Above maximum ──────────────────────────────────────────────────────────

    def test_adult_excluded_from_pediatric_trial(self):
        # 55yo from a trial for patients under 21
        doc = (
            "Inclusion Criteria: Patients under 21 years of age at risk of bone fragility, "
            "Exclusion Criteria: Age > 16 years"
        )
        assert _trial_excludes_age(doc, patient_age=55.0) is True

    # ── Within range → not excluded ────────────────────────────────────────────

    def test_adult_not_excluded_from_adult_trial(self):
        doc = "Inclusion Criteria: age ≥ 18 years, Exclusion Criteria: pregnancy"
        assert _trial_excludes_age(doc, patient_age=45.0) is False

    def test_child_not_excluded_from_pediatric_trial(self):
        doc = "Inclusion Criteria: age < 16 years, Exclusion Criteria: prior surgery"
        assert _trial_excludes_age(doc, patient_age=10.0) is False

    def test_patient_age_at_boundary_not_excluded(self):
        # 18yo with min_age=18 — the 0.5-year buffer means 18 ≥ 18-0.5 → not excluded
        doc = "Inclusion Criteria: age ≥ 18 years"
        assert _trial_excludes_age(doc, patient_age=18.0) is False

    # ── No age criteria → pass through ────────────────────────────────────────

    def test_no_age_in_doc_returns_false(self):
        doc = "Inclusion Criteria: confirmed diagnosis, written consent"
        assert _trial_excludes_age(doc, patient_age=55.0) is False

    # ── 0.5-year buffer ────────────────────────────────────────────────────────

    def test_buffer_prevents_exclusion_at_boundary(self):
        # min_age=18, patient_age=17.7 (> 18 - 0.5 = 17.5) → NOT excluded
        doc = "Inclusion Criteria: age ≥ 18 years"
        assert _trial_excludes_age(doc, patient_age=17.7) is False

    def test_clear_exclusion_below_buffer(self):
        # min_age=18, patient_age=15 (< 17.5) → excluded
        doc = "Inclusion Criteria: age ≥ 18 years"
        assert _trial_excludes_age(doc, patient_age=15.0) is True

    # ── Gestational age in trial must NOT exclude neonates ────────────────────

    def test_neonate_not_excluded_by_gestational_age_criterion(self):
        # 3-day-old neonate (pa≈0.008y) with "gestational age ≥ 33 weeks" criterion
        doc = (
            "Inclusion Criteria: Newborns with gestational age ≥ 33 weeks and "
            "neonatal hyperbilirubinemia requiring phototherapy"
        )
        assert _trial_excludes_age(doc, patient_age=3 / 365.0) is False
