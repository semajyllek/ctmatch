"""ctmatch.experiments — shared, portable experiment layer for ctmatch.

Single source of truth for paths, the document *representation*, the truncation
constant, and the small composable functions each notebook builds an experiment
from. This exists because scores were previously not interpretable: the same
model was fed different text across notebooks (eligibility-only vs full-text,
512 tok vs 1800 char, eligibility truncated off) — see deep_dive_outline.md §2g.

Design rules (keep them):
  * One `ExperimentConfig`. Every notebook constructs one and passes it down.
    Nothing reads a global path or magic number directly.
  * The representation is a *function of config*, not a pre-baked blob. Change
    `repr_strategy` / `max_length` in one place and every stage follows.
  * `cfg.repr_tag()` stamps every logged result, so a number can never again be
    ambiguous about what text produced it.
  * Heavy deps (torch/transformers/sentence-transformers/rank_bm25/lightgbm/
    pytrec_eval) are imported lazily inside functions, so this module imports
    with nothing installed and the pure logic is unit-testable.
  * Every function is small and side-effect-free where possible.

This ships INSIDE the ctmatch package, so notebooks get it from the same
`pip install git+https://github.com/semajyllek/ctmatch.git` they already run —
no manual Drive copy. Colab usage:
    from ctmatch.experiments import ExperimentConfig, load_corpus, encode_corpus, ...
    cfg = ExperimentConfig()          # or ExperimentConfig(max_length=384, repr_strategy='head_tail')
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass, field, asdict, replace
from typing import Callable, Iterable, Sequence

# Exclusion-header split, matching ctproc.eligibility.process_eligibility_naive (IGNORECASE).
_EXC_HEADER_RE = re.compile(r"(?:exclu(?:de|sion) criteria:?)|(?:ineligibility criteria:?)", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Config — the ONLY place paths, models, and the representation are defined.
# ─────────────────────────────────────────────────────────────────────────────

# Canonical field order of a trial document. Eligibility is intentionally listed
# so a strategy can move it; do not hard-code this order anywhere else.
# NOTE: these must match the keys written to doc_fulltext.jsonl by build_fulltext_corpus
# (parse_study) — 'detailed_desc' (not 'detailed_description'), else the field reads empty.
DEFAULT_DOC_FIELDS = (
    "brief_title",
    "official_title",
    "conditions",
    "brief_summary",
    "detailed_desc",
    "interventions",
    "eligibility",
)


@dataclass(frozen=True)
class ExperimentConfig:
    # --- storage ---------------------------------------------------------------
    data_root: str = "/content/drive/MyDrive/ct_data23"

    # --- corpus / eval data ----------------------------------------------------
    # doc_fulltext.jsonl: one JSON record per trial with the DEFAULT_DOC_FIELDS
    # keys (written by build_fulltext_corpus). index2docid.txt: id order.
    corpus_fields_file: str = "doc_fulltext.jsonl"
    index2docid_hf: str = "semaj83/ctmatch_ir"          # holds index2docid.txt
    trec_root: str = "evaluation/trec_data"
    kz_root: str = "evaluation/kz_data"

    # --- REPRESENTATION (the controlled variable) ------------------------------
    doc_fields: tuple = DEFAULT_DOC_FIELDS
    label_fields: bool = True          # prefix each field with "NAME: " in the blob
    # FROZEN by exp_truncation.ipynb (TREC21 judged-pool rerank NDCG@10, leg 2a/2b):
    # elig_first-L512 = 0.893 (winner) > elig_first-L384 0.877 > head_tail-L512 0.829 >
    # budget_incexc-L512 0.846 > head-L512 0.807. NDCG tracks eligibility coverage; the
    # eligibility-forward representation wins. Do NOT tune max_length again.
    repr_strategy: str = "elig_first"  # 'head' | 'head_tail' | 'elig_first' | 'budget_incexc'
    max_length: int = 512              # THE truncation constant (tokens), frozen.
    head_frac: float = 0.5             # head_tail only (unused by elig_first)
    retrieval_doc_chars: int = 0       # 0 = no char cap for BM25/dense blob; >0 caps it

    # budget_incexc: give topicality / inclusion / exclusion each a slice of the doc token
    # budget, reserving the EXCLUSION floor first (it's the disqualifier and the most-truncated).
    # inc/exc come from ctproc's process_eligibility_naive. inclusion gets whatever the other
    # two don't use.
    topicality_fields: tuple = ("conditions", "brief_title", "brief_summary")
    budget_topic_frac: float = 0.30
    budget_exc_frac: float = 0.25

    # Retrieval uses its OWN representation: recall wants TOPICALITY (does the doc name the
    # patient's condition?), not eligibility, and BM25 already indexes the whole doc — so the
    # dense budget should be topicality-dense, not a truncated full blob. Ablated by
    # exp_retrieval_repr.ipynb (field packing vs a longer-context encoder).
    retrieval_fields: tuple = DEFAULT_DOC_FIELDS   # order/subset that goes into the DENSE vector
    retriever_max_tokens: int = 256                # dense encoder context window (MiniLM-L6 = 256)

    # --- models -----------------------------------------------------------------
    # The binding fix is REPRESENTATION consistency (frozen R), NOT fine-tuning. Any
    # ckpt is valid here — off-the-shelf is a first-class option and the DEFAULT when it
    # ties/beats a fine-tune on clean R (simpler, no training). fine-tuned vs off-the-shelf
    # is an empirical ablation on R (the prior §7b retriever +18% and §11c judge-null were
    # on the confounded repr, so both are re-open). Swap freely, e.g.:
    #   retriever_ckpt = 'sentence-transformers/all-MiniLM-L6-v2'  # off-the-shelf
    #   llm_ckpt       = data_root + '/judge_lora_R'               # fine-tuned judge
    # clf is now the frozen-R model (clf_R); retriever/reranker still pre-R pending their retrain.
    # _resolve_ckpt prefers a local dir if it exists, so 'models/clf_R' on Drive works too.
    retriever_ckpt: str = "semaj83/ctmatch-retriever-v2"
    clf_ckpt: str = "semaj83/ctmatch-clf-R"               # eligibility-view clf (elig_first); PUSH from Colab
    clf_topic_ckpt: str = "semaj83/ctmatch-clf-topic"    # topicality-view clf (topic_first) — diverse reranker
    reranker_ckpt: str = "models/reranker_v3"              # frozen-R reranker (local dir on Drive)
    llm_ckpt: str = "Qwen/Qwen2.5-7B-Instruct"             # zero-shot judge by default
    llm_max_tokens: int = 2048   # judge context budget — the LLM has 32k context, so it must NOT
                                 # be gated by the 512 cross-encoder max_length (§7f used 2048)

    # --- retrieval / ensemble --------------------------------------------------
    cand_k: int = 1000
    llm_top_k: int = 500
    llm_floor: float = -36.0
    rrf_k: int = 60
    seed: int = 42

    # --- bookkeeping -----------------------------------------------------------
    results_file: str = "results/exp_results.jsonl"

    # -- derived paths ----------------------------------------------------------
    def path(self, *parts: str) -> str:
        return os.path.join(self.data_root, *parts)

    @property
    def corpus_fields_path(self) -> str:
        return self.path(self.corpus_fields_file)

    @property
    def results_path(self) -> str:
        return self.path(self.results_file)

    def emb_file(self, tag: str) -> str:
        """Per-representation embedding cache — never let two reprs collide."""
        return self.path(f"cache/doc_emb_{self.retriever_ckpt.split('/')[-1]}_{tag}.npy")

    def bm25_file(self, tag: str) -> str:
        return self.path(f"cache/bm25_{tag}.pkl")

    # -- identity ---------------------------------------------------------------
    def repr_tag(self) -> str:
        """Stamps every result. `head_tail-L512-h50` etc. Two runs with the same
        tag saw identical text; different tags are not comparable."""
        base = f"{self.repr_strategy}-L{self.max_length}"
        if self.repr_strategy == "head_tail":
            base += f"-h{int(self.head_frac * 100)}"
        if self.repr_strategy == "budget_incexc":
            base += f"-t{int(self.budget_topic_frac * 100)}e{int(self.budget_exc_frac * 100)}"
        if self.retrieval_doc_chars:
            base += f"-c{self.retrieval_doc_chars}"
        return base

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    def with_(self, **kw) -> "ExperimentConfig":
        """Return a copy with overrides — the sweep primitive."""
        return replace(self, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Representation — build the doc text and the (topic, doc) model input.
# This is the fix for §2g: one place decides what text the models see.
# ─────────────────────────────────────────────────────────────────────────────

def _field_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if v)
    return str(value)


def doc_segments(fields: dict, cfg: ExperimentConfig) -> list[tuple[str, str]]:
    """Ordered [(label, text)] for the configured fields, non-empty only.

    `elig_first` moves eligibility (and conditions) to the front so plain
    longest-first truncation keeps the eligibility signal; `topic_first` moves the
    topicality fields to the front (the topicality *view* — a deliberately different
    representation for a diverse reranker, orthogonal to the eligibility view). Other
    strategies keep the natural order and rely on head/head_tail token selection.
    """
    order = list(cfg.doc_fields)
    if cfg.repr_strategy == "elig_first":
        front = [f for f in ("eligibility", "conditions") if f in order]
        order = front + [f for f in order if f not in front]
    elif cfg.repr_strategy == "topic_first":
        front = [f for f in cfg.topicality_fields if f in order]
        order = front + [f for f in order if f not in front]
    segs = []
    for name in order:
        txt = _field_text(fields.get(name)).strip()
        if txt:
            segs.append((name, txt))
    return segs


def doc_blob(fields: dict, cfg: ExperimentConfig) -> str:
    """Single string for BM25 / dense encoding (retrieval sees the whole doc;
    truncation there is the encoder's own max_seq_length, handled at encode time)."""
    segs = doc_segments(fields, cfg)
    if cfg.label_fields:
        blob = "\n".join(f"{name.replace('_', ' ').upper()}: {txt}" for name, txt in segs)
    else:
        blob = "\n".join(txt for _, txt in segs)
    if cfg.retrieval_doc_chars:
        blob = blob[: cfg.retrieval_doc_chars]
    return blob


def _blob_from(fields: dict, order: Sequence[str], cfg: ExperimentConfig) -> str:
    segs = [(n, _field_text(fields.get(n)).strip()) for n in order]
    segs = [(n, t) for n, t in segs if t]
    if cfg.label_fields:
        return "\n".join(f"{n.replace('_', ' ').upper()}: {t}" for n, t in segs)
    return "\n".join(t for _, t in segs)


def full_blob(fields: dict, cfg: ExperimentConfig) -> str:
    """All fields, natural order, no token limit — for BM25 (not context-limited, so it
    should see the whole document; field order is irrelevant to a bag-of-words model)."""
    return _blob_from(fields, cfg.doc_fields, cfg)


def retrieval_blob(fields: dict, cfg: ExperimentConfig) -> str:
    """DENSE-retrieval representation: `cfg.retrieval_fields` (order + subset). The encoder
    truncates to `cfg.retriever_max_tokens`, so lead with topicality (conditions/title/
    summary) and drop low-value-per-token fields — BM25 covers the rest in the hybrid."""
    return _blob_from(fields, cfg.retrieval_fields, cfg)


def head_tail_ids(doc_ids: list[int], budget: int, head_frac: float) -> list[int]:
    """Keep the first `head` and last `tail` doc tokens so a trailing field
    (eligibility) survives. Budget is the doc-side token allowance."""
    if budget <= 0 or len(doc_ids) <= budget:
        return doc_ids[:budget] if budget > 0 else doc_ids
    head = int(round(budget * head_frac))
    tail = budget - head
    if tail <= 0:
        return doc_ids[:budget]
    return doc_ids[:head] + doc_ids[-tail:]


def split_eligibility(fields: dict, cfg: ExperimentConfig) -> tuple[str, str]:
    """(inclusion_text, exclusion_text). Uses precomputed 'inclusion'/'exclusion' fields if
    the corpus carries them (cache), else ctproc's tested splitter on the raw eligibility text."""
    if fields.get("inclusion") is not None or fields.get("exclusion") is not None:
        return _field_text(fields.get("inclusion")), _field_text(fields.get("exclusion"))
    from ctproc.eligibility import process_eligibility_naive   # ctproc is a ctmatch dependency
    inc, exc = process_eligibility_naive(_field_text(fields.get("eligibility", "")))
    return " ".join(inc), " ".join(exc)


def _cap_tokens(tokenizer, text: str, n: int) -> tuple[str, int]:
    if n <= 0 or not text:
        return "", 0
    ids = tokenizer.encode(text, add_special_tokens=False)[:n]
    return tokenizer.decode(ids), len(ids)


def _budget_incexc_text(tokenizer, fields: dict, cfg: ExperimentConfig, budget: int) -> str:
    """topicality + inclusion + exclusion, each given a slice of the doc token budget.
    Exclusion's floor is reserved FIRST (disqualifier, most-truncated); topicality is capped;
    inclusion fills whatever remains."""
    topical = _blob_from(fields, cfg.topicality_fields, cfg)
    inc_text, exc_text = split_eligibility(fields, cfg)
    t_txt, t_used = _cap_tokens(tokenizer, topical, int(budget * cfg.budget_topic_frac))
    e_txt, e_used = _cap_tokens(tokenizer, exc_text, int(budget * cfg.budget_exc_frac))
    i_txt, _ = _cap_tokens(tokenizer, inc_text, budget - t_used - e_used)
    parts = [p for p in (t_txt, f"INCLUSION: {i_txt}" if i_txt else "",
                         f"EXCLUSION: {e_txt}" if e_txt else "") if p]
    return "\n".join(parts)


def truncated_doc_text(tokenizer, fields: dict, cfg: ExperimentConfig, reserve: int,
                       max_length: int | None = None) -> str:
    """Doc text truncated per cfg.repr_strategy to fit `max_length - reserve` tokens (defaults to
    the cross-encoder `cfg.max_length`; the LLM judge passes its own larger `cfg.llm_max_tokens`),
    returned as a STRING so the caller can let the model's own tokenizer assemble the pair
    (correct [CLS]/[SEP] and token_type_ids — hand-rolling those silently degrades a BERT
    cross-encoder). Strategies:
      'head'         → leading budget tokens (drops eligibility on long docs).
      'head_tail'    → head_frac of the budget as head + the rest as tail (keeps trailing eligibility).
      'elig_first'   → leading budget, but doc_segments already moved eligibility to the front.
      'budget_incexc'→ topicality/inclusion/exclusion each budgeted, exclusion floor reserved first.
    """
    budget = max((cfg.max_length if max_length is None else max_length) - reserve, 0)
    if cfg.repr_strategy == "budget_incexc":
        return _budget_incexc_text(tokenizer, fields, cfg, budget)
    ids = tokenizer.encode(doc_blob(fields, cfg), add_special_tokens=False)
    if cfg.repr_strategy == "head_tail":
        ids = head_tail_ids(ids, budget, cfg.head_frac)
    else:
        ids = ids[:budget]
    return tokenizer.decode(ids)


def encode_cross_batch(tokenizer, topic: str, docs_fields: Sequence[dict], cfg: ExperimentConfig):
    """Tokenize (topic, doc) pairs for a cross-encoder under cfg.repr_strategy so eligibility
    is not silently dropped. Pre-truncates each doc per strategy, then lets the tokenizer build
    the pair — so special tokens AND token_type_ids match how the model was trained. Returns a
    padded BatchEncoding (input_ids / attention_mask / token_type_ids)."""
    reserve = 3 + len(tokenizer.encode(topic, add_special_tokens=False))  # [CLS] topic [SEP] ... [SEP]
    docs = [truncated_doc_text(tokenizer, f, cfg, reserve) for f in docs_fields]
    return tokenizer([topic] * len(docs), docs, truncation="longest_first",
                     max_length=cfg.max_length, padding=True, return_tensors="pt")


def relevant_index(model, label: str = "relevant") -> int:
    """Index of the fully-relevant class. EXACT match — NOT substring: `'relevant' in
    'not_relevant'` (and `'irrelevant'`) is True, so a substring match silently selects the
    *not-relevant* class and ranks everything backwards. Falls back to the highest label id
    (the not/partial/relevant = 0/1/2 convention) when labels are generic `LABEL_x`."""
    id2label = model.config.id2label
    for k, v in id2label.items():
        if str(v).lower() == label:
            return int(k)
    return max(int(k) for k in id2label)


def _kept_doc_indices(tokenizer, topic: str, fields: dict, cfg: ExperimentConfig):
    """Set of doc-token indices (into doc_blob) that survive into the model input, plus n_doc.
    For the blob strategies only (budget_incexc assembles text differently)."""
    n_topic = len(tokenizer.encode(topic, add_special_tokens=False))
    budget = max(cfg.max_length - (3 + n_topic), 0)
    n_doc = len(tokenizer.encode(doc_blob(fields, cfg), add_special_tokens=False))
    if n_doc <= budget:
        return set(range(n_doc)), n_doc
    if cfg.repr_strategy == "head_tail":
        head = int(round(budget * cfg.head_frac))
        tail = budget - head
        return set(range(head)) | set(range(n_doc - tail, n_doc)), n_doc
    return set(range(budget)), n_doc   # 'head' / 'elig_first' keep the leading budget


def _span_retained(tokenizer, topic: str, fields: dict, cfg: ExperimentConfig, span_text: str) -> float:
    """Fraction of `span_text` (a substring of doc_blob) whose tokens survive truncation."""
    if not span_text.strip():
        return 1.0
    blob = doc_blob(fields, cfg)
    ci = blob.find(span_text)
    if ci < 0:
        return 1.0
    start = len(tokenizer.encode(blob[:ci], add_special_tokens=False))
    n = len(tokenizer.encode(span_text, add_special_tokens=False))
    if n == 0:
        return 1.0
    kept, _ = _kept_doc_indices(tokenizer, topic, fields, cfg)
    return len(set(range(start, start + n)) & kept) / n


def eligibility_retained_frac(tokenizer, topic: str, fields: dict, cfg: ExperimentConfig) -> float:
    """Model-free diagnostic: fraction of the eligibility field's tokens that survive into the
    (topic, doc) model input under cfg. Evaluates the truncation constant by itself."""
    return _span_retained(tokenizer, topic, fields, cfg, _field_text(fields.get("eligibility", "")))


def exclusion_retained_frac(tokenizer, topic: str, fields: dict, cfg: ExperimentConfig) -> float:
    """Fraction of the EXCLUSION criteria tokens that survive — the disqualifier signal, which
    elig_first/head keep *after* inclusion (so a long inclusion list buries it) but budget_incexc
    reserves a floor for. Returns 1.0 when there is no exclusion text."""
    _, exc_text = split_eligibility(fields, cfg)
    n_exc = len(tokenizer.encode(exc_text, add_special_tokens=False))
    if n_exc == 0:
        return 1.0
    if cfg.repr_strategy == "budget_incexc":
        n_topic = len(tokenizer.encode(topic, add_special_tokens=False))
        budget = max(cfg.max_length - (3 + n_topic), 0)
        return min(n_exc, int(budget * cfg.budget_exc_frac)) / n_exc
    # blob strategies: locate the raw exclusion block (from the exclusion header) inside doc_blob
    elig_raw = _field_text(fields.get("eligibility", ""))
    m = _EXC_HEADER_RE.search(elig_raw)
    exc_raw = elig_raw[m.start():] if m else ""
    return _span_retained(tokenizer, topic, fields, cfg, exc_raw)


def llm_prompt(tokenizer, topic: str, fields: dict, cfg: ExperimentConfig) -> str:
    """Judge prompt using the same repr STRATEGY as the cross-encoders (so the judge sees
    eligibility) but the LLM's own, much larger token budget (`cfg.llm_max_tokens`, not the 512
    BERT `max_length`). Reserve covers the system+instruction+patient+chat-template scaffold so the
    'yes/no' question is never truncated off."""
    trial = truncated_doc_text(tokenizer, fields, cfg, reserve=300, max_length=cfg.llm_max_tokens)
    user = (
        "You are a clinical trial matching expert.\n\n"
        f"Patient:\n{topic}\n\n"
        f"Trial:\n{trial}\n\n"
        "Is this patient likely eligible for this trial? Answer with a single word: yes or no."
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def llm_topicality_prompt(tokenizer, topic: str, fields: dict, cfg: ExperimentConfig) -> str:
    """TOPICALITY (not eligibility) prompt: is this trial *about* the patient's condition? Uses only
    the topicality fields (conditions/title/summary) — deliberately orthogonal to the eligibility
    judge (§9e #2), so the ensemble gets a signal it can't get from the eligibility readers."""
    ids = tokenizer.encode(_blob_from(fields, cfg.topicality_fields, cfg), add_special_tokens=False)
    trial = tokenizer.decode(ids[: max(cfg.llm_max_tokens - 300, 0)])
    user = (
        "You are a clinical trial matching expert.\n\n"
        f"Patient:\n{topic}\n\n"
        f"Trial:\n{trial}\n\n"
        "Is this trial studying the patient's condition or disease? Answer with a single word: yes or no."
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(cfg: ExperimentConfig) -> tuple[list[str], list[dict]]:
    """Return (corpus_ids, corpus_fields) aligned to index2docid order.
    corpus_fields[i] is the raw field dict for building any representation."""
    from datasets import load_dataset

    idx = load_dataset(cfg.index2docid_hf, data_files="index2docid.txt", split="train")
    corpus_ids = [r["text"].strip() for r in idx]
    id2fields = {}
    with open(cfg.corpus_fields_path) as f:
        for line in f:
            rec = json.loads(line)
            id2fields[rec.get("nct_id") or rec.get("doc_id")] = rec
    corpus_fields = [id2fields.get(d, {}) for d in corpus_ids]
    return corpus_ids, corpus_fields


def clf_pairs(cfg: ExperimentConfig, sets: Sequence[str], id_filter: set | None = None):
    """Labeled (topic, doc) training pairs from the judged qrels of `sets` — the whole judged
    pool per topic (all rels, mostly 0). Each pair is {source, topic_id, topic_text, doc_id,
    label}. NO doc text baked in — the representation is applied at tokenize time from cfg, so
    it can never drift from what the model is evaluated on (§2g). `id_filter` keeps only docs
    present in the corpus."""
    out = []
    for name, ds in load_eval(cfg, sets).items():
        rel, topic2text = ds["rel_dict"], ds["topic2text"]
        for tid, d2r in rel.items():
            if tid not in topic2text:
                continue
            for doc_id, label in d2r.items():
                if id_filter is not None and doc_id not in id_filter:
                    continue
                out.append({"source": name, "topic_id": tid, "topic_text": topic2text[tid],
                            "doc_id": doc_id, "label": int(label)})
    return out


def build_clf_dataset(pairs: Sequence[dict], id2fields: dict, tokenizer, cfg: ExperimentConfig):
    """HF Dataset of tokenized (topic, doc) pairs + int labels, with the doc rendered in the
    FROZEN representation (cfg) via truncated_doc_text — the same path the reranker is scored on.
    Dynamic padding: pad at collate time, not here."""
    from datasets import Dataset

    def gen():
        for p in pairs:
            fields = id2fields.get(p["doc_id"], {})
            reserve = 3 + len(tokenizer.encode(p["topic_text"], add_special_tokens=False))
            doc = truncated_doc_text(tokenizer, fields, cfg, reserve)
            enc = tokenizer(p["topic_text"], doc, truncation="longest_first", max_length=cfg.max_length)
            enc["labels"] = int(p["label"])
            yield enc

    return Dataset.from_generator(gen)


def mine_dense_hard_negatives(cfg: ExperimentConfig, pairs: Sequence[dict], id2fields: dict,
                              n_per_topic: int = 50, batch_size: int = 64):
    """Per topic, the rel=0 judged docs the DENSE retriever ranks MOST similar to the topic — the
    semantic near-misses a pointwise cross-encoder tends to score as false positives (§7i Finding 4).
    Returns those pairs, to be oversampled into reranker training. Uses cfg.retriever_ckpt on the
    retrieval representation."""
    import numpy as np
    from collections import defaultdict
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_resolve_ckpt(cfg, cfg.retriever_ckpt))
    model.max_seq_length = cfg.retriever_max_tokens
    uniq = sorted({p["doc_id"] for p in pairs})
    demb = model.encode([retrieval_blob(id2fields.get(d, {}), cfg) for d in uniq],
                        convert_to_numpy=True, normalize_embeddings=True,
                        batch_size=batch_size, show_progress_bar=True).astype("float32")
    didx = {d: i for i, d in enumerate(uniq)}

    by_topic = defaultdict(list)
    for p in pairs:
        by_topic[p["topic_id"]].append(p)
    hard = []
    for group in by_topic.values():
        negs = [p for p in group if int(p["label"]) == 0]
        if not negs:
            continue
        q = model.encode([group[0]["topic_text"]], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = [(p, float(q @ demb[didx[p["doc_id"]]])) for p in negs]
        hard.extend(p for p, _ in sorted(sims, key=lambda x: -x[1])[:n_per_topic])
    return hard


def load_eval(cfg: ExperimentConfig, sets: Sequence[str]):
    """Thin wrapper over the repo loader so notebooks don't reimplement it."""
    from ctmatch.evaluation.eval_utils import load_eval_datasets

    os.environ["CTMATCH_DATA_ROOT"] = cfg.data_root
    all_sets = load_eval_datasets(cfg.path(cfg.trec_root), cfg.path(cfg.kz_root))
    return {k: v for k, v in all_sets.items() if k in sets}


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25(corpus_fields: Sequence[dict], cfg: ExperimentConfig):
    from rank_bm25 import BM25Okapi

    tokenized = [full_blob(f, cfg).lower().split() for f in corpus_fields]   # BM25 sees the whole doc
    return BM25Okapi(tokenized)


def encode_corpus(corpus_fields: Sequence[dict], cfg: ExperimentConfig, batch_size: int = 64):
    """Dense-encode the corpus with `cfg.retriever_ckpt` on the RETRIEVAL representation,
    truncated to `cfg.retriever_max_tokens`. (encode the query side with the same model +
    window at query time.)"""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_resolve_ckpt(cfg, cfg.retriever_ckpt))
    model.max_seq_length = cfg.retriever_max_tokens
    blobs = [retrieval_blob(f, cfg) for f in corpus_fields]
    emb = model.encode(blobs, convert_to_numpy=True, normalize_embeddings=True,
                       batch_size=batch_size, show_progress_bar=True).astype(np.float32)
    return emb


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Scoring (cross-encoders, LLM judge) — all through the one representation
# ─────────────────────────────────────────────────────────────────────────────

def cross_encoder_scores(model, tokenizer, topic: str, docs_fields, cfg, rel_idx: int, batch: int = 64):
    import torch
    import torch.nn.functional as F

    out = []
    device = next(model.parameters()).device
    for i in range(0, len(docs_fields), batch):
        enc = encode_cross_batch(tokenizer, topic, docs_fields[i:i + batch], cfg)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            p = F.softmax(model(**enc).logits, dim=-1)
        out.extend(p[:, rel_idx].cpu().tolist())
    return out


def llm_yesno_scores(model, tokenizer, topic: str, docs_fields, cfg, batch: int = 8, prompt_fn=None):
    """yes/no logprob margin per (topic, doc). `prompt_fn` builds each prompt — defaults to the
    eligibility judge (`llm_prompt`); pass `llm_topicality_prompt` for the topicality feature."""
    import torch

    prompt_fn = prompt_fn or llm_prompt
    yes = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in ["yes", "Yes", " yes", " Yes", "YES"]})
    no = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in ["no", "No", " no", " No", "NO"]})
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(docs_fields), batch):
        prompts = [prompt_fn(tokenizer, topic, f, cfg) for f in docs_fields[i:i + batch]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=cfg.llm_max_tokens).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
            lp = torch.log_softmax(logits.float(), dim=-1)
        y = torch.logsumexp(lp[:, yes], dim=-1)
        n = torch.logsumexp(lp[:, no], dim=-1)
        out.extend((y - n).cpu().tolist())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation + result logging
# ─────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_ids: Sequence[str], rel: dict, k: int = 10) -> float:
    import numpy as np

    dcg = sum((2 ** rel.get(d, 0) - 1) / np.log2(i + 2) for i, d in enumerate(ranked_ids[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: Sequence[str], rel: dict, k: int, rel_level: int = 1) -> float:
    """Fraction of judged-relevant docs (rel >= rel_level) retrieved in the top-k. The
    retrieval metric — pool membership — not NDCG. rel_level=1 counts Excluded+Eligible;
    rel_level=2 counts Eligible only."""
    rel_docs = {d for d, r in rel.items() if r >= rel_level}
    if not rel_docs:
        return 0.0
    return len(set(ranked_ids[:k]) & rel_docs) / len(rel_docs)


def write_trec_run(path: str, run: dict, tag: str) -> None:
    with open(path, "w") as f:
        for qid, d2s in run.items():
            for rank, (doc, score) in enumerate(sorted(d2s.items(), key=lambda x: -x[1]), 1):
                f.write(f"{qid} Q0 {doc} {rank} {score:.6f} {tag}\n")


def pytrec_ndcg(run: dict, qrels: dict, k: int = 10) -> float:
    import pytrec_eval

    ev = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}"})
    res = ev.evaluate(run)
    vals = [v[f"ndcg_cut_{k}"] for v in res.values()]
    return sum(vals) / len(vals) if vals else 0.0


def pytrec_metrics(run: dict, qrels: dict, k: int = 10) -> dict:
    """TREC CT protocol metrics: NDCG@k on the GRADED qrels (Eligible=2/Excluded=1/Not=0),
    but P@k and MRR on ELIGIBLE-ONLY (rel>=2) relevance — matching the overview so the numbers
    are comparable to h2oloo. Returns {ndcg@k, P@k, mrr}."""
    import numpy as np
    import pytrec_eval

    ndcg = pytrec_ndcg(run, qrels, k)
    elig = {t: {d: (1 if r >= 2 else 0) for d, r in dd.items()} for t, dd in qrels.items()}
    res = pytrec_eval.RelevanceEvaluator(elig, {f"P.{k}", "recip_rank"}).evaluate(run)
    p = float(np.mean([v[f"P_{k}"] for v in res.values()])) if res else 0.0
    mrr = float(np.mean([v["recip_rank"] for v in res.values()])) if res else 0.0
    return {f"ndcg@{k}": round(ndcg, 4), f"P@{k}": round(p, 4), "mrr": round(mrr, 4)}


def log_result(cfg: ExperimentConfig, experiment: str, split: str, metrics: dict, extra: dict | None = None) -> None:
    """Append one line so every number is stored WITH its representation tag and
    the full config fingerprint. This is the anti-drift ledger."""
    os.makedirs(os.path.dirname(cfg.results_path), exist_ok=True)
    row = {
        "experiment": experiment,
        "split": split,
        "repr_tag": cfg.repr_tag(),
        "fingerprint": cfg.fingerprint(),
        "max_length": cfg.max_length,
        "repr_strategy": cfg.repr_strategy,
        "metrics": metrics,
        "config": asdict(cfg),
    }
    if extra:
        row["extra"] = extra
    with open(cfg.results_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def resolve_ckpt(cfg: ExperimentConfig, ckpt: str) -> str:
    """Resolve a checkpoint to a `from_pretrained`-ready value: a local path (absolute or
    data_root-relative) if one exists on disk, otherwise an HF hub id. (Checking existence first
    means a local model dir like 'models/reranker_v3' — which contains a '/' — isn't mistaken for
    an HF id.) Use this whenever loading a model from cfg, so local frozen-R dirs work."""
    if os.path.exists(ckpt):
        return ckpt
    local = cfg.path(ckpt)
    return local if os.path.exists(local) else ckpt


_resolve_ckpt = resolve_ckpt   # internal alias (encode_corpus / mine_dense_hard_negatives)
