import os
from pathlib import Path
from typing import List, Tuple


def _data_root() -> Path:
    root = os.environ.get("CTMATCH_DATA_ROOT")
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[3] / "data"


def _d(relpath: str) -> str:
    return str(_data_root() / relpath)


TREC_REL_PATH = _d("trec_data/trec_21_judgments.txt")
KZ_REL_PATH = _d("kz_data/qrels-clinical_trials.txt")

TREC_RELLED_TOPIC_PATH = _d("trec_data/processed_trec_data/processed_trec21_topics.jsonl")
KZ_RELLED_TOPIC_PATH = _d("kz_data/processed_kz_data/processed_kz_topics.jsonl")

KZ_DOC_PATH = _d("kz_data/clinicaltrials.gov-16_dec_2015.zip")
KZ_PROCESSED_DOC_PATH = _d("kz_data/processed_kz_data/processed_kz_docs.jsonl")

TREC_ML_PATH = _d("trec_data/trec_data.jsonl")
KZ_ML_PATH = _d("kz_data/kz_data.jsonl")


def get_data_tuples(trec_or_kz: str = 'trec') -> List[Tuple[str, str]]:
    if trec_or_kz == 'trec':
        return get_trec_doc_data_tuples(), get_trec_topic_data_tuples()
    return get_kz_doc_data_tuples(), get_kz_topic_data_tuples()


def get_trec_doc_data_tuples() -> List[Tuple[str, str]]:
    pairs = []
    for i in range(1, 6):
        src = _d(f"trec_data/trec_docs_21/ClinicalTrials.2021-04-27.part{i}.zip")
        tgt = _d(f"trec_data/processed_trec_data/processed_trec22_docs_part{i}.jsonl")
        pairs.append((src, tgt))
    return pairs


def get_trec_topic_data_tuples() -> List[Tuple[str, str]]:
    return [
        (_d("trec_data/trec_21_topics.xml"), _d("trec_data/processed_trec_data/processed_trec21_topics.jsonl")),
        (_d("trec_data/trec_22_topics.xml"), _d("trec_data/processed_trec_data/processed_trec22_topics.jsonl")),
    ]


def get_kz_doc_data_tuples() -> List[Tuple[str, str]]:
    return [(KZ_DOC_PATH, KZ_PROCESSED_DOC_PATH)]


def get_kz_topic_data_tuples() -> List[Tuple[str, str]]:
    return [
        (_d("kz_data/topics-2014_2015-description.topics"), _d("kz_data/processed_kz_data/processed_kz_topics.jsonl")),
    ]
