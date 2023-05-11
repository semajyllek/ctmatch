
from typing import Dict, List, NamedTuple, Optional
from pathlib import Path

class ModelConfig(NamedTuple):
    name: str = 'scibert_finetuned_ctmatch'
    classifier_model_checkpoint: str = 'semaj83/scibert_finetuned_ctmatch'
    max_length: int = 512
    padding: str = True
    truncation: bool = True
    batch_size: int = 16
    learning_rate: float = 2e-5
    train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    seed: int  = 42
    splits: Dict[str, float] = {"train":0.8, "val":0.1}
    classifier_data_path: Path = Path("combined_classifier_data.jsonl")
    output_dir: Optional[str] = None
    convert_snli: bool = False
    use_trainer: bool = True
    num_classes: int = 3
    fp16: bool = False
    early_stopping: bool = False
    push_to_hub: bool = False
    ir_save_path: Optional[str] = None
    category_path: Optional[str] = None
    processed_data_paths: Optional[List[str]] = None
    max_query_length: int = 1200
    category_model_checkpoint: str = "facebook/bart-large-mnli"
    embedding_model_checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2"
    gen_model_checkpoint: str = 'text-davinci-003'
    openai_api_key: Optional[str] = None
    ir_setup: bool = False                  # if true, use the IR model setup, no classifier training or dataprep
    mode: str = 'normal'                    # normal or eval


