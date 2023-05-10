
from typing import Dict, List, NamedTuple, Optional
from pathlib import Path

class ModelConfig(NamedTuple):
    name: str
    classifier_model_checkpoint: str
    max_length: int
    padding: str
    truncation: bool
    batch_size: int
    learning_rate: float
    train_epochs: int
    weight_decay: float
    warmup_steps: int
    seed: int
    splits: Dict[str, float]
    classifier_data_path: Path = "combined_classifier_data.jsonl"
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
    ir_setup: bool = False
    open_api_key: Optional[str] = None
    category_model_checkpoint: str = "facebook/bart-large-mnli"
    embedding_model_checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2"
    gen_model_checkpoint: str = 'text-davinci-003'


