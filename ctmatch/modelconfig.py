
from typing import Dict, List, NamedTuple, Optional
from pathlib import Path

class ModelConfig(NamedTuple):
    name: str
    model_checkpoint: str
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
    classified_data_path: Path = "semaj83/ctmatch/combined_classifier_data.jsonl"
    output_dir: Optional[str] = None
    convert_snli: bool = False
    use_trainer: bool = True
    gen_model: str = 'biogpt'
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    num_classes: int = 3
    fp16: bool = False
    early_stopping: bool = False
    push_to_hub: bool = False
    ir_save_path: Optional[str] = None
    category_path: Optional[str] = None
    processed_data_paths: Optional[List[str]] = None


