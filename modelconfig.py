
from typing import Dict, NamedTuple, Optional
from pathlib import Path

class ModelConfig(NamedTuple):
    data_path: Path
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
    output_dir: Optional[str] = None
    convert_snli: bool = False
    use_trainer: bool = True
    gen_model: str = None
    num_labels: int = 3

