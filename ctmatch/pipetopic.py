
from typing import Any, NamedTuple

from .modelconfig import ModelConfig

class PipeTopic(NamedTuple):
    topic_text: str
    embedding_vec: Any 
    category_vec: Any
  


