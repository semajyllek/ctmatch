
from typing import List, Optional
import re

from ...config import PipeConfig


class GenModel:
    def __init__(self, pipe_config: PipeConfig) -> None:
        self.pipe_config = pipe_config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic()
            except ImportError:
                raise ImportError(
                    "anthropic package required for gen filter. "
                    "Install with: pip install ctmatch[claude]"
                )
        return self._client

    def gen_response(self, query_prompt: str, doc_set: Optional[List[int]] = None) -> List[int]:
        client = self._get_client()
        message = client.messages.create(
            model=self.pipe_config.gen_model_checkpoint,
            max_tokens=512,
            messages=[{"role": "user", "content": query_prompt}],
        )
        text = message.content[0].text
        return self._parse_response(text, doc_set)

    def _parse_response(self, text: str, doc_set: Optional[List[int]]) -> List[int]:
        ranking = [int(s) for s in re.findall(r'\b(\d+)\b', text)]
        if doc_set is not None:
            for ncid in doc_set:
                if ncid not in ranking:
                    ranking.append(ncid)
        return ranking
