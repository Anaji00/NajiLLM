import sentencepiece as spm
import torch

class SPTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text: str) -> torch.Tensor:
        ids = self.sp.encode(text, out_type=int)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp.decode(ids)
