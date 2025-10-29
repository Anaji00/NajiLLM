import os
import torch
from tokenizer.sp_tokenizer import SPTokenizer

# -------- settings --------
TOKENIZER_PATH = "tokenizer/spm.model"   # already trained SentencePiece model
RAW_TEXT_PATH  = "data/Wiki_Corpus.txt"   # your combined general corpus text file
OUT_IDS_PATH   = "data/corpus_ids.pt"    # output tensor of token IDs

os.makedirs("data", exist_ok=True)

# -------- load tokenizer --------
tok = SPTokenizer(TOKENIZER_PATH)

# -------- read raw text --------
with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# -------- encode entire corpus to token IDs --------
ids = tok.encode(text)  # torch.Tensor([id, id, id, ...])
print("Token count:", ids.shape[0])

# -------- save token IDs --------
torch.save(ids, OUT_IDS_PATH)
print(f"Saved -> {OUT_IDS_PATH}")
