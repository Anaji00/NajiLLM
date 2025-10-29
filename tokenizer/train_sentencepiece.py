import sentencepiece as spm
import os

os.makedirs("tokenizer", exist_ok=True)

input_file = "data/Wiki_Corpus.txt"   # general English corpus (Wikipedia-style, Gutenberg, etc.)
model_prefix = "tokenizer/spm"
vocab_size = 16000

spm.SentencePieceTrainer.Train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=0.9995,
    model_type="unigram",
    unk_id=0,
    pad_id=1,
    bos_id=2,
    eos_id=3,
)

print("Tokenizer trained and saved to tokenizer/spm.model and tokenizer/spm.vocab")
