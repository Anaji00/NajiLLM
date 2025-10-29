import os 
import torch
import torch.optim as optim
import madplotlib.pyplot as plt

from model.model_config import ModelConfig
from model.model import Transformer
from model.utils import count_parameters

DEVICE = torch.device("cpu")
SEQ_LEN = 256
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

CORPUS_IDS_PATH = "data/corpus_ids1.pt"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "stage1_general_pretrain.pt")

def make_batch_stream(id_tensor: torch.Tensor,
                      seq_len: int,
                      batch_size: int,
                      device: torch.device):
    """
    Convert a long 1D token-id tensor into shuffled minibatches of shape (B, seq_len).
    """
    total_tokens = id_tensor.size(0)
    num_sequences = total_tokens // seq_len

    trimmed = id_tensor[: num_sequences * seq_len] # (num_sequences * seq_len,) bevause we need full sequences
    data = trimmed.view(num_sequences, seq_len)  # (num_sequences, seq_len)

    indices = torch.randperm(num_sequences)

    for i in range(0, num_sequences, batch_size):
        batch_idx = indices[i:i+batch_size] # this loops through the shuffled indices in chunks of batch_size
        x = data[batch_idx].to(device) # This code will automatically handle the case when the last batch is smaller than batch_size
        y = x.clone()  # targets are the same sequence; model shifts internally
        yield x, y

# ----- load dataset -----
print(f"Loading tokenized corpus from {CORPUS_IDS_PATH} ...")
ids = torch.load(CORPUS_IDS_PATH)  # 1D tensor of token IDs
print("Total tokens in corpus:", ids.shape[0])
