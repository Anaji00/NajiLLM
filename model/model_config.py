# Model Configuration File

from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 16000

    # Transfromer parameters
    n_layers: int = 8           # Number of transformer layers
    n_heads: int = 8            # Number of attention heads
    d_model: int = 256          # hidden size (Embedding dimension)
    d_ff: int = 1024            # Feedforward network dimension

    # Context Window
    context_window: int = 512    # Maximum sequence length

    dropout_rate: float = 0.1    # Dropout rate for regularization
    