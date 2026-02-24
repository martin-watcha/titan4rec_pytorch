from dataclasses import dataclass, field


@dataclass
class DataConfig:
    dataset: str = "ml-1m"  # ml-100k, ml-1m, ml-10m, ml-20m, ml-25m
    max_len: int = 200
    num_workers: int = 4


@dataclass
class TrainConfig:
    num_epochs: int = 200
    batch_size: int = 128
    num_batches_per_epoch: int = 1000
    patience: int = 5
    min_delta: float = 0.0
    seed: int = 42
    lr: float = 0.001
    weight_decay: float = 0.0
    betas: tuple = field(default_factory=lambda: (0.9, 0.98))
    accelerator: str = "auto"
    gradient_clip_val: float = 1.0
    use_wandb: bool = False
    wandb_project: str = "titan4rec"


@dataclass
class Titan4RecConfig:
    d_model: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    segment_size: int = 20
    memory_depth: int = 2
    memory_heads: int = 2
    expansion_factor: int = 4
    num_persistent: int = 4
    dropout_rate: float = 0.2
    max_lr: float = 0.01
    tbptt_k: int = 3


@dataclass
class SASRecConfig:
    hidden_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 1
    dropout: float = 0.2


@dataclass
class BERT4RecConfig:
    hidden_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.2
    mask_prob: float = 0.15


@dataclass
class Mamba4RecConfig:
    hidden_dim: int = 64
    num_blocks: int = 2
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    dropout: float = 0.2


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model_name: str = "titan4rec"
    titan4rec: Titan4RecConfig = field(default_factory=Titan4RecConfig)
    sasrec: SASRecConfig = field(default_factory=SASRecConfig)
    bert4rec: BERT4RecConfig = field(default_factory=BERT4RecConfig)
    mamba4rec: Mamba4RecConfig = field(default_factory=Mamba4RecConfig)


def get_config():
    import argparse

    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--model_name", type=str, default="titan4rec")

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
        choices=["ml-100k", "ml-1m", "ml-10m", "ml-20m", "ml-25m"],
    )
    parser.add_argument("--max_len", type=int, default=200)

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_batches_per_epoch", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="titan4rec")

    # Shared model args
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    # Titan4Rec
    parser.add_argument("--segment_size", type=int, default=20)
    parser.add_argument("--memory_depth", type=int, default=2)
    parser.add_argument("--memory_heads", type=int, default=2)
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--num_persistent", type=int, default=4)
    parser.add_argument("--max_lr", type=float, default=0.01)
    parser.add_argument("--tbptt_k", type=int, default=3)

    # BERT4Rec
    parser.add_argument("--mask_prob", type=float, default=0.15)

    # Mamba4Rec
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)

    args = parser.parse_args()

    config = Config()
    config.model_name = args.model_name
    # Data
    config.data.dataset = args.dataset
    config.data.max_len = args.max_len
    # Training
    config.train.batch_size = args.batch_size
    config.train.lr = args.lr
    config.train.num_epochs = args.num_epochs
    config.train.num_batches_per_epoch = args.num_batches_per_epoch
    config.train.patience = args.patience
    config.train.seed = args.seed
    config.train.accelerator = args.accelerator
    config.train.gradient_clip_val = args.gradient_clip_val
    config.train.use_wandb = args.use_wandb
    config.train.wandb_project = args.wandb_project
    # Titan4Rec
    config.titan4rec.d_model = args.d_model
    config.titan4rec.num_blocks = args.num_blocks
    config.titan4rec.num_heads = args.num_heads
    config.titan4rec.segment_size = args.segment_size
    config.titan4rec.memory_depth = args.memory_depth
    config.titan4rec.memory_heads = args.memory_heads
    config.titan4rec.expansion_factor = args.expansion_factor
    config.titan4rec.num_persistent = args.num_persistent
    config.titan4rec.dropout_rate = args.dropout_rate
    config.titan4rec.max_lr = args.max_lr
    config.titan4rec.tbptt_k = args.tbptt_k
    # SASRec (num_heads default=1 per original paper, not overridden by shared arg)
    config.sasrec.hidden_dim = args.d_model
    config.sasrec.num_blocks = args.num_blocks
    config.sasrec.dropout = args.dropout_rate
    # BERT4Rec
    config.bert4rec.hidden_dim = args.d_model
    config.bert4rec.num_blocks = args.num_blocks
    config.bert4rec.num_heads = args.num_heads
    config.bert4rec.dropout = args.dropout_rate
    config.bert4rec.mask_prob = args.mask_prob
    # Mamba4Rec
    config.mamba4rec.hidden_dim = args.d_model
    config.mamba4rec.num_blocks = args.num_blocks
    config.mamba4rec.d_state = args.d_state
    config.mamba4rec.expand = args.expand
    config.mamba4rec.d_conv = args.d_conv
    config.mamba4rec.dropout = args.dropout_rate
    return config
