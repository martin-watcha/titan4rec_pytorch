import csv
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from config import get_config
from data.dataset import SeqRecDataModule
from model.evaluate import evaluate


class SeqRecLitModule(L.LightningModule):
    def __init__(self, model, config, dataset, max_len):
        super().__init__()
        self.model = model
        self.config = config
        self.dataset = dataset
        self.max_len = max_len
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            # BERT4Rec: (masked_seqs, labels)
            masked_seqs, labels = batch
            loss = self.model(masked_seqs, labels)
        else:
            # SASRec / Mamba4Rec / Titan4Rec: (seqs, pos, neg)
            seqs, pos, neg = batch
            pos_logits, neg_logits = self.model(seqs, pos, neg)

            mask = pos != 0
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)

            loss = self.criterion(pos_logits[mask], pos_labels[mask])
            loss += self.criterion(neg_logits[mask], neg_labels[mask])

        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Dummy — actual evaluation happens in on_validation_epoch_end
        pass

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        val_ndcg, val_hr = evaluate(
            self.model, self.dataset, self.max_len, is_test=False
        )
        test_ndcg, test_hr = evaluate(
            self.model, self.dataset, self.max_len, is_test=True
        )
        self.log("val/ndcg@10", val_ndcg, prog_bar=True, sync_dist=True)
        self.log("val/hr@10", val_hr, sync_dist=True)
        self.log("test/ndcg@10", test_ndcg, sync_dist=True)
        self.log("test/hr@10", test_hr, sync_dist=True)
        print(
            f"[Epoch {self.current_epoch:3d}]"
            f"  val_ndcg@10={val_ndcg:.4f}  val_hr@10={val_hr:.4f}"
            f"  test_ndcg@10={test_ndcg:.4f}  test_hr@10={test_hr:.4f}",
            flush=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.train.lr,
            betas=self.config.train.betas,
        )


def build_model(config, item_num):
    if config.model_name == "titan4rec":
        from model.proposed.titan4rec import Titan4Rec

        cfg = config.titan4rec
        return Titan4Rec(
            num_items=item_num,
            d_model=cfg.d_model,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            segment_size=cfg.segment_size,
            memory_depth=cfg.memory_depth,
            memory_heads=cfg.memory_heads,
            expansion_factor=cfg.expansion_factor,
            num_persistent=cfg.num_persistent,
            max_len=config.data.max_len,
            max_lr=cfg.max_lr,
            dropout_rate=cfg.dropout_rate,
            tbptt_k=cfg.tbptt_k,
        )
    elif config.model_name == "sasrec":
        from model.baseline.sasrec import SASRec

        cfg = config.sasrec
        return SASRec(
            item_num=item_num,
            hidden_units=cfg.hidden_dim,
            max_len=config.data.max_len,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout,
        )
    elif config.model_name == "bert4rec":
        from model.baseline.bert4rec import BERT4Rec

        cfg = config.bert4rec
        return BERT4Rec(
            item_num=item_num,
            hidden_units=cfg.hidden_dim,
            max_len=config.data.max_len,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout,
        )
    elif config.model_name == "mamba4rec":
        from model.baseline.mamba4rec import Mamba4Rec

        cfg = config.mamba4rec
        return Mamba4Rec(
            item_num=item_num,
            hidden_units=cfg.hidden_dim,
            max_len=config.data.max_len,
            num_blocks=cfg.num_blocks,
            d_state=cfg.d_state,
            expand=cfg.expand,
            d_conv=cfg.d_conv,
            dropout_rate=cfg.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {config.model_name}")


def main():
    config = get_config()
    L.seed_everything(config.train.seed)

    # DataModule
    dm = SeqRecDataModule(
        dataset_name=config.data.dataset,
        max_len=config.data.max_len,
        batch_size=config.train.batch_size,
        num_workers=config.data.num_workers,
        num_batches_per_epoch=config.train.num_batches_per_epoch,
        model_name=config.model_name,
        mask_prob=config.bert4rec.mask_prob,
    )
    dm.setup("fit")

    # Model
    model = build_model(config, dm.item_num)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model_name}, Parameters: {num_params:,}")

    lit = SeqRecLitModule(
        model=model,
        config=config,
        dataset=dm.dataset,
        max_len=config.data.max_len,
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val/ndcg@10",
            patience=config.train.patience,
            min_delta=config.train.min_delta,
            mode="max",
        ),
        ModelCheckpoint(
            monitor="val/ndcg@10",
            mode="max",
            dirpath=f"checkpoints/{config.model_name}_{config.data.dataset}",
            filename="best",
            save_top_k=1,
        ),
    ]

    # Logger
    logger: list | bool
    if config.train.use_wandb:
        from lightning.pytorch.loggers import WandbLogger

        logger = [WandbLogger(project=config.train.wandb_project)]
    else:
        logger = True

    # Trainer
    trainer = L.Trainer(
        max_epochs=config.train.num_epochs,
        accelerator=config.train.accelerator,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=1,
        gradient_clip_val=config.train.gradient_clip_val,
        enable_progress_bar=True,
    )

    t0 = time.time()
    trainer.fit(lit, dm)
    elapsed_min = (time.time() - t0) / 60

    # --- Load best checkpoint and re-evaluate for accurate test metrics ---
    ckpt_cb = trainer.checkpoint_callback
    if ckpt_cb and ckpt_cb.best_model_path:
        print(f"\nLoading best checkpoint: {ckpt_cb.best_model_path}")
        best_lit = SeqRecLitModule.load_from_checkpoint(
            ckpt_cb.best_model_path,
            model=build_model(config, dm.item_num),
            config=config,
            dataset=dm.dataset,
            max_len=config.data.max_len,
        )
        best_lit.model.to(lit.device)
        best_val_ndcg, best_val_hr = evaluate(
            best_lit.model, dm.dataset, config.data.max_len, is_test=False
        )
        best_test_ndcg, best_test_hr = evaluate(
            best_lit.model, dm.dataset, config.data.max_len, is_test=True
        )
        best_metrics = {
            "val/ndcg@10": best_val_ndcg,
            "val/hr@10": best_val_hr,
            "test/ndcg@10": best_test_ndcg,
            "test/hr@10": best_test_hr,
        }
        print(
            f"  Best ckpt: val_ndcg@10={best_val_ndcg:.4f}"
            f"  test_ndcg@10={best_test_ndcg:.4f}"
        )
    else:
        best_metrics = None

    # --- Save best results to CSV ---
    save_results(config, trainer, num_params, elapsed_min, best_metrics)


def save_results(config, trainer, num_params, elapsed_min, best_metrics=None):
    """Append best metrics to results/{dataset}_results.csv."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f"{config.data.dataset}_results.csv"

    # Use best-checkpoint metrics if available, else fall back to last-epoch
    if best_metrics is not None:
        best_val_ndcg = best_metrics["val/ndcg@10"]
        best_val_hr = best_metrics["val/hr@10"]
        best_test_ndcg = best_metrics["test/ndcg@10"]
        best_test_hr = best_metrics["test/hr@10"]
    else:
        metrics = trainer.callback_metrics
        best_val_ndcg = metrics.get("val/ndcg@10", float("nan"))
        best_val_hr = metrics.get("val/hr@10", float("nan"))
        best_test_ndcg = metrics.get("test/ndcg@10", float("nan"))
        best_test_hr = metrics.get("test/hr@10", float("nan"))

    if isinstance(best_val_ndcg, torch.Tensor):
        best_val_ndcg = best_val_ndcg.item()
    if isinstance(best_val_hr, torch.Tensor):
        best_val_hr = best_val_hr.item()
    if isinstance(best_test_ndcg, torch.Tensor):
        best_test_ndcg = best_test_ndcg.item()
    if isinstance(best_test_hr, torch.Tensor):
        best_test_hr = best_test_hr.item()

    # Build experiment tag for ablation identification
    tag_parts = [config.model_name]
    if config.model_name == "titan4rec":
        cfg = config.titan4rec
        defaults = {
            "memory_depth": 2, "num_persistent": 4, "segment_size": 20,
        }
        for k, v in defaults.items():
            actual = getattr(cfg, k)
            if actual != v:
                tag_parts.append(f"{k}={actual}")
    experiment = "_".join(tag_parts)

    # Model-specific hyperparams string
    if config.model_name == "titan4rec":
        cfg = config.titan4rec
        hparams = (
            f"d={cfg.d_model} blk={cfg.num_blocks} seg={cfg.segment_size} "
            f"mem_d={cfg.memory_depth} mem_h={cfg.memory_heads} "
            f"persist={cfg.num_persistent} exp={cfg.expansion_factor}"
        )
    elif config.model_name in ("sasrec", "bert4rec"):
        cfg = getattr(config, config.model_name)
        hparams = f"d={cfg.hidden_dim} blk={cfg.num_blocks} heads={cfg.num_heads}"
    elif config.model_name == "mamba4rec":
        cfg = config.mamba4rec
        hparams = (
            f"d={cfg.hidden_dim} blk={cfg.num_blocks} "
            f"state={cfg.d_state} expand={cfg.expand}"
        )
    else:
        hparams = ""

    row = {
        "experiment": experiment,
        "model": config.model_name,
        "dataset": config.data.dataset,
        "hparams": hparams,
        "params": num_params,
        "val_ndcg@10": f"{best_val_ndcg:.6f}",
        "val_hr@10": f"{best_val_hr:.6f}",
        "test_ndcg@10": f"{best_test_ndcg:.6f}",
        "test_hr@10": f"{best_test_hr:.6f}",
        "epochs": trainer.current_epoch,
        "time_min": f"{elapsed_min:.2f}",
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nResults saved to {csv_path}")
    print(f"  {experiment}: val_ndcg@10={best_val_ndcg:.4f}, test_ndcg@10={best_test_ndcg:.4f}")
