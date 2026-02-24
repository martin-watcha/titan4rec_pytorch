# Titan4Rec

Applies the MAC (Memory as a Context) architecture from the Titans paper (Behrouz et al., 2024) to sequential recommendation. Implements next-item prediction using a neural long-term memory (LTM) that updates its weights at inference time, alongside standard Transformer attention as short-term memory.

Baselines: SASRec, BERT4Rec, Mamba4Rec. All models share the same data pipeline and evaluation code.

## Directory Structure

```
titan4rec_pytorch/
├── train.py              # Entry point — calls model/lit_module.main()
├── config.py             # All hyperparameters (dataclasses + argparse)
├── run_all.sh            # Experiment driver (baseline / ablation phases)
├── model/
│   ├── lit_module.py     # LightningModule, build_model(), results saving
│   ├── evaluate.py       # NDCG@10 / HR@10 evaluation (GPU-vectorized)
│   ├── proposed/         # Titan4Rec implementation
│   │   ├── titan4rec.py          # Top-level model (segment loop, TBPTT)
│   │   ├── mac_block.py          # MACBlock: LTM + Attention + FFN + PersistentMemory
│   │   ├── long_term_memory.py   # NeuralLongTermMemory (inner-loop weight updates)
│   │   ├── attention.py          # CausalAttention with MAC mask
│   │   └── embedding.py          # Item + positional embeddings
│   └── baseline/
│       ├── sasrec.py
│       ├── bert4rec.py
│       └── mamba4rec.py
├── data/
│   ├── dataset.py        # SeqRecDataModule, leave-one-out split
│   └── preprocess.py     # 5-core filtering, time-sort, padding
└── results/              # CSV experiment results per dataset
```

## Running Experiments

```bash
python train.py --model_name titan4rec --dataset ml-1m
python train.py --model_name sasrec --dataset ml-100k

PHASE=baseline bash run_all.sh ml-1m        # all baselines
PHASE=ablation bash run_all.sh ml-1m        # ablation study
bash run_all.sh ml-100k ml-1m               # full experiments
```

## Non-obvious Implementation Details

**padding_mask convention**: `True = valid token, False = padding` — consistent across titan4rec.py, mac_block.py, long_term_memory.py. Reversing this silently corrupts results.

**LTM inner-loop gradient**: `memory_state` weights are `.detach()`ed inside `_memory_grad()` — intentional to prevent second-order graph. Outer-loop gradients flow via `keys`/`values` (W_K, W_V stay attached). Do not remove the detach.

**Gate averaging**: alpha/eta must be divided by `valid_counts` (number of valid tokens), not by `C`. Padding positions are zeroed before averaging, so dividing by `C` underestimates the gate on short segments.

**TBPTT**: only `memory_state` and `momentum_state` are detached after each segment (outside the last `tbptt_k` segments). Segment outputs are never detached — attention/FFN receive gradients from all segments.

**gate_bias in MACBlock**: initialized to 2.0 so `sigmoid(2.0) ≈ 0.88`. Without this, the multiplicative gate starts at 0.5 and halves the signal per block.

**Baseline-specific notes**:
- SASRec: FFN uses ReLU (not GELU); keep the `sqrt(d_model)` embedding scale
- BERT4Rec: `mask_prob=1.0` (full mask at eval) — correct per original paper
- Mamba4Rec: no positional embedding — adding one hurts performance 4–9%

## Current Performance (101-candidate evaluation)

| Model | ml-100k NDCG@10 | ml-100k HR@10 | ml-1m NDCG@10 | ml-1m HR@10 |
|-------|:-:|:-:|:-:|:-:|
| SASRec | 0.425 | 0.686 | 0.240 | 0.425 |
| BERT4Rec | 0.378 | 0.634 | 0.210 | 0.381 |
| Mamba4Rec | 0.425 | 0.691 | 0.206 | 0.373 |
| Titan4Rec | 0.334 | 0.596 | 0.456 | 0.697 |

Titan4Rec outperforms on long histories (ml-1m), underperforms on short ones (ml-100k). LTM is most effective when users have rich interaction history.
