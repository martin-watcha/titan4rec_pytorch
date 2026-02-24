# Titan4Rec

PyTorch implementation of **Titan4Rec**, which applies the [Titans](https://arxiv.org/abs/2501.00663) MAC (Memory as a Context) architecture to sequential recommendation.

The model combines a neural long-term memory (LTM) — whose weights update at inference time — with standard Transformer attention as short-term memory, enabling the model to adapt to each user's preferences without retraining.

## Results (NDCG@10 / HR@10, 101-candidate evaluation)

| Model | ml-100k | ml-1m |
|-------|---------|-------|
| SASRec | 0.425 / 0.686 | 0.240 / 0.425 |
| BERT4Rec | 0.378 / 0.634 | 0.210 / 0.381 |
| Mamba4Rec | 0.425 / 0.691 | 0.206 / 0.373 |
| **Titan4Rec** | 0.334 / 0.596 | **0.456 / 0.697** |

Titan4Rec performs best on datasets with long user histories (ml-1m), where the LTM can accumulate meaningful context.

## Usage

```bash
# Single experiment
python train.py --model_name titan4rec --dataset ml-1m

# Full benchmark (baseline + ablation)
bash run_all.sh ml-100k ml-1m
```

Key options: `--d_model`, `--num_blocks`, `--segment_size`, `--memory_depth`, `--tbptt_k`, `--max_lr`

## References

- Behrouz et al., [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (2024)
- Kang & McAuley, [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (2018)
