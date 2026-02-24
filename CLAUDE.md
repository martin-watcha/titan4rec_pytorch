# Titan4Rec 모델 구현 명세서

## 1. 개요

Titan4Rec은 Titans 논문 (Behrouz et al., 2024)의 **Memory as a Context (MAC)** 아키텍처를 Sequential Recommendation에 적용한 모델입니다. SASRec과 동일한 next-item prediction 태스크를 수행하되, neural long-term memory를 통해 inference time에도 유저의 선호를 학습할 수 있습니다.

**기반 논문**: "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024)
**참고 모델**: SASRec (Kang and McAuley, 2018)
**프레임워크**: PyTorch
**태스크**: Next-item prediction (sequential recommendation)

---

## 2. 전체 아키텍처

```
Input: User interaction sequence [i_1, i_2, ..., i_T]
        │
        ▼
   Item Embedding + Positional Encoding (segment 내부)
        │
        ▼
   ┌─── Segment into chunks of size C ───┐
   │                                       │
   │  For each segment S^(t):              │
   │                                       │
   │  1. Query LTM: h_t = M*_{t-1}(q_t)   │  ← Long-term Memory Retrieval
   │  2. Concat: [P || h_t || S^(t)]       │  ← Persistent Memory + LTM + Current
   │  3. Causal Attention + Residual       │  ← Short-term Memory
   │  4. Feedforward + Residual            │  ← Position-wise FFN
   │  5. Update LTM: M_t = M_{t-1}(y_t)   │  ← Long-term Memory Update (inner loop)
   │  6. Output: o_t = y_t * σ(M*_t(y_t)) │  ← Sigmoid-gated Output
   │                                       │
   └───────────────────────────────────────┘
        │
        ▼
   Prediction Layer → next item scores
```

---

## 3. 모듈별 상세 구현

### 3.1 Item Embedding Layer

SASRec과 동일한 구조를 기본으로 하되, positional encoding은 segment 단위로 적용합니다.

```python
class Titan4RecEmbedding(nn.Module):
    """
    - item_embedding: nn.Embedding(num_items + 1, d_model, padding_idx=0)
    - position_embedding: nn.Embedding(segment_size, d_model)
      ※ max_len이 아닌 segment_size 크기로 생성 (segment 내부에서만 사용)
    - dropout + RMSNorm (전체 아키텍처와 통일)
    """
```

**파라미터:**
- `num_items`: 아이템 수
- `d_model`: 임베딩 차원 (기본값: 64)
- `segment_size (C)`: segment 크기 (positional embedding 크기 결정)
- `dropout_rate`: 0.2 또는 0.5 (데이터셋 density에 따라)

**주의사항:**
- padding_idx=0 으로 설정 (item id는 1부터 시작)
- positional embedding은 segment 내부에서만 사용하므로 크기가 segment_size (C)

### 3.2 Neural Long-term Memory (LTM)

Titans 논문의 핵심 모듈입니다. MLP로 구현되며, inference time에도 weight가 업데이트됩니다.
**각 MAC block이 독립된 LTM을 소유합니다** (블록 간 공유하지 않음).

```python
class NeuralLongTermMemory(nn.Module):
    """
    Architecture: MLP with L_M layers
    - L_M = 1: Linear memory (W ∈ R^{d_model × d_model})
    - L_M >= 2: Deep memory (hidden_dim = d_model * expansion_factor)

    Parameters to define:
    - d_model: input/output dimension
    - num_layers (L_M): memory depth (기본값: 2, 논문에서 L_M >= 2 권장)
    - expansion_factor: MLP hidden dimension 배수 (기본값: 4, lucidrains 구현 참고)
    - num_heads: memory head 수 (기본값: 2, 각 head가 독립된 memory weight 보유)

    Learnable projections (outer loop에서 학습):
    - W_K: key projection (d_model → d_model)
    - W_V: value projection (d_model → d_model)
    - W_Q: query projection for retrieval (d_model → d_model)

    Data-dependent gates (outer loop에서 학습, per-token per-head scalar):
    - alpha_t: forgetting/decay gate
      → nn.Linear(d_model, num_heads) → sigmoid → shape: (batch, seq, num_heads)
    - theta_t: learning rate
      → nn.Linear(d_model, num_heads) → sigmoid * max_lr → shape: (batch, seq, num_heads)
    - eta_t: momentum decay
      → nn.Linear(d_model, num_heads) → sigmoid → shape: (batch, seq, num_heads)
    """
```

**논문 근거**: "Deep memory modules (L_M >= 2) are strictly more expressive than linear models" (Section 3.1)

#### 3.2.1 Memory Update (Inner Loop) - 핵심 로직

Titans 논문 Equation 13-14를 따릅니다:

```python
def memory_update(self, x_t, memory_state, momentum_state):
    """
    Inner loop: memory weight를 gradient descent로 업데이트.
    이 함수는 training과 inference 모두에서 호출됩니다.

    Args:
        x_t: current input (batch, seq_len, d_model) - segment의 각 토큰
        memory_state: dict of memory MLP weights, per batch sample
                      shape: {name: (batch, *param_shape)}
        momentum_state: 이전 momentum S_{t-1}, same shape as memory_state

    Returns:
        updated_memory_state, updated_momentum_state

    수식:
        k_t = x_t @ W_K   # (batch, seq, d_model)
        v_t = x_t @ W_V   # (batch, seq, d_model)

        # Associative memory loss (inner loop objective)
        # Eq. 12: ℓ(M_{t-1}; x_t) = ||M_{t-1}(k_t) - v_t||_2^2
        loss = ||M(k_t) - v_t||^2

        # Gradient of loss w.r.t. memory weights
        # 방법: torch.func.grad + vmap (per-sample gradient)
        grad = d(loss) / d(memory_weights)  # per batch sample

        # Data-dependent gates (per-token per-head scalar)
        alpha_t = sigmoid(linear_alpha(x_t))  # (batch, seq, heads) → decay
        theta_t = sigmoid(linear_theta(x_t)) * max_lr  # (batch, seq, heads) → lr
        eta_t = sigmoid(linear_eta(x_t))      # (batch, seq, heads) → momentum

        # theta_t를 loss weight으로 사용하여 gradient에 반영
        # weighted_loss = loss * theta_t → gradient에 lr이 곱해짐

        # Momentum update via associative scan (Eq. 14)
        # S_t = eta_t * S_{t-1} - theta_t * grad
        # → parallel associative scan으로 chunk 내 병렬 계산 가능

        # Memory update with decay via associative scan (Eq. 13)
        # M_t = (1 - alpha_t) * M_{t-1} + S_t
        # → assoc_scan(1 - alpha_t, update) 형태로 구현

    Inner loop gradient 계산 방법 (torch.func 사용):
        # 1. memory forward를 functional하게 정의
        def forward_and_loss(params, inputs, loss_weights, target):
            pred = functional_call(memory_model, params, inputs)
            loss = mse_loss(pred, target)
            return (loss * loss_weights).sum()

        # 2. per-sample gradient 계산
        grad_fn = torch.func.grad(forward_and_loss)
        per_sample_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, 0, 0))
        grads = per_sample_grad_fn(memory_weights, keys, adaptive_lr, values)

        # 3. momentum과 decay를 associative scan으로 적용

    주의사항:
    - torch.func.grad + vmap을 사용하면 create_graph 없이도 per-sample gradient 계산 가능
    - outer loop gradient는 vmap을 통해 자동으로 흐름
    - batch 내 각 sample의 memory weight가 독립적으로 관리됨
    """
```

#### 3.2.2 Memory Retrieval

```python
def retrieve(self, query, memory_state):
    """
    Memory에서 정보를 retrieval. Forward pass만 수행 (weight 업데이트 없음).

    Args:
        query: (batch, seq_len, d_model)
        memory_state: dict of memory weights, {name: (batch, *param_shape)}

    Returns:
        retrieved: (batch, seq_len, d_model)

    구현:
        q_t = query @ W_Q                    # (batch, seq, d_model)
        q_t = split_heads(q_t)               # (batch, heads, seq, head_dim)
        output = functional_call(memory_model, memory_state, q_t)
        output = merge_heads(output)          # (batch, seq, d_model)
    """
```

#### 3.2.3 Parallelization (Chunk-wise)

Titans 논문 Section 3.2의 병렬화 기법을 따릅니다.

```python
"""
Chunk 내부 (intra-chunk): 병렬 처리
- torch.func.vmap으로 chunk 내 모든 토큰의 gradient를 동시에 계산
- Momentum 계산: S_t = eta_t * S_{t-1} - theta_t * u_t (linear recurrence)
  → parallel associative scan으로 chunk 내 병렬 계산
- Decay 적용: M_t = (1-alpha_t) * M_{t-1} + S_t
  → parallel associative scan으로 chunk 내 병렬 계산

Chunk 간 (inter-chunk): 순차 처리
- chunk의 결과로 memory weight를 업데이트한 후 다음 chunk로 넘김

Associative Scan 구현:
- PyTorch의 torch.cumsum/cumprod 기반 구현 또는
- 별도의 parallel prefix sum 구현
- lucidrains 구현에서는 AssocScan 클래스로 추상화

초기 구현에서는 순차적 처리(for loop)로 먼저 구현하고,
정확성 검증 후 associative scan으로 병렬화할 것을 권장합니다.
"""
```

### 3.3 Persistent Memory

```python
class PersistentMemory(nn.Module):
    """
    Learnable, data-independent parameters.
    Attention의 시작 부분에 prepend됩니다.
    Test time에는 고정 (학습 가능하지만 input에 의존하지 않음).

    Parameters:
    - N_p: persistent memory token 수 (기본값: 4~16)
    - d_model: dimension

    Implementation:
    - self.persistent_tokens = nn.Parameter(torch.randn(N_p, d_model))
    - 각 segment 처리 시 persistent_tokens를 batch 차원으로 expand하여 concat

    논문: "encode the knowledge about a task" (Section 3.3, Eq. 19)
    """
```

### 3.4 MAC (Memory as a Context) Block

하나의 MAC 블록은 다음 순서로 동작합니다:

```python
class MACBlock(nn.Module):
    """
    Titans 논문 Section 4.1의 MAC 아키텍처.
    각 블록은 독립된 LTM, Attention, FFN, PersistentMemory를 소유합니다.

    Components:
    - ltm: NeuralLongTermMemory (블록별 독립)
    - persistent_memory: PersistentMemory
    - attention: CausalAttention
    - ffn: PositionwiseFFN
    - pre_attn_norm: nn.RMSNorm(d_model)
    - pre_ffn_norm: nn.RMSNorm(d_model)
    - gate_norm: nn.RMSNorm(d_model)  # gated output용

    Forward pass (하나의 segment S^(t) 처리):

    1. Memory Retrieval (Eq. 21):
       h_t = LTM.retrieve(S^(t), memory_state)  # (batch, C, d_model)
       # 내부에서 W_Q projection 수행 (retrieve 함수 참고)

    2. Context Construction (Eq. 22):
       S_tilde = concat([persistent_memory, h_t, S^(t)], dim=seq)
       # shape: (batch, N_p + C + C, d_model)

    3. Pre-norm + Causal Attention + Residual (Eq. 23):
       attn_out = Attention(RMSNorm(S_tilde))
       y_t = S_tilde + attn_out  # residual connection

    4. Pre-norm + FFN + Residual:
       ffn_out = FFN(RMSNorm(y_t))
       y_t = y_t + ffn_out  # residual connection

    5. Memory Update (Eq. 24):
       # y_t의 segment 부분만 사용하여 LTM 업데이트
       memory_state, momentum = LTM.update(y_t_segment, memory_state, momentum)

    6. Sigmoid-gated Output (Eq. 25):
       # 업데이트된 memory에서 다시 retrieval
       z_t = LTM.retrieve(y_t_segment, memory_state)  # (batch, C, d_model)
       # sigmoid gating: 논문의 "learnable vector-valued weights + σ(.)"
       o_t = RMSNorm(y_t_segment) * sigmoid(RMSNorm(z_t))
       # 또는 간단한 형태: o_t = y_t_segment * sigmoid(z_t)

    Returns: (o_t, updated_memory_state, updated_momentum_state)
       - o_t: (batch, C, d_model) — segment 부분만 반환
       - updated_memory_state: dict, 업데이트된 memory weights
       - updated_momentum_state: dict, 업데이트된 momentum
    """
```

**Gated Output 상세 설명:**
- 논문 원문: "we normalize the outputs y and M(x̃) using learnable vector-valued weights, followed by a non-linearity σ(.)"
- `σ`는 sigmoid activation
- 구현: `output = norm1(y) * sigmoid(norm2(memory_retrieved))` 또는 `output = y * sigmoid(memory_retrieved)`
- kolejnyy 구현: `o = y * sigmoid(z)` (z는 memory retrieval 결과)
- lucidrains 구현: `attn_out_gates = retrieved.sigmoid()` → `out = out * output_gating`

### 3.5 Attention Module

```python
class CausalAttention(nn.Module):
    """
    Multi-head causal self-attention with MAC mask 구조.

    Parameters:
    - d_model: dimension
    - num_heads: attention head 수 (기본값: 2)
    - dropout_rate: attention dropout

    Q, K, V Projections + Activation + Normalization (Section 4.4):
    - Q = L2Norm(SiLU(x @ W_Q))  # SiLU activation 후 L2 normalization
    - K = L2Norm(SiLU(x @ W_K))
    - V = SiLU(x @ W_V)          # V는 L2 norm 불필요

    선택적 Convolution (Section 4.4):
    - 1D depthwise-separable convolution after Q, K, V projections
    - 논문에서 언급되나 kernel size 미지정 (일반적으로 3 또는 5)
    - 초기 구현에서는 생략 가능

    Attention Mask 구조 (Figure 3a):
    ┌─────────────────────────────────────────────┐
    │ Persistent │  LTM Retrieved  │   Segment    │
    │  (N_p)     │     (C)         │    (C)       │
    ├────────────┼─────────────────┼──────────────┤
    │   ■■■■     │   ■■■■■■■■■    │   ■          │  ← token 1 of segment
    │   ■■■■     │   ■■■■■■■■■    │   ■■         │  ← token 2 of segment
    │   ■■■■     │   ■■■■■■■■■    │   ■■■        │  ← token 3 of segment
    │   ...      │   ...           │   ...        │
    └────────────┴─────────────────┴──────────────┘

    ■ = attend 가능
    - Persistent memory (N_p개): 모든 segment token이 attend 가능 (prefix)
    - LTM retrieved (C개): 모든 segment token이 attend 가능 (prefix)
    - Segment 내부 (C개): causal mask (하삼각)
    - 즉, prefix 부분 (N_p + C)은 fully visible, segment 부분만 causal

    Padding Mask:
    - padding token (item_id=0)은 attention에서 masking 처리
    """
```

### 3.6 Position-wise Feedforward Network

```python
class PositionwiseFFN(nn.Module):
    """
    Standard Transformer FFN (attention 뒤에 위치).
    Pre-norm Transformer 구조: RMSNorm → FFN → Residual

    Implementation:
    - Linear(d_model, d_model * 4)
    - SiLU activation (또는 GELU)
    - Dropout
    - Linear(d_model * 4, d_model)
    - Dropout
    """
```

### 3.7 전체 모델

```python
class Titan4Rec(nn.Module):
    """
    전체 모델 구조.

    Parameters:
    - num_items: 아이템 수
    - d_model: 임베딩 차원 (기본값: 64)
    - num_blocks: MAC block 수 (기본값: 2, 각 블록은 독립된 LTM 소유)
    - num_heads: attention head 수 (기본값: 2)
    - segment_size (C): segment 크기 (기본값: 20~50)
    - memory_depth (L_M): LTM의 MLP 층 수 (기본값: 2)
    - memory_heads: LTM의 head 수 (기본값: 2)
    - expansion_factor: LTM MLP hidden dim 배수 (기본값: 4)
    - num_persistent (N_p): persistent memory token 수 (기본값: 4)
    - max_len: 최대 시퀀스 길이 (기본값: 200)
    - dropout_rate: dropout rate
    - tbptt_k: TBPTT window 크기 (기본값: 3)
      ※ 마지막 tbptt_k개 segment만 memory_state backward 허용
      ※ num_segments <= tbptt_k이면 full BPTT (detach 없음)

    Forward:
    1. Item Embedding (segment 분할 전에 한 번만 수행):
       x = item_emb(input_seq)  # (batch, seq_len, d_model)
    2. Sequence를 segment로 분할:
       segments = split(x, C)  # list of (batch, C, d_model)
       - 길이가 C의 배수가 아니면 왼쪽을 0으로 padding 후 분할
    3. 각 MAC block에 대해 LTM state를 외부 dict로 초기화:
       memory_states = [
           {name: param.unsqueeze(0).expand(batch, ...).clone()}
           for block in mac_blocks
       ]
       momentum_states = [zeros_like(ms) for ms in memory_states]
    4. 각 segment에 대해 positional encoding 추가 후 모든 MAC block 순차 처리.
       segment 처리 후 steps_remaining = num_segments - 1 - t를 계산.
       steps_remaining >= tbptt_k이면 memory_states/momentum_states를 detach():
       for t, segment in enumerate(segments):
           seg = segment + pos_emb(range(C))
           for i, block in enumerate(mac_blocks):
               seg, memory_states[i], momentum_states[i] = block(...)
           outputs.append(seg)
           if (num_segments - 1 - t) >= tbptt_k:
               memory_states[i] = {k: v.detach() ...}  # TBPTT
    5. 모든 segment output을 concat → (batch, seq_len, d_model)
    6. RMSNorm → Prediction: dot product with item embeddings

    Prediction Layer:
    - SASRec과 동일: output embedding과 item embedding의 dot product
    - item_embedding layer를 prediction layer와 공유 (shared embedding)

    Normalization:
    - 전체적으로 RMSNorm 사용 (Pre-norm 구조)
    - Pre-attention RMSNorm, Pre-FFN RMSNorm, Final output RMSNorm
    """
```

---

## 4. Training

### 4.1 Loss Function

```python
"""
Outer Loop Loss (전체 모델 학습):
  - Binary Cross-Entropy with negative sampling (SASRec과 동일)
  - 각 position t에서:
    - positive item: 다음 아이템 i_{t+1}
    - negative item: 해당 유저가 interact하지 않은 아이템 중 random 1개
  - padding position (item_id=0)에서는 loss 계산하지 않음

Inner Loop Loss (memory update, forward pass 내부에서 자동 실행):
  - Associative memory loss: ||M(k_t) - v_t||^2 (Eq. 12)
  - 이 loss는 outer loop에서 직접 optimize하지 않음
  - memory weight update에만 사용됨
  - torch.func.grad + vmap으로 per-sample gradient 계산
"""
```

### 4.2 Optimizer & Hyperparameters

```python
"""
Optimizer: Adam
Learning rate: 0.001 (SASRec 기본값)
Batch size: 128
Weight decay: 없음 (outer loop에서는 사용하지 않음)
  ※ inner loop의 alpha_t (decay gate)가 memory에 대한 weight decay 역할 수행

Hyperparameter search 범위:
- d_model: [32, 64, 128]
- segment_size C: [10, 20, 50]
- memory_depth L_M: [1, 2, 3]  (논문은 L_M >= 2 권장)
- expansion_factor: [2, 4]
- num_persistent N_p: [4, 8, 16]
- num_blocks: [1, 2, 3]
- num_heads (attention): [1, 2, 4]
- memory_heads: [1, 2, 4]
- dropout: [0.2, 0.5]
- max_lr (theta_t의 상한): [0.01, 0.1, 1.0]
- tbptt_k: [2, 3, 5]  (default=3)
"""
```

### 4.3 Training Procedure

```python
"""
1. 데이터 전처리: SASRec과 동일
   - 각 유저의 interaction sequence를 시간순 정렬
   - 마지막 item: test, 마지막에서 두번째: validation, 나머지: training
   - max_len보다 긴 시퀀스는 최근 max_len개로 truncate
   - max_len보다 짧으면 왼쪽을 0으로 padding

2. Training loop:
   - 각 batch에서 user sequence를 입력
   - 모델 forward pass (내부에서 LTM의 inner loop 자동 실행)
   - BCE loss 계산 (padding position 제외)
   - Outer loop backprop (TBPTT로 graph depth 제한)
   - Optimizer step

3. Evaluation:
   - 각 유저에 대해 full sequence를 입력 (train + valid까지)
   - LTM은 M_0부터 시작하여 시퀀스를 순차 처리하며 memory 축적
   - 마지막 position의 output으로 전체 아이템에 대해 ranking
   - Metrics: Hit@10, NDCG@10
   - 후보 아이템: 전체 아이템 (full ranking)
     ※ 일부 실험에서 random 100개 negative + 1 positive로 간소화 가능 (명시할 것)
"""
```

### 4.4 Truncated BPTT — 구현 완료

```python
"""
[현재 구현 상태: titan4rec.py에 tbptt_k=3으로 적용됨]

Memory State의 Gradient 전파 관리:

- max_len=200, segment_size=20이면 10개 segment를 순차 처리
- full backprop 시 computational graph가 10단계 깊이로 쌓여 극도로 느려짐
  (측정: full BPTT ~1,198ms/batch vs TBPTT ~1.1ms — 약 1,000x 차이)

현재 구현 전략 (tbptt_k=3):
  for t in range(num_segments):
      ... process segment t ...
      outputs.append(seg)                    # 모든 seg output은 backward에 참여
      steps_remaining = num_segments - 1 - t
      if steps_remaining >= tbptt_k:         # 마지막 tbptt_k개 segment 이외는 detach
          memory_states[i] = {k: v.detach() ...}

- tbptt_k=3: 마지막 3 segment (t=7,8,9 @ 10 segments)만 memory 체인 backward 허용
- num_segments <= tbptt_k이면 detach 없이 full BPTT (짧은 sequence 자동 처리)
- segment output 자체(outputs 리스트)는 detach 안 함 → attention/FFN 파라미터는
  모든 segment에서 gradient 수신
- outer loop 파라미터 (W_K, W_V, W_Q, to_alpha, to_theta, to_eta)는
  tbptt_k segment 이내의 keys/values 경로를 통해 모두 gradient 수신 확인됨

CLI: --tbptt_k (default=3), config.py Titan4RecConfig.tbptt_k 필드 존재
"""
```

---

## 5. 구현 우선순위

### Phase 1: 기본 동작 확인 ✅ 완료

1. **순차적 LTM 구현** — 완료
2. **SASRec baseline 재현** — 완료
3. **MAC block 통합** — 완료 (end-to-end 학습, NDCG@10=0.324 달성)

### Phase 2: 성능 최적화 (진행 중)

4. **TBPTT + Padding gate 버그 수정** — 완료 (섹션 11 참고)
5. **Associative scan으로 momentum/decay 병렬화** — 미구현 (현재 chunk-level for-loop)
6. **Hyperparameter tuning** — 진행 중

### Phase 3: 실험 확장

7. 다양한 데이터셋 실험
8. Ablation study (run_ablation.sh)
9. Long history 유저 분석

---

## 6. 데이터셋

SASRec 논문과 동일한 데이터셋 사용:

| Dataset | Domain | 특징 |
|---------|--------|------|
| MovieLens-100K | 영화 | 가장 작음, 빠른 실험용 |
| MovieLens-1M | 영화 | Dense, 평균 시퀀스 길이 길음 |
| Amazon Beauty | 뷰티 제품 | Sparse |
| Amazon Games | 게임 | Sparse |
| Steam | 게임 | Dense, 시퀀스 길이 다양 |

전처리: 5-core filtering (interaction 5개 미만 유저/아이템 제거)

---

## 7. 핵심 구현 주의사항

### 7.1 Inner Loop Gradient 계산

```python
"""
현재 구현: Explicit bmm (batched matrix multiplication) 방식

vmap(grad)+functional_call 대신 순수 batched matmul로 forward/backward를 직접 구현.
memory_state weights는 detach하여 second-order graph 방지, outer-loop gradient는
keys/values 경로(W_K, W_V에 attached)로 흐름.

구현 위치: long_term_memory.py의 _memory_forward(), _memory_grad()

Forward: h = x → bmm(h, W0) → GELU → bmm(h, W1) → RMSNorm → + x (residual)
Backward: explicit chain rule with bmm, _gelu_grad, _rmsnorm_backward

Per-sample gradient norm clipping (_clip_grad_norm):
- 모든 W_i의 gradient를 합산한 total norm 계산
- max_norm=10.0 초과 시 scale down
- clip coefficient는 torch.no_grad() 내에서 계산 (autograd graph에 포함시키면 NaN 발생)

이전 구현 (vmap 기반)은 ~2.7x 느렸으므로 bmm으로 교체됨.
"""
```

### 7.2 Memory State 관리

```python
"""
Per-batch-sample 독립 관리:
- memory_state는 dict 형태: {param_name: tensor of shape (B*H, *param_shape)}
  ※ B*H = batch_size * memory_heads (head별로 독립 memory)
- 초기화: memory MLP의 nn.Parameter를 B*H 차원으로 expand + clone
  → memory_state = {name: param.unsqueeze(0).expand(bh, *param.shape).clone()}
- momentum_state: memory_state와 동일한 shape, zeros로 초기화

메모리 비용 분석 (d_model=64, memory_heads=2, expansion_factor=4, L_M=2, batch=128):
- head_dim = d_model / memory_heads = 32
- 한 head의 MLP: layer1 (32 * 128) + layer2 (128 * 32) = 8K params per head
- 2 heads, 2 layers: ~32K params (weight + bias 포함)
- Per batch: 128 * 32K * 4 bytes ≈ 16MB
- num_blocks=2: ~32MB
- momentum도 동일 크기 → 총 ~64MB (ms + mom, 2 blocks)
→ 관리 가능한 수준이지만, d_model이 커지면 주의 필요

Training 시 매 batch마다 memory_state를 M_0로 re-초기화합니다.
Evaluation 시에도 각 유저마다 M_0부터 시작합니다.

TBPTT와 detach 관계:
- titan4rec.py의 segment loop에서 TBPTT detach 수행 (섹션 4.4 참고)
- memory_state 자체를 detach()하면 memory MLP weights의 gradient chain이 끊김
- 이는 의도적: outer loop gradient는 keys/values 경로(W_K, W_V)로 계속 흐름
- 완전 detach(tbptt_k=0)도 버그 아님 — outer loop 파라미터는 별도 경로로 gradient 수신
"""
```

### 7.3 Segment 처리 시 Padding

```python
"""
시퀀스 길이가 segment_size C의 배수가 아닐 수 있습니다.
- 마지막 segment는 C보다 짧을 수 있음
- 왼쪽을 0으로 padding하여 C에 맞춤
- padding_mask 규약: True = 유효 토큰 (item_id != 0), False = padding
  ※ 이 규약이 titan4rec.py, mac_block.py, long_term_memory.py 전체에서 일관되게 사용됨
- attention mask에서 padding token (item_id=0)은 masking 처리
- segment 분할 후 inverse 함수로 원래 위치 복원 (lucidrains 참고)

Padding gate 평균 계산 주의사항 (long_term_memory.py):
- padding position의 alpha/eta는 mask=0으로 zeroing 처리됨
- 평균을 낼 때 반드시 유효 토큰 수(valid_counts)로 나눠야 함
- alpha.mean(dim=1)은 padding 0을 포함해 C로 나누므로 잘못된 결과
- 예: C=20, 유효 토큰=5이면 alpha.mean()은 정확한 값의 1/4로 underestimate
- 현재 구현: valid_counts = padding_mask.float().sum(dim=1)으로 나눔 (수정 완료)
"""
```

### 7.4 Residual Connection 구조

```python
"""
논문 Section 4.4: "In all blocks, we use residual connections"

Pre-norm Transformer 구조:
    # Attention sublayer
    x_norm = RMSNorm(x)
    attn_out = Attention(x_norm)
    x = x + attn_out

    # FFN sublayer
    x_norm = RMSNorm(x)
    ffn_out = FFN(x_norm)
    x = x + ffn_out

주의: Residual은 full context (persistent + LTM + segment) 에 대해 적용.
최종 output에서 segment 부분만 추출.
"""
```

---

## 8. 파일 구조

```
titan4rec_pytorch/
├── CLAUDE.md                  # 이 파일 (모델 구현 명세서)
├── README.md                  # 프로젝트 소개
├── config.py                  # Hyperparameters (dataclass + argparse)
├── train.py                   # main() — Lightning Trainer 실행
├── pyproject.toml             # Python 프로젝트 설정
├── uv.lock                    # 의존성 lock 파일
├── .gitignore                 # Git 무시 파일
├── run_all.sh                 # 전체 실험 스크립트
├── model/
│   ├── __init__.py
│   ├── lit_module.py          # PyTorch Lightning 모듈 + build_model() + best checkpoint 재평가
│   ├── evaluate.py            # Evaluation (NDCG@10, HR@10, 101-candidate, GPU 벡터화)
│   ├── proposed/              # Titan4Rec (제안 모델)
│   │   ├── __init__.py        # RMSNorm export
│   │   ├── embedding.py       # Item embedding + absolute positional encoding
│   │   ├── long_term_memory.py  # NeuralLongTermMemory (explicit bmm gradient + norm clipping)
│   │   ├── attention.py       # CausalAttention (MAC mask: prefix visible + segment causal)
│   │   ├── mac_block.py       # MACBlock (LTM + PersistentMemory + Attention + FFN)
│   │   └── titan4rec.py       # Titan4Rec 전체 모델 (TBPTT 포함)
│   └── baseline/              # Baseline 모델들
│       ├── __init__.py
│       ├── sasrec.py           # SASRec — FFN: ReLU (원 논문 일치)
│       ├── bert4rec.py         # BERT4Rec — mask_prob=1.0 (원 논문 기본값)
│       └── mamba4rec.py        # Mamba4Rec — positional embedding 없음 (원 논문)
├── data/
│   ├── __init__.py
│   ├── preprocess.py          # 데이터 전처리 (SASRec과 동일)
│   └── dataset.py             # SeqRecDataModule, leave-one-out split
└── results/                   # CSV 결과 저장 디렉토리
    ├── ml-100k_results.csv
    └── ml-1m_results.csv
```

**모델 분리 원칙:**
- `proposed/`: Titan4Rec 관련 모듈만 포함. 모듈별 분리로 실험/수정 용이.
- `baseline/`: 각 baseline은 단일 파일로 자기 완결적 구현. 공통 인터페이스 준수.
- 공통 인터페이스: 모든 모델은 `forward(seqs, pos, neg) → (pos_logits, neg_logits)` 형태를 따르며, `train.py`와 `evaluate.py`에서 동일한 코드로 학습/평가 가능.
- `data/`, `model/lit_module.py`, `model/evaluate.py`는 모델 간 공유.

---

## 9. 기대 결과 (실험 설계)

### 9.1 Main Results
- Titan4Rec vs SASRec vs BERT4Rec vs Mamba4Rec on 4 datasets
- Metrics: Hit@10, NDCG@10
- 비교 모델: `model/baseline/`에 직접 구현 (동일 데이터 파이프라인, 동일 평가 코드 사용)

### 9.2 Ablation Study
- Full model vs Linear memory (L_M=1) vs No momentum (eta=0) vs No decay (alpha=0) vs No persistent memory

### 9.3 Memory Depth Analysis
- L_M = 1, 2, 3, 4에 따른 성능 변화

### 9.4 Sequence Length Analysis
- 유저를 히스토리 길이별로 그룹화 (short/medium/long)
- 각 그룹에서 모델별 성능 비교

### 9.5 Architecture Variants
- MAC vs MAG vs MAL 비교 (가능하면)

---

## 10. 참고 자료

- [Titans 논문 (arXiv)](https://arxiv.org/abs/2501.00663)
- [lucidrains/titans-pytorch (비공식 구현)](https://github.com/lucidrains/titans-pytorch)
- [kolejnyy/titans-lmm (PoC 구현)](https://github.com/kolejnyy/titans-lmm)

---

## 11. 구현 이력 및 현재 상태

### 11.1 베이스라인 수정 이력

아래 수정은 각 모델의 원 논문 코드베이스를 직접 확인 후 적용됨.

#### SASRec (`model/baseline/sasrec.py`)
- **수정**: FFN activation `GELU` → `ReLU`
- **근거**: 원 논문 공식 TF/PyTorch 구현 모두 ReLU 사용
- **주의**: sqrt(d_model) scaling은 원 논문에 있으므로 **유지** (제거하면 안 됨)

#### BERT4Rec (`model/baseline/bert4rec.py`)
- **수정 없음**: 현재 구현의 mask_prob=1.0 (전체 마스킹)이 원 논문 공식 코드(FeiSun/BERT4Rec) 기본값과 일치
- **주의**: 80/10/10 masking split은 MLM 사전학습 논문의 방식이고, BERT4Rec 추천 모델에서는 적용하지 않음

#### Mamba4Rec (`model/baseline/mamba4rec.py`)
- **수정**: positional embedding 완전 제거 (`self.pos_emb` + 관련 코드)
- **근거**: 원 Mamba4Rec 논문이 명시적으로 "No positional embedding" 언급; PE 추가 시 성능 4~9% 하락 확인됨

### 11.2 Titan4Rec 버그 수정 이력

#### Fix 1: TBPTT (Truncated Backpropagation Through Time)

**파일**: `model/proposed/titan4rec.py`, `config.py`, `model/lit_module.py`

**문제**: segment loop 전체를 하나의 computation graph로 유지하면 backward가 10단계 깊이의 graph를 역전파해야 함. 측정 결과 full BPTT는 TBPTT 대비 ~1,000x 느림.

**수정 내용**:
```python
# titan4rec.py segment loop 내부 (처리 후)
steps_remaining = num_segments - 1 - t
if steps_remaining >= self.tbptt_k:   # 마지막 tbptt_k segment만 backward 허용
    for i in range(len(self.mac_blocks)):
        memory_states[i] = {k: v.detach() for k, v in memory_states[i].items()}
        momentum_states[i] = {k: v.detach() for k, v in momentum_states[i].items()}
```

**검증**: segment output(outputs 리스트)은 detach하지 않으므로 attention/FFN은 모든 segment에서 gradient 수신. W_K/W_V/W_Q/to_alpha/to_theta/to_eta 모두 비-0 gradient 확인.

#### Fix 2: Padding Gate Averaging 오류

**파일**: `model/proposed/long_term_memory.py`

**문제**: `alpha.mean(dim=1)` / `eta.mean(dim=1)`이 padding으로 zeroing된 위치를 포함해 전체 C로 나눔.

**수정**: `valid_counts = padding_mask.float().sum(dim=1)`으로 유효 토큰 수만으로 나눔.

#### Fix 3: Inner-loop Gradient Norm Clipping

**파일**: `model/proposed/long_term_memory.py`

**문제**: inner-loop gradient가 폭발하면 memory weights가 Inf → isfinite fallback이 `w.detach()`를 사용하여 outer-loop gradient flow가 끊김.

**수정**: `_clip_grad_norm(grads, max_norm=10.0)` 추가. per-sample gradient norm을 모든 W_i에 대해 계산 후 초과 시 scale down.

**주의**: clip coefficient 계산은 반드시 `torch.no_grad()` 내에서 수행. autograd graph에 포함시키면 모든 gradient 원소 간 cross-dependency가 생겨 NaN backward 발생 (검증 완료).

#### Fix 4: Best Checkpoint 재평가

**파일**: `model/lit_module.py`

**문제**: `trainer.callback_metrics`는 마지막 epoch의 메트릭을 반환하므로, early stopping으로 선택된 best checkpoint의 실제 성능과 다를 수 있음.

**수정**: training 완료 후 best checkpoint를 다시 로드하여 val/test 메트릭을 재계산. `save_results()`에 `best_metrics` 파라미터 추가.

#### Fix 5: Evaluation 결정론성 + Target 중복 방지

**파일**: `model/evaluate.py`

**문제 1**: `np.random.randint`와 `random.sample`이 global state 사용 → epoch별 결과 비결정적.
**수정 1**: local `random.Random(seed)`, `np.random.RandomState(seed)` 사용.

**문제 2**: `rated` set에 target이 포함되지 않아 negative로 샘플링될 수 있음 (~7% 확률, ml-100k).
**수정 2**: `rated.add(target)` 추가.

### 11.3 GPU 최적화 이력

#### Eval GPU 벡터화
**파일**: `model/evaluate.py`
- per-user Python/NumPy 루프 → batched `torch.argsort` on GPU
- NDCG/HR 계산도 GPU에서 벡터화

#### Tensor 할당 최적화
- `mac_block.py`: `_persistent_valid`을 `register_buffer`로 등록 (매 forward마다 생성 방지)
- `titan4rec.py`: `seg_mask_float`를 segment당 1회 사전 계산 (블록별 재계산 방지)

#### Flash Attention 조사 결과 (적용 불가)
- MAC mask(prefix visible + segment causal)는 비표준 mask → Flash Attention 비활성화
- split attention 접근은 softmax 정규화가 달라져 결과 불일치
- `flex_attention` (PyTorch >= 2.5 필요), `flash_attn` 패키지(미설치) 모두 사용 불가
- PyTorch 업그레이드 시 재검토 가능

### 11.4 현재 성능 결과

| Model | ml-100k NDCG@10 | ml-100k HR@10 | ml-1m NDCG@10 | ml-1m HR@10 |
|-------|-----------------|---------------|---------------|-------------|
| SASRec | 0.425 | 0.686 | 0.240 | 0.425 |
| BERT4Rec | 0.378 | 0.634 | 0.210 | 0.381 |
| Mamba4Rec | 0.425 | 0.691 | 0.206 | 0.373 |
| Titan4Rec | 0.334 | 0.596 | **0.456** | **0.697** |

**핵심 관찰**: Titan4Rec은 ml-1m(긴 시퀀스)에서 SASRec 대비 NDCG +90% 우위, ml-100k(짧은 시퀀스)에서는 열위. LTM은 긴 히스토리에서 효과적.

### 11.5 미해결 이슈 / 향후 작업

1. **Associative scan 병렬화**: 현재 momentum/decay는 chunk-level 단일 스텝. 토큰별 병렬 scan 전환 가능하나 chunk_size=20에서는 효과 미미.

2. **Fix 3~5 적용 후 성능 재측정 필요**: `python train.py --model_name titan4rec --dataset ml-100k` 실행.

3. **Hyperparameter tuning**: Fix 적용 후 grid search 필요 (특히 tbptt_k, max_lr, segment_size).

4. **추가 데이터셋 실험**: Amazon Beauty, Amazon Games, Steam 등.
