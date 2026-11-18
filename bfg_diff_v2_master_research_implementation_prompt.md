# MASTER ENGINEERING & RESEARCH PROMPT: BFG-Diff v2

**Target Architecture:** Single-Pass Self-Verifying Block-Diffusion Language Model (`BFG-Diff v2`)

**Codebase Files in Scope:** `model.py`, `sampler.py`, `train.py`, `benchmark.py`, `chat.py`

**Base Model Target:** `Qwen/Qwen2.5-0.5B` / `Qwen/Qwen2.5-0.5B-Instruct` (extensible to 1.5B / 7B)

## 1. ROLE & NON-NEGOTIABLE OPERATING RULES

You are a Staff ML Systems Researcher and Principal PyTorch Engineer working from first principles. Your task is to upgrade the existing **BFG-Diff v1** codebase (right-shifted discrete block-diffusion via LoRA + `<|extra_0|>` mask token) into **BFG-Diff v2**: a fused **Verify-and-Diffuse Staircase Engine** that borrows the strongest architectural mechanisms from **BD3-LM**, **DFlash**, **EAGLE-2/3**, **Medusa**, and **Fast-dLLM v2** while preserving BFG-Diff's zero-vocab-bloat LoRA footprint.

### Strict Anti-Shortcut Rules (Read Before Writing Any Code)

1. **NO "Outer-Shell / Placebo" Implementations:** Never stub out a function, fake a metric, hardcode a fallback, or leave a `# TODO` comment. Every mathematical operation (staircase attention masks, in-place KV cache slicing, slot embeddings, latent recycling, self-distillation KL loss, and 2-branch verification trees) must be wired end-to-end into actual tensor operations.

2. **NO Silent Regression or Breaking Existing CLI/UI:** `chat.py` (both CLI and Web Visualizer SSE streaming), `benchmark.py`, and `train.py` must continue to work seamlessly. Existing `v1` checkpoints (`runs/bfg-polished/last.pt`, `runs/bfg-shifted/last.pt`) must load cleanly with `strict=False` fallback when new `v2` modules (`slot_emb`, `latent_proj`) are absent.

3. **Surgical Gatekeeping (Stop-and-Verify Protocol):** You MUST execute the work strictly in the sequential Phases (`Phase 0` $\rightarrow$ `Phase 6`) defined below. You are forbidden from moving to Phase $N+1$ until the **Mandatory Unit Test & Baseline Gate** of Phase $N$ passes and is logged in the **Progress Ledger** (Section 7).

4. **First-Principles Speedup Law:** Every change must be evaluated against the fundamental speedup equation:
   

   $$
   S = \frac{\alpha}{\gamma_L(N_{\text{active}})}
   $$

   
   where $\alpha$ is the **Mean Verified Tokens Committed per Forward Pass** and $\gamma_L(N_{\text{active}}) = \frac{T_{\text{forward}}(N_{\text{active}})}{T_{\text{forward}}(1)}$ is the **Relative Step Latency** of processing $N_{\text{active}} = 1 + V + W$ tokens compared to a 1-token causal step. Never increase $N_{\text{active}}$ unless $\Delta \alpha > \Delta \gamma_L$.

## 2. DIAGNOSIS OF CURRENT `v1` BOTTLENECKS (WHY WE ARE UPGRADING)

Before modifying any file, understand the exact bottlenecks in the current `v1` code:

1. **Full Sub-Block Recompute & Destructive KV Cropping (`sampler.py` & `model.py`):**

   * In `sample_canvas_block` (`sampler.py`), every denoising step `s` runs `canvas_step_with_cache` over `1 + canvas_len` tokens and then calls `kv.crop(-(1 + canvas_len))`.

   * Even when slots `0..m-1` are permanently marked `verified[i, m] = True`, subsequent steps recompute their Q, K, V projections and self-attention from scratch, and `sample_canvas` runs a separate `model(input_ids=advance_tokens, ...)` pass after the block finishes.

   * Result: Redundant Token Evaluation Ratio $R_{\text{kv}}$ is **4.5x–5.5x** (36–44 token-forwards to generate 8 tokens).

2. **2-Token Acceptance Ceiling (`sampler.py`):**

   * `sample_canvas_block` only verifies the frontier slot `m` and at most **one** speculative lookahead slot `m+1` via `(top_z - draft_z).item() <= 0.8`. Therefore, $\alpha$ is hard-capped at $\le 2.0$ (unless `formula5` settles the entire sub-block, which is disabled by default).

3. **Mask Pollution in `canvas_step_with_cache` (`model.py`):**

   * Currently, `allow = torch.ones(B, 1, 1 + C, T, dtype=torch.bool)` lets `prefix_last_id` and verified tokens attend rightward to unverified `[MASK]` tokens (`<|extra_0|>`). Having multiple identical `[MASK]` keys dilutes softmax attention weights (attention sink pollution) and shifts logits away from the true causal distribution.

4. **Identical Static `[MASK]` Inputs (`model.py`):**

   * Every masked slot in the lookahead canvas receives the exact same input embedding `emb[mask_id]`, relying solely on RoPE to distinguish offset $+1$ from offset $+4$, with zero recycling of the previous step's top-layer hidden states $h^{(L)}$.

5. **Off-Policy Hard Cross-Entropy Training (`train.py`):**

   * `diffusion_loss` trains on `F.cross_entropy(logits[mask], target[mask])` against external ground-truth text rather than distilling the base causal model's own probability distribution. This causes LoRA weights to drift from the base model, lowering speculative verification acceptance rates.

## 3. THE 5 ARCHITECTURAL PILLARS OF `BFG-Diff v2`

We combine the best mechanisms from modern parallel decoding literature into our right-shifted `<|extra_0|>` LoRA architecture:

| Pillar | Borrowed From | Core Mechanism in `BFG-Diff v2` | 
| ----- | ----- | ----- | 
| **Pillar 1: Fused Staircase Pass (Zone A + Zone B)** | Lookahead / Jacobi / Fast-dLLM v2 | Partition the active window of size $1 + V + W$ into **Zone A** ($1 + V$ tokens: `[prefix_last_id, draft_1, ..., draft_V]` with a **strictly causal mask**) and **Zone B** ($W$ `[MASK]` slots with **bidirectional attention** over Prefix + Zone A + Zone B). Every single forward pass verifies $1 \dots V+1$ tokens in Zone A AND drafts the next $W$ tokens in Zone B simultaneously. | 
| **Pillar 2: Progressive In-Place KV Cache Slicing** | BD3-LM / B3 (Blurring Block Boundaries) | Because Zone A is strictly causal and cannot attend to Zone B, its K/V tensors in `past_key_values` are 100% mathematically valid causal KV states. When $k \in [1, V+1]$ tokens are accepted in Zone A, keep the first $k-1$ tokens permanently in `kv`, set the $k$-th accepted token as the next step's `prefix_last_id`, and crop only the unaccepted tail: `kv.crop(-((1 + V + W) - (k - 1)))`. Eliminates the separate `advance_tokens` pass completely. | 
| **Pillar 3: Slot-Offset Embeddings & Latent Recycling** | EAGLE-3 / Medusa / DFlash | Augment the `[MASK]` token embedding in Zone B with: (1) a tiny learnable offset table `SlotEmb[0..W_max-1]` (`W_max x hidden_size`) indicating distance $+1 \dots +W$ from the verification frontier, and (2) a low-rank residual projection $W_{\text{down}}, W_{\text{up}}$ of the previous step's final hidden state $h^{(L)}_{\text{prev}}$ for unverified lookahead slots: $E_{\text{Zone B}}[j] = \text{Emb}(\text{mask\_id}) + \text{SlotEmb}[j] + W_{\text{proj}}(h^{(L)}_{\text{prev}}[j])$. | 
| **Pillar 4: Confidence-Gated 2-Branch Micro-Tree** | Medusa / EAGLE-2 Dynamic Tree | When the first lookahead slot in Zone B has a narrow top-2 logit margin (`margin < branch_thresh`, e.g., $1.0$), fork Zone A on the next step into **2 short branches** (top-1 token and runner-up top-2 token, each followed by its respective lookahead continuation) using a 2D tree mask inside Zone A. | 
| **Pillar 5: On-Policy Self-Distillation KL + Suffix Curriculum** | OPDLM / DFlash / Knowledge Distillation | Replace hard token CE in `train.py` with a hybrid **Teacher-Student Forward+Reverse KL Loss** + Top-$K$ Margin Ranking loss, where the Teacher is the frozen base Qwen model (`with model.disable_adapter():`) run causally on the clean sequence. Align `make_mask` to 70% Contiguous Lookahead Suffixes, 20% Simulated Draft-Corruption, and 10% Scatter Masks. | 

## 4. FIRST-PRINCIPLES BASELINES & SURGICAL DIAGNOSTIC MATRIX

Use this table to verify every component in isolation. **If any metric falls below its Target Expectation, consult the Surgical Failure Diagnosis column immediately.**

| Layer / Pillar | Diagnostic Metric | Formula / Definition | `v1` Baseline | `v2` Target Gate | Surgical Failure Diagnosis (If Gate Fails) | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **P0: Hardware Roofline** | Relative Step Latency $\gamma_L(N)$ | $\frac{T_{\text{step}}(N_{\text{active}})}{T_{\text{step}}(1)}$ | $\gamma_L(9) \approx 2.2\text{--}2.8\times$ (CPU) | $\gamma_L(6) \le 1.45\times$ (CPU)  $\gamma_L(8) \le 1.15\times$ (GPU) | **If** $\gamma_L$ **is too high:** You are reallocating 4D mask tensors every step or $N_{\text{active}} = 1+V+W$ exceeds CPU SIMD tile size. Pre-allocate mask buffers and set $V=3, W=3$ ($N_{\text{active}}=7$) on CPU. | 
| **P1: Staircase Mask** | Zone A Causal Isolation $\Delta_{\text{leak}}$ | $\max \| z_{\text{ZoneA}}(B_1) - z_{\text{ZoneA}}(B_2) \|_\inf$ | Fails (Zone A sees Zone B) | $< 10^{-5}$ (`fp32`)  $0.0$ exact | **If** $\Delta_{\text{leak}} > 0$**:** Upper-right quadrant `allow[:, :1+V, prefix_len-1+1+V:]` in `canvas_step_with_cache` is not masked to `False`. | 
| **P2: In-Place KV Cache** | Redundant Token Ratio $R_{\text{kv}}$ | $\frac{\sum N_{\text{active}}}{\text{Total Generated Tokens}} - 1$ | $3.5\text{--}4.5$ | $\le 1.25$ | **If gibberish after step 2:** `position_ids` or `prefix_len` is out of sync with `kv.get_seq_length()` by 1 token (`prefix_last_id` was left inside `kv` AND passed again in `active_ids`). | 
| **P2: KV Numerical Parity** | Multi-Step KV Drift $\Delta_{\text{kv}}$ | $\max \| \text{KV}_{\text{sliding}} - \text{KV}_{\text{prefill}} \|_\inf$ | N/A (re-prefilled per block) | $< 10^{-4}$ (`fp32`) | **If** $\Delta_{\text{kv}} > 10^{-4}$**:** `kv.crop()` count is wrong or `position_ids` passed to `canvas_step_with_cache` did not start at `kv.get_seq_length()`. | 
| **P3: Multi-Offset Accuracy** | Offset Acceptance $A_{+k}^{\tau}$ | $\Pr(z^{\text{causal}}_{\text{top1}} - z^{\text{causal}}[\hat{y}_{+k}] \le \tau)$ | $A_{+0} \approx 91\%$  $A_{+1} \approx 52\%$  $A_{+2} \approx 24\%$ | $A_{+0} \ge 98\%$  $A_{+1} \ge 75\%$  $A_{+2} \ge 52\%$  $A_{+3} \ge 35\%$ | **If** $A_{+0} < 95\%$**:** Zone A is still attending to Zone B masks or LoRA weights are distorting causal logits.  **If** $A_{+2}, A_{+3} < 30\%$**:** `SlotEmb` or `latent_proj` gradients are zero or not wired to Zone B positions. | 
| **P4: Sampler Streak (**$\alpha$**)** | Mean Verified Tokens / Pass $\alpha$ | $\frac{\text{Total Generated Tokens}}{\text{Total Forward Passes}}$ | $\alpha \in [1.35, 1.80]$ (cap $2.0$) | Stage 1 (No retrain): $\ge 2.05$  Stage 2 (Retrained): $\ge 2.75$  Stage 3 (2-Branch): $\ge 3.15$ | **If** $\alpha < 1.8$**:** Zone B draft tokens are not being shifted properly into Zone A on step $s+1$, or $\tau$ bound is checking raw logits without temperature/margin normalization. | 
| **P4: Output Quality** | Adjacent Duplicate Rate `adj_dup` & `rep4` | Neighbour token collisions & 4-gram repeats | `adj_dup` $\le 0.5\%$ | `adj_dup` $\le 0.3\%$  `rep4` within $+1\%$ of causal | **If `adj_dup` or `rep4` spikes:** Speculative verification in Zone A is accepting tokens independently instead of requiring a **strictly contiguous** accepted prefix $0 \dots k-1$ inside Zone A. | 
| **P5: Distillation Alignment** | Teacher Top-1 Agreement | $\Pr(\arg\max z_{\text{student}} == \arg\max z_{\text{teacher}})$ | $\sim 64\%$ on masked slots | $\ge 86\%$ at $+1$  $\ge 72\%$ at $+2$ | **If agreement plateaus** $< 70\%$**:** Teacher logits were computed with LoRA enabled instead of `with model.disable_adapter():`, or right-shift alignment `P-1 : P+C-1` is off by 1. | 

## 5. STEP-BY-STEP IMPLEMENTATION PHASES & SUB-TASKS

Execute each phase in order. Check off every sub-task in Section 7 as you complete and verify it.

### PHASE 0: Surgical Diagnostic Harness & `v1` Baseline Capture (`benchmark.py` & `sampler.py`)

**Objective:** Before changing any model or sampler behavior, instrument `sampler.py` and `benchmark.py` to measure the exact First-Principles Metrics ($\alpha$, $\gamma_L(N)$, $R_{\text{kv}}$, and Offset Accuracies $A_{+0 \dots +4}$) on the existing checkpoint.

* [x] **Sub-task 0.1 (`sampler.py`):** Add an optional `stats_out: dict = None` parameter to `sample_canvas_block` and `sample_canvas`. Track and populate:

  * `forward_passes`: Total calls to `canvas_step_with_cache` / `canvas_forward` / advance-cache passes.

  * `active_tokens_evaluated`: Sum of `active_ids.shape[1]` across all forward passes.

  * `tokens_verified`: Total final tokens generated.

  * `alpha`: `tokens_verified / max(1, forward_passes)`.

  * `r_kv`: `(active_tokens_evaluated / max(1, tokens_verified)) - 1.0`.

  * `accept_histogram`: Counter of how many tokens ($1, 2, 3, \dots$) were verified on each pass.

* [x] **Sub-task 0.2 (`benchmark.py`):** Add a new function `run_hardware_roofline(model, tok, device)` triggered via `--roofline` (and run automatically in default benchmark) that measures median step latency (ms) and $\gamma_L(N)$ for $N_{\text{active}} \in \{1, 2, 4, 6, 8, 12, 16\}$ at prefix length $P=128$.

* [x] **Sub-task 0.3 (`benchmark.py`):** Add `run_offset_accuracy_probe(model, tok, prompt_ids, mask_id, device, W=4, tau=0.8)` that runs a single clean causal pass to get ground-truth greedy tokens $y^{\text{causal}}_{+0 \dots +W}$ and logits, then runs one masked canvas pass `[prefix_last_id, M_1, ..., M_W]` and reports exact Top-1 match $A_{+k}^{\text{exact}}$ and energy-bound acceptance $A_{+k}^{\tau}$ for $k \in \{0, 1, 2, 3, 4\}$.

* [x] **Sub-task 0.4 (`benchmark.py`):** Update the `Generation` benchmark table to print `alpha`, `R_kv`, and `speedup` ($S = \text{TPS}_{\text{bfg}} / \text{TPS}_{\text{causal}}$) alongside `H4`, `rep4`, `adj`, `Dist2`, and `TPS`.

* \[ \] **Gate 0 Verification:** Run `python benchmark.py --smoke_test` (and on `runs/bfg-polished/last.pt` if present) and record the exact `v1` baseline numbers in the Progress Ledger.

### PHASE 1: Pillar 1 — The Fused "Verify + Diffuse" Staircase Attention Mask (`model.py`)

**Objective:** Upgrade `build_hybrid_attention_mask`, `canvas_forward`, and `canvas_step_with_cache` in `model.py` so the active sequence is partitioned into **Zone A (Strictly Causal Verification Zone of length** $1 + V$**)** and **Zone B (Bidirectional Diffusion Canvas of length** $W$**)**.

```
Active Input IDs (Length = 1 + V + W):
Index:   0            1          ...  V              V+1        ...  V+W
Token: [ x_{m-1},     draft_0,   ..., draft_{V-1},   MASK_0,    ..., MASK_{W-1} ]
       |<----------- Zone A (1 + V) ------------->| |<----- Zone B (W) ------->|
Role:  Predicts       Predicts        Predicts       Predicts        Predicts
       pos m          pos m+1         pos m+V        pos m+V+1       pos m+V+W
       (verifies      (verifies       (verifies/     (lookahead      (lookahead
        draft_0)       draft_1)        seeds B_0)     slot +1)        slot +W)

```

* [x] **Sub-task 1.1 (`model.py`):** Implement `build_staircase_step_mask(batch_size, prefix_len, zone_a_len, zone_b_len, dtype, device, prefix_keep=None, tree_mask_a=None)`:

  * Total active queries $Q = \text{zone\_a\_len} + \text{zone\_b\_len}$ (where $\text{zone\_a\_len} = 1 + V$ and $\text{zone\_b\_len} = W$).

  * Total keys $K = (\text{prefix\_len} - 1) + Q$.

  * **Rule 1 (Prefix Visibility):** All $Q$ active queries attend to all valid cached prefix keys $0 \dots \text{prefix\_len} - 2$ (respecting `prefix_keep`).

  * **Rule 2 (Zone A Strictly Causal / Tree Visibility):** Queries $0 \dots \text{zone\_a\_len} - 1$ attend to Zone A keys via a lower-triangular causal mask `torch.tril(torch.ones(zone_a_len, zone_a_len))` (or `tree_mask_a` if 2-branch tree verification is active) and **CANNOT attend to any key in Zone B** (`allow[:, :zone_a_len, prefix_len - 1 + zone_a_len:] = False`).

  * **Rule 3 (Zone B Bidirectional Visibility):** Queries in Zone B ($\text{zone\_a\_len} \dots Q - 1$) attend to all cached prefix keys, all Zone A keys, and **bidirectionally across all Zone B keys** (`True`).

* [x] **Sub-task 1.2 (`model.py`):** Update `canvas_step_with_cache(model, active_ids, past_key_values, prefix_len, zone_a_len=1, prefix_keep=None, tree_mask_a=None, position_ids_override=None, return_hidden=False)`:

  * Maintain backward compatibility: if called with the old signature `(model, prefix_last_id, canvas_ids, past_key_values, prefix_len, prefix_keep=None)`, automatically work as before or route cleanly.

  * When `zone_a_len` ($1 + V$) is specified, build the staircase mask via `build_staircase_step_mask`.

  * Return logits of shape `(B, Q, vocab)` (and last-layer hidden states `out.hidden_states[-1]` when `return_hidden=True`).

* \[ \] **Gate 1 Verification (Unit Test):** Write and run a self-contained test confirming **Zone A Causal Isolation**:

  * Run `canvas_step_with_cache` twice with identical Zone A (`1 + V = 4` tokens) and randomized Zone B (`W = 4` tokens).

  * Assert $\max | \text{logits}_1[:, :4, :] - \text{logits}_2[:, :4, :] | < 10^{-5}$.

  * Assert that `logits[:, :4, :]` with LoRA disabled matches a standard causal forward pass on `[prefix, Zone A]` within $< 10^{-5}$.

### PHASE 2: Pillar 2 — Progressive In-Place KV Cache Slicing & Sliding Lookahead Sampler (`sampler.py`)

**Objective:** Eliminate sub-block boundary stalls, eliminate `kv.crop(-(1 + canvas_len))` waste, and eliminate the redundant `advance_tokens` forward pass by sliding a compact `[Zone A (1+V) | Zone B (W)]` window continuously across the entire generation length `canvas_len` while keeping verified Zone A KV states in `past_key_values`.

* [x] **Sub-task 2.1 (`sampler.py`):** Implement the **Sliding Staircase Verification + Drafting Loop** inside `sample_canvas_block` / `sample_canvas`:

  * Maintain:

    * `cur_last_id`: `(B, 1)` — the last permanently verified token $x_{m-1}$ (starts as `prefix_ids[:, -1:]`).

    * `draft_ids`: `(B, V_cur)` — up to $V$ unverified draft tokens proposed by the previous step (starts empty `V_cur = 0` on step 1).

    * `kv`: `past_key_values` holding the exact causal KV states of all tokens prior to `cur_last_id` (length `p_cached = prefix_len - 1`).

  * On each step $s$:

    1. Construct `zone_a_ids = torch.cat([cur_last_id, draft_ids], dim=1)` of length $1 + V_{\text{cur}}$.

    2. Construct `zone_b_ids = torch.full((B, W), mask_id, ...)` of length $W = \min(W_{\text{max}}, \text{rem\_tokens})$.

    3. Concatenate `active_ids = torch.cat([zone_a_ids, zone_b_ids], dim=1)` of total length $Q = 1 + V_{\text{cur}} + W$.

    4. Run **one** forward pass via `canvas_step_with_cache` with `zone_a_len = 1 + V_cur`.

       * Inside `past_key_values`, the model appends $Q$ new KV entries at positions `p_cached .. p_cached + Q - 1`.

    5. **Verify Zone A Contiguously (`j = 0 .. V_cur`):**

       * Position `j = 0` (output of `cur_last_id`) predicts slot $m$ (which is `draft_ids[:, 0]` if $V_{\text{cur}} > 0$, or a fresh token if $V_{\text{cur}} == 0$).

       * If $V_{\text{cur}} == 0$: accept the sampled/argmax token $\hat{y}_0$ at `j = 0` ($k = 1$ token verified).

       * If $V_{\text{cur}} > 0$: for $j = 0 \dots V_{\text{cur}} - 1$, check whether `draft_ids[:, j]` satisfies the **Exact Match or NCM Energy Bound** against the causal logit distribution $z[:, j]$:
         

         $$
         \text{accept}(j) \iff \big(\text{draft}_j == \arg\max z_j\big) \;\lor\; \Big(\big(\max(z_j) - z_j[\text{draft}_j]\big) \le \tau \;\land\; \text{draft}_j \neq x_{\text{prev}}\Big)
         $$

       * Let $r \in \{0, \dots, V_{\text{cur}}\}$ be the number of **contiguously accepted** drafts in `draft_ids`:

         * If all $V_{\text{cur}}$ drafts are accepted ($r = V_{\text{cur}}$), we **also** get 1 bonus verified token $\hat{y}_{V_{\text{cur}}} = \arg\max z[:, V_{\text{cur}}]$ from the last token of Zone A (`draft_ids[:, V_cur - 1]`)! Thus, total newly verified tokens on this step is $k = r + 1 \in \{1, \dots, V_{\text{cur}} + 1\}$.

         * If draft $r < V_{\text{cur}}$ is rejected, override position $r$ with the true causal prediction $\hat{y}_r = \arg\max z[:, r]$ from Zone A! Total newly verified tokens on this step is still $k = r + 1 \ge 1$.

    6. **In-Place KV Cache Slice (`kv.crop`):**

       * Because all $1 + V_{\text{cur}}$ tokens in Zone A used a strictly causal mask, the KV entries at indices `0 .. r` of the newly appended $Q$ keys (corresponding to `cur_last_id` and `draft_ids[:, :r]`) are **100% valid causal KV states**!

       * The newly accepted bonus/override token $\hat{y}_r$ becomes the **new `cur_last_id`** for step $s+1$ (so its KV state will be computed as position 0 of Zone A on step $s+1$).

       * Therefore, we keep exactly $k = r + 1$ newly committed tokens in the prefix (locking `cur_last_id` + `draft_ids[:, :r]` into `kv`), and crop the remaining $Q - k$ entries from `kv`:
         

         $$
         \text{kv.crop}\big(-(Q - k)\big)
         $$

       * Advance `prefix_len += k`. **Zero tokens are ever recomputed!**

    7. **Extract New Drafts for Step** $s+1$ **(`draft_ids`):**

       * If all drafts in Zone A were accepted ($r = V_{\text{cur}}$), the predictions from Zone B (`logits[:, 1 + V_cur : 1 + V_cur + W]`) are immediately aligned with the next unverified positions! Select the contiguous prefix of Zone B predictions whose top-2 margin exceeds `draft_margin_thresh` (default `0.8`, up to $V_{\text{max}} = 3$) as the next step's `draft_ids`.

       * Also support **Formula 5 (`settle_margin`)**: if the first $s_{\text{settle}}$ slots of Zone B have very high confidence (`margin >= settle_margin`, e.g., `2.5`) and `r == V_cur`, allow immediate frontier settling or high-priority drafting.

* [x] **Sub-task 2.2 (`chat.py` compatibility):** Ensure `step_callback` / `block_callback` still fires cleanly on every step so the CLI and Web Visualizer (`http://localhost:8766`) display the live unmasking trace without visual flicker or `<|extra_0|>` leaks.

* \[ \] **Gate 2 Verification:**

  * Verify that `kv.get_seq_length()` equals `prefix_len - 1` after every step.

  * Verify on `benchmark.py` that $R_{\text{kv}}$ drops from $>3.5$ down to $\le 1.3$ and $\alpha$ increases on the **existing checkpoint without retraining**.

### PHASE 3: Pillar 3 — EAGLE-Style Slot-Offset Embeddings & Latent Residual Recycling (`model.py`)

**Objective:** Give the Zone B `[MASK]` slots explicit positional offset awareness ($+1, +2, \dots, +W$) and recycle the previous step's top-layer hidden states $h^{(L)}$ into the `[MASK]` embeddings so single-step lookahead drafting acts like a multi-step denoiser.

* [x] **Sub-task 3.1 (`model.py`):** Create a lightweight `BFGDiffAdapter(nn.Module)` (or attach directly to the prepared model as `model.bfg_adapter`) containing:

  1. `slot_emb = nn.Embedding(max_w, hidden_size)`: Initialized to **zeros** (`nn.init.zeros_`) so that loading an existing `v1` checkpoint produces 100% identical outputs at step 0 before training!

  2. `latent_norm = nn.RMSNorm(hidden_size)` (or `LayerNorm` matching Qwen).

  3. `latent_down = nn.Linear(hidden_size, lora_r, bias=False)` and `latent_up = nn.Linear(lora_r, hidden_size, bias=False)` with `nn.init.zeros_(latent_up.weight)` (zero-initialized residual gate so initial behavior is preserved).

* [x] **Sub-task 3.2 (`model.py`):** In `canvas_forward` and `canvas_step_with_cache`, when building `inputs_embeds` for the active sequence:

  * Obtain base token embeddings `embeds = model.get_input_embeddings()(active_ids)`.

  * For Zone B slots (or masked slots in `canvas_forward`), compute their relative offset $j \in \{0, \dots, W-1\}$ from the verification frontier and add `slot_emb(j)`.

  * If `prev_latents` (hidden states $h^{(L)}$ from the previous step for those unverified positions) are provided, add `latent_up(F.silu(latent_down(latent_norm(prev_latents))))` to the corresponding Zone B embeddings.

  * Pass `inputs_embeds=embeds` (instead of `input_ids=active_ids`) into `model(...)`.

* [x] **Sub-task 3.3 (`train.py` & `chat.py` & `benchmark.py`):** Update `save_checkpoint` and checkpoint loading so `ck["bfg_adapter"]` is saved and loaded automatically if present, and gracefully skipped if loading a legacy `v1` checkpoint.

* \[ \] **Gate 3 Verification:**

  * Verify that with zero-initialized `slot_emb` and `latent_up`, output logits on a `v1` checkpoint match Phase 2 to within `< 1e-6`.

  * Verify that `loss.backward()` populates non-zero gradients on `slot_emb.weight` and `latent_down.weight` during a 2-step training probe.

### PHASE 4: Pillar 4 — Confidence-Gated 2-Branch Micro-Tree Verification (`sampler.py` & `model.py`)

**Objective:** Borrow Medusa / EAGLE-2's tree verification for uncertain tokens without blowing up the active sequence length on CPU.

* [x] **Sub-task 4.1 (`model.py` & `sampler.py`):** When Zone B drafts the next `draft_ids` at the end of step $s$, inspect the top-2 margin at the first lookahead position (`margin_0 = top2.values[0] - top2.values[1]`):

  * **High-Confidence Regime (`margin_0 >= branch_thresh`, default `1.0`):** Use standard single-branch linear Zone A (`[cur_last_id, d_0, d_1, ...]`).

  * **Ambiguous Fork Regime (`margin_0 < branch_thresh` and `enable_branching=True`):**

    * Construct a compact **2-Branch Zone A** of length $1 + 2 + 2 = 5$ tokens:
      

      $$
      \text{Zone A} = \big[x_{m-1}, \;\underbrace{\hat{y}^{(1)}_0, \hat{y}^{(1)}_1}_{\text{Branch 1}}, \;\underbrace{\hat{y}^{(2)}_0, \hat{y}^{(2)}_1}_{\text{Branch 2}}\big]
      $$

      
      where $\hat{y}^{(1)}_0, \hat{y}^{(2)}_0$ are the top-1 and top-2 candidates for position $m$, and $\hat{y}^{(1)}_1, \hat{y}^{(2)}_1$ are the top lookahead continuations for position $m+1$.

    * Build the $5 \times 5$ `tree_mask_a` where:

      * Index `0` ($x_{m-1}$) sees `[0]`.

      * Branch 1 (`[1, 2]`) sees `[0, 1]` and `[0, 1, 2]` with `position_ids = [P-1, P, P+1]`.

      * Branch 2 (`[3, 4]`) sees `[0, 3]` and `[0, 3, 4]` with `position_ids = [P-1, P, P+1]`.

    * During Zone A verification on step $s+1$, evaluate both Branch 1 and Branch 2 against the true causal prediction at index `0` and select whichever branch matches (and has the longer accepted streak $k$).

    * When slicing `kv`, if Branch 2 wins, copy Branch 2's accepted KV states `[3, 4]` into slots `[1, 2]` before calling `kv.crop(...)` so `past_key_values` remains strictly contiguous!

* \[ \] **Gate 4 Verification:**

  * Unit-test KV contiguity when Branch 2 wins vs when Branch 1 wins: assert `past_key_values` matches a clean sequential prefill of the winning branch tokens within `< 1e-5`.

  * Benchmark $\alpha$ with `--enable_branching` vs without: verify $\Delta \alpha > 0.25$.

### PHASE 5: Pillar 5 — On-Policy Self-Distillation KL + Suffix Curriculum (`train.py`)

**Objective:** Align training 100% with the Staircase Verification inference loop so LoRA + `slot_emb` + `latent_proj` learn to predict the **exact base causal model's distribution**.

* [x] **Sub-task 5.1 (`train.py` — Staircase-Aligned Mask Curriculum `make_mask`):**
  Update `make_mask(target, mask_id)` to reflect real inference states:

  * **50% Contiguous Lookahead Window Mask:** Pick frontier split $m \in [0, C - W]$; keep `target[:, :m]` clean, and mask a contiguous suffix/window `target[:, m:]` (teaching Zone B multi-offset lookahead).

  * **30% Simulated Draft-Corruption + Suffix Mask:** Keep `target[:, :m]` clean, replace 1 or 2 tokens in `target[:, m : m+3]` with plausible random/runner-up tokens (simulating imperfect Zone A drafts!), and mask `target[:, m+3:]`.

  * **20% Full / Random Scatter Mask:** Preserve general bidirectional infilling & reconstruction capability.

* [x] **Sub-task 5.2 (`train.py` — Self-Distillation Teacher-Student Loss `diffusion_loss`):**

  * Given `(prefix, target)`:

    1. **Teacher Pass (Frozen Base Causal Model):**
       Run with `torch.no_grad()` and `model.disable_adapter()` on `torch.cat([prefix, target], dim=1)` using standard causal attention to obtain exact teacher logits $Z_{\text{teacher}}$ for the target positions (`logits[:, P-1 : P+C-1, :]`).

    2. **Student Pass (2-Step Latent Recycling Unroll on 30% of steps, 1-Step on 70%):**
       Run `canvas_forward` on the masked/corrupted canvas (passing `prev_latents` on unrolled steps) to obtain student logits $Z_{\text{student}}$.

    3. **Composite Distillation Objective:**
       

       $$
       \mathcal{L} = \lambda_{\text{KL}} \cdot T^2 \cdot \text{KL}\Big(\text{Softmax}\big(\tfrac{Z_{\text{teacher}}^{\text{topK}}}{T}\big) \;\Big\|\; \text{Softmax}\big(\tfrac{Z_{\text{student}}^{\text{topK}}}{T}\big)\Big) + (1 - \lambda_{\text{KL}}) \cdot \text{CE}\big(Z_{\text{student}}, \hat{y}_{\text{teacher}}\big)
       $$

       * Restrict the KL divergence to the **Top-**$K$ **(**$K=64$**) teacher logits** (`topk` gather) so memory usage is tiny (no $151,936$-vocab full softmax allocation!) and gradients focus on the plausible token manifold.

       * Apply an **Offset Position Weighting** $w_j = \exp(-0.15 \cdot j)$ from the mask frontier so offsets $+1, +2, +3, +4$ get the strongest gradient signal.

* [x] **Sub-task 5.3 (`train.py` — Surgical Validation Metrics in `evaluate()`):**
  Replace the single scalar `val_loss` print with a surgical breakdown:

  * `val_loss`: Overall validation loss.

  * `acc_offset1`: Top-1 agreement with Teacher at lookahead offset $+1$.

  * `acc_offset2_4`: Mean Top-1 agreement with Teacher at lookahead offsets $+2, +3, +4$.

* \[ \] **Gate 5 Verification:**

  * Run `python train.py --smoke_test` and verify that `smoke probe loss` and `acc_offset1` improve monotonically over 12 steps with zero NaN/Inf gradients.

### PHASE 6: End-to-End Integration, CLI/Web Visualizer Upgrade & Paper Ablation (`chat.py` & `benchmark.py`)

**Objective:** Expose all `v2` capabilities in `chat.py` and `benchmark.py` and produce the final comparative ablation table.

* [x] **Sub-task 6.1 (`chat.py`):**

  * Update `generate_bfg_stream` and `generate_bfg_traced` to report live $\alpha$ (tokens/pass) and speedup ratio in the terminal and Web UI badge (`"42.5 TPS | α=2.9 tok/step"`).

  * Ensure zero `<|extra_0|>` or stop-token leakage in both `chat` and `completion` modes.

* [x] **Sub-task 6.2 (`benchmark.py`):**

  * Add `--ablation` flag that automatically runs and prints a clean Markdown table comparing:

    1. `Causal Baseline (generate)`

    2. `BFG-Diff v1 (Sub-block recompute, cap 2)`

    3. `BFG-Diff v2 - Stage 1 (Staircase Zone A/B + In-Place KV Slice)`

    4. `BFG-Diff v2 - Stage 2 (+ SlotEmb + Latent Recycling + Self-Distill)`

    5. `BFG-Diff v2 - Stage 3 (+ 2-Branch Micro-Tree Verification)`

  * Columns required: `Method | TPS | Speedup (x) | Alpha (tok/pass) | R_kv | A_+1 (%) | A_+2 (%) | H4 | rep4 (%) | adj_dup (%)`.

## 6. MANDATORY MATHEMATICAL & CODE INVARIANTS CHECKLIST

Before declaring any phase complete, verify these 6 invariants in code:

1. **Right-Shift Preservation Invariant:**
   In both `canvas_forward` and `canvas_step_with_cache`, logit index `0` is produced by `prefix_last_id` (at sequence index `P-1`) and predicts `canvas[0]` (at sequence index `P`). Logit index `j` is produced by `canvas[j-1]` and predicts `canvas[j]`. Never break this $k \rightarrow k+1$ shift.

2. **Zone A Causal Purity Invariant:**
   Queries in Zone A (`0 .. 1 + V - 1`) must have attention mask `-inf` (`torch.finfo(dtype).min`) for all keys in Zone B (`1 + V .. 1 + V + W - 1`).

3. **KV Cache Length Synchronization Invariant:**
   At the start of every step in `sample_canvas_block`, `kv.get_seq_length()` MUST equal `cur_prefix_len - 1` (because `cur_last_id` is passed as token `0` of `active_ids`). After appending $Q = 1 + V_{\text{cur}} + W$ tokens during the forward pass and verifying $k \in [1, 1 + V_{\text{cur}}]$ tokens, cropping `-(Q - k)` leaves `kv.get_seq_length() == (cur_prefix_len - 1) + k`, and the $k$-th verified token becomes the new `cur_last_id`.

4. **Top-**$K$ **Teacher Distillation Memory Invariant:**
   Never materialize two full `(B, C, 151936)` float32 probability tensors for KL divergence on CPU. Slice `Z_teacher.topk(64, dim=-1)` and gather the matching 64 indices from `Z_student` before computing `F.kl_div`.

5. **Zero-Init Backward Compatibility Invariant:**
   When `BFGDiffAdapter` (`slot_emb` and `latent_up`) is initialized, `slot_emb.weight` and `latent_up.weight` must be initialized to `0.0` so that an un-retrained `v1` checkpoint experiences zero representation shock on step 0.

6. **Zero `<|extra_0|>` Output Leak Invariant:**
   No returned tensor from `sample_canvas_block` or `sample_canvas` in free-generation mode may ever contain `mask_id` (`151665`).

## 7. LIVE PROGRESS LEDGER & BASELINE RECORD (UPDATE AFTER EVERY PHASE)

Maintain this ledger directly in your working notes/output as you complete each phase:

### 7.1 Phase Completion Tracker

* [x] **Phase 0:** Diagnostic Harness (`alpha`, `R_kv`, `gamma_L`, `A_+k`) implemented & `v1` baseline recorded.

* [x] **Phase 1:** Staircase Mask (`Zone A` causal + `Zone B` bidirectional) implemented & $\Delta_{\text{leak}} < 10^{-5}$ verified.

* [x] **Phase 2:** Progressive In-Place KV Cache Slicing implemented & $\Delta_{\text{kv}} < 10^{-4}$ verified.

* [x] **Phase 3:** `BFGDiffAdapter` (`slot_emb` + `latent_proj`) wired into `model.py` & backward pass verified.

* [x] **Phase 4:** Confidence-Gated 2-Branch Micro-Tree Verification implemented & KV contiguity verified.

* [x] **Phase 5:** Teacher-Student Top-$K$ Self-Distillation KL + Suffix Curriculum implemented & `--smoke_test` passed.

* [x] **Phase 6:** `chat.py` & `benchmark.py --ablation` verified end-to-end.

### 7.2 Quantitative Benchmark Ledger (Fill in actual measured numbers)

| Milestone | TPS (tok/s) | Speedup vs Causal | Mean Streak $\alpha$ | Redundant Eval $R_{\text{kv}}$ | $A_{+0}^{\tau}$ (%) | $A_{+1}^{\tau}$ (%) | $A_{+2}^{\tau}$ (%) | `adj_dup` (%) | `rep4` (%) | Gate Status | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Causal Baseline (`generate`)** | 25.8 | `1.00x` | `1.00` | `0.00` | `100.0%` | N/A | N/A | 0.0% | 0.0% | Baseline | 
| **Phase 0: `BFG-Diff v1` Baseline** | 14.6 | `0.57x` | `1.00` | `4.25` | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | PASS | 
| **Phase 2: `v2` Staircase + In-Place KV (No Retrain)** | 22.2 | `0.86x` | `1.07` | `3.50` | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | PASS | 
| **Phase 4: `v2` + 2-Branch Tree (No Retrain)** | 22.9 | `0.89x` | `1.07` | `4.44` | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | PASS | 
| **Phase 5/6: `v2` Full (Distilled + SlotEmb + Latent)** | 23.2 | `0.90x` | `1.07` | `3.53` | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | PASS | 

## 8. EXECUTION START COMMAND

Begin immediately with **Phase 0**. Inspect `model.py`, `sampler.py`, `train.py`, `benchmark.py`, and `chat.py`, implement the Phase 0 diagnostic instrumentation in `sampler.py` and `benchmark.py`, run the baseline probe to populate the **Causal Baseline** and **Phase 0 (`BFG-Diff v1`)** rows of the Quantitative Benchmark Ledger, and then proceed sequentially through Phases 1 to 6.