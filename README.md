# omega-engine

**Universal, Model-Agnostic Pure Seeded-OM-2 Zero-CUDA Inference Engine for NVIDIA Blackwell (DGX Spark GB10, SM 12.1).**

`omega-engine` is a single **`811 KB` static AArch64 ELF** (`ldd: not a dynamic executable` — zero `libcuda.so.1`, zero cuBLAS/cuDNN, zero `libc`, zero C, zero Rust, zero Python) compiled directly from pinned `omega-mv` (`om/gpu/engine/`) by the self-hosted seeded compiler (`seed/out/omc`).

Instead of hardcoding a single model's dimensions or tensor byte offsets, `omega-engine` reads any standard HuggingFace `config.json`, dynamically indexes safetensors headers across all three checkpoint layouts (up to `163,840` tensors via O(1) FNV-1a hash table), emits **shape-specialized SM121 Blackwell SASS machine code at runtime**, quantizes BF16 or FP8 weights to block-32 W4 on GPU, runs **Full-APU (20 ARMv9 CPU cores + 48 Blackwell SMs = 6,164 cores simultaneously)** with **Live Arboris online co-activation expert prefetching enabled by default**, and serves an OpenAI-compatible HTTP API with a 128K context window.

> **Driver & Firmware Disclosure:** "Zero CUDA" means zero NVIDIA userspace libraries or CUDA toolkit. Execution still opens `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia-uvm` through Linux kernel open modules (`nvidia` / `nvidia-uvm` `580.173.02`) and NVIDIA's signed GSP firmware.

---

## Architecture (`om/gpu/engine/` in `omega-mv`)

1. **`config.om` — Universal `ModelSpec` & `config.json` Parser:**
   Auto-detects `model_type`, `hidden_size`, `intermediate_size`, `moe_intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `num_experts` / `top_k`, `full_attention_interval`, `linear_num_key_heads` / `linear_num_value_heads`, `partial_rotary_factor`, `rope_theta`, `tie_word_embeddings`, `attn_output_gate`, RMSNorm convention (`(1+w)` for Qwen 3.5/3.6/3.8/4 & Gemma vs. `w` for Llama 3 / Qwen 2.5 / Qwen 3 / Mistral / DeepSeek), and quantization scheme (`BF16`, `FP8`, `modelopt`).
2. **`safetensors.om` — Universal 3-Layout Tensor Registry (Zero Hardcoded Offsets):**
   Automatically discovers and indexes tensors by canonical name (`model.*` and `model.language_model.*`) across:
   - Single-file HuggingFace (`model.safetensors`)
   - Sharded HuggingFace (`model.safetensors.index.json` + `model-00001-of-*.safetensors`)
   - Omega per-layer packed (`outside.safetensors` + `layers-L.safetensors`)
3. **`emit_generic.om` & `emit_qwen4.om` — Runtime Shape-Parameterized SM121 SASS Generator:**
   Emits specialized SM121 SASS kernels for the loaded `ModelSpec`:
   - `uW4Gemv4R` (4 KiB streaming SMEM W4 GEMV for `K % 1024 == 0`) & `uW4GemvAnyK` (universal W4 block-32 GEMV for any `K % 128 == 0`, any `N`)
   - `uBf16ToW4Quant` (on-GPU BF16 → W4 block-32 quantizer for any `K % 64 == 0`, any `N`) & `qFp8ToW4B32Quant` (on-GPU FP8 block-128 → W4 block-32)
   - `uBf16Gemv1R` (`K % 256 == 0`) & `uBf16GemvAnyK` (universal BF16 GEMV for any `K % 64 == 0`)
   - `uRmsNormAdd(H, addOne, eps)`, `uGroupedRmsNorm(hc_count, H)`, & `uSwiGlu(I)`
   - `uHcMixReduce` / `uHcInjectAdd` / `uScaleSilu` (4-stream Gated Residual Hyper-Connections)
   - `uPleGate` / `uPleDilatedConvAdd` (51B N-gram Per-Layer Embedding with dilated depthwise Conv1d `K=4, dilation=3`)
   - `uRouterTopK` (512-expert softmax + top-10 + `norm_topk_prob`), `uMoeW4Gu` (fused top-K gate+up+SiLU*up), & `uMoeW4DnAdd` (fused top-K down_proj + weighted sum + gated shared expert)
   - `uFaPrepKv` / `uFaPrepQ` / `uFaAttendSplitF32(nQ, nKV, hd)` / `uFaReduceGate(hd, attnGate)` (Split-K `S=8` Flash-Decoding for any GQA ratio `nQ/nKV` and `hd` in `{128, 256}`, gated or ungated)
   - `uGdnStepReal(nK, nV, gdnSigmoid)` (30 KiB SMEM Gated DeltaNet supporting both `silu` and `sigmoid` output gates)
   - `uBf16RescoreCands(H)` (Two-Pass Exact `lm_head`) & `uArgmaxEmbedStep(H)` (GPU-side ZOS argmax + embed + `pos++`)
4. **`arboris_q4.om`, `runtime.om`, `runtime_qwen4.om` & `cli.om` — Full-APU + Default Live Arboris Runtime, First-Fit UVM Slab Allocator, CLI & OpenAI HTTP Server:**
   - **Full-APU (`20` ARMv9 CPU cores + `48` Blackwell SMs = `6,164` cores):** overlaps 16-core parallel NVMe 51B N-gram PLE lookup with Layer 0 on the 48 SMs, and splits the 248,320-vocab `lm_head` across all 20 CPU cores + 48 SMs simultaneously over uncached host-coherent NVLink-C2C shm.
   - **Live Arboris (ON by default):** 48-layer × 512-expert × top-10 online adaptive Space-Saving co-activation graph (`arboris_q4.om`), asynchronously overlapped with `lm_head` for zero critical-path latency.
   - `omega-engine inspect <model-dir>` (CPU-only: inspects any checkpoint, prints `ModelSpec`, tensor count, and kernel plan)
   - `omega-engine run <model-dir> [gen_tokens]` (specializes kernels, loads weights by name, runs live GB10 decode)
   - `omega-engine serve <model-dir> [--port PORT] [--self-test]` (`GET /health`, `GET /v1/models`, `POST /v1/chat/completions`)

---

## Verified Multi-Model Inspection & Live GB10 Silicon Proof

### 1. Supported Models & Multi-Model Auto-Detection (`omega-engine inspect`, Spark AArch64)

| Checkpoint | Auto-Detected `model_type` | Architecture Family | Dimensions | Layers (FA / GDN) | Layout & Indexed Tensors |
|---|---|---|---|---|---|
| **`qwen3.8-flash-next-fp8` (180B)** | `qwen4_exp_text` | Hybrid GDN(`sig`)+QSA MoE (`512` exp, top-10) + 4-Stream HC + 51B NVMe PLE + **Full-APU + Live Arboris** | `H=2560`, `moe_I=640`, `24Q/2KV`, `hd=256`, `rope=64` | `48` (`12` FA / `36` GDN) | `sharded-hf` (`131` files, `148,655` tensors) |
| `qwen3.8-27b-fp8` | `qwen3_5_text` | Hybrid GDN+FA Dense | `H=5120`, `I=17408`, `24Q/4KV`, `hd=256`, `rope=64` | `64` (`16` FA / `48` GDN) | `omega-per-layer` (`65` files, `1,251` tensors) |
| `qwen3.6-35b-a3b-fp8` | `qwen3_5_moe_text` | Hybrid GDN+FA MoE (`256` exp, top-8) | `H=2048`, `moe_I=512`, `16Q/2KV`, `hd=256`, `rope=64` | `40` (`10` FA / `30` GDN) | `omega-per-layer` (`41` files, `62,303` tensors) |
| `inkling-2-mini` | `inkling2_loop_text` | Standard Transformer (`128` exp, top-8) | `H=2304`, `I=1152`, `24Q/8KV`, `hd=128`, `rope=128` | `24` (`24` FA / `0` GDN) | `sharded-hf` (`1` file, `822` tensors) |

### 2. Live Quiet DGX Spark GB10 Silicon Verification

#### A. Qwen 3.8 Flash Next (180B: 125B MoE + 51B PLE + 4B MTP) — Full-APU + Default Live Arboris

Measured on **NVIDIA DGX Spark (GB10, `sm_121`, driver `580.173.02`)** inside exclusive quiet `spark-window` `20261125T013853Z-ecbea3` (`exit: 0`) on `2026-11-24` — [receipt JSON](evidence/gb10-omega-engine-qwen38-flash-next-apu.6c9a9b62faa1408fece3460b12be46ef21cf2f7ca709bab8be6d63fd65608937.json):

| Metric | Measured Result (`omega-engine serve ~/models/qwen3.8-flash-next-fp8-7fb48d70 --self-test`) |
|---|---|
| Binary | `omega-engine` (`811 KB` static ELF, `ldd: not a dynamic executable`, SHA-256 `1901511f3898bd21518bb2b7484f06d6eefc80bd026bd67acb43b89473992f32`) |
| Dynamic Tensor Resolution | `148,655` tensors across `131` HF shards resolved in `0.224 s` purely by canonical name (zero hardcoded offsets) |
| Execution Mode | **Full-APU** (`20` ARMv9 CPU cores: `10× X925 + 10× A725` + `48` Blackwell SMs = `6,164` cores simultaneously) + **Live Arboris ON by default** |
| Memory Footprint (First-Fit UVM Slabs + NVMe 51B PLE) | Fits cleanly inside Spark's 128 GB unified memory (`~77 GiB` resident incl. 128K KV; `32.5 GiB` free headroom; `0 GiB` GPU RAM for 51B PLE table) |
| Live Output | **`The capital of France is **Paris**.`** |
| Decode Speed (180B model, `N=1`) | **`13.64 tok/s`** (`73.29 ms/tok`) |
| Live Arboris Hit Rate (48L × 512E top-10) | **`41.6%`** (`arboris_hit_pct_x10=416`, asynchronously overlapped with `lm_head`, 0 critical-path latency) |

#### B. Qwen 3.8 27B Dense — Generic Zero-CUDA Path

Measured on **NVIDIA DGX Spark (GB10, `sm_121`, driver `580.173.02`)** inside exclusive quiet `spark-window` `20261124T225745Z-ab9d16` (`exit: 0`, `nvrm_errors_delta: 0`) on `2026-11-24` — [receipt JSON](evidence/gb10-omega-engine-verify.78669f86b247f11f792f89de9cad471f5de52887f92925d58894805556241e13.json):

| Metric | Measured Result (`omega-engine serve /home/aienalpha/models/qwen3.8-27b-fp8-017b9c7a --self-test`) |
|---|---|
| Binary | `omega-engine` (`671 KB` static ELF, `ldd: not a dynamic executable`, SHA-256 `e5e59e89be8001109198d17199b22673384de7603dc97654b5a6cfea75b97da4`) |
| Dynamic Tensor Resolution | `1,251` tensors across `65` files resolved purely by canonical name (zero hardcoded offsets) |
| Load + On-GPU W4 Quant + Runtime SM121 SASS Emit | `30.26 s` (`30,259 ms`) |
| Live Output | `The capital of France is **Paris**.` |
| Decode Speed (unbatched `N=1` generic path) | **`11.80 tok/s`** (`84.71 ms/tok`) |
| Kernel Driver Errors | **`nvrm_errors_delta: 0`** |

---

## Companion Model-Tuned Repositories (With Full Cross-Stack Benchmarks)

For peak hand-tuned throughput (HMMA Tensor-Core chunked prefill, Split-K `S=16` batched FA2, Continuous Batch-4/8, Speculative Verify / Native MTP, Live Arboris, and Full-Chip GB10 APU `4.4+ T ops/s`), see the dedicated model repositories:
- **[omega-qwen36-35b](https://github.com/drakestapleton/omega-qwen36-35b)** — Qwen 3.6 35B-A3B MoE (`67.47 tok/s` prose, `75.99 tok/s` MTP, `241.47 tok/s` Batch-8, `1,841 tok/s` prefill, vs. vLLM & SGLang).
- **[omega-qwen38-27b](https://github.com/drakestapleton/omega-qwen38-27b)** — Qwen 3.8 27B Dense (`13.69 tok/s` prose, `42.00 tok/s` Batch-8, `42.11 tok/s` Spec-8, `268 tok/s` prefill, vs. TandemLLM & PyTorch).

---

## Quick Start (DGX Spark AArch64)

```sh
# 1. Clone pinned omega-mv and build the 671 KB static omega-engine binary
sh scripts/build.sh

# 2. Inspect any HuggingFace or Omega model directory (CPU-only)
sh scripts/inspect.sh /path/to/model

# 3. Serve any supported Dense / Hybrid model via OpenAI-compatible HTTP API
sh scripts/serve.sh /path/to/model --port 18095
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).
