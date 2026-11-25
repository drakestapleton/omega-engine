# omega-engine

**Universal, Model-Agnostic Pure Seeded-OM-2 Zero-CUDA Inference Engine for NVIDIA Blackwell (DGX Spark GB10, SM 12.1).**

`omega-engine` is a single **`988 KB` static AArch64 ELF** (`ldd: not a dynamic executable` — zero `libcuda.so.1`, zero cuBLAS/cuDNN, zero `libc`, zero C, zero Rust, zero Python) compiled directly from pinned `omega-mv` (`om/gpu/engine/`) by the self-hosted seeded compiler (`seed/out/omc`).

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
   - `uW4Gemv4R` (4 KiB streaming SMEM W4 GEMV for `K % 1024 == 0`), `uW4Gemv4R512` (2 KiB streaming SMEM 4-row/warp `LDG.E.64` W4 GEMV for any `K % 512 == 0`), `uW4GemmB8` (true batched `B=8` W4 GEMM reading weights once across 8 streams), & `uW4GemvAnyK` (universal W4 block-32 GEMV for any `K % 128 == 0`, any `N`)
   - `uBf16ToW4Quant` (on-GPU BF16 → W4 block-32 quantizer for any `K % 64 == 0`, any `N`) & `qFp8ToW4B32Quant` (on-GPU FP8 block-128 → W4 block-32)
   - `uBf16Gemv4R` (4-row/warp `LDG.E.64` SMEM BF16 GEMV for `K % 128 == 0`), `uBf16GemmB8` (true batched `B=8` BF16 GEMM), `uBf16Gemv1R` (`K % 256 == 0`), & `uBf16GemvAnyK` (universal BF16 GEMV for any `K % 64 == 0`)
   - `uRmsNormAdd(H, addOne, eps)`, `uGroupedRmsNorm(hc_count, H)`, & `uSwiGlu(I)`
   - `uHcMixReduce` / `uHcInjectAdd` / `uScaleSilu` (4-stream Gated Residual Hyper-Connections)
   - `uPleGate` / `uPleDilatedConvAdd` (51B N-gram Per-Layer Embedding with dilated depthwise Conv1d `K=4, dilation=3`)
   - `uRouterTopK` (512-expert softmax + top-10 + `norm_topk_prob`), `uMoeW4Gu` (fused top-K gate+up+SiLU*up), & `uMoeW4DnAdd` (fused top-K down_proj + weighted sum + gated shared expert)
   - `uFaPrepKv` / `uFaPrepQ` / `uFaAttendSplitF32(nQ, nKV, hd)` / `uFaReduceGate(hd, attnGate)` (Split-K `S=8` Flash-Decoding for any GQA ratio `nQ/nKV` and `hd` in `{128, 256}`, gated or ungated)
   - `uGdnStepReal(nK, nV, gdnSigmoid)` (30 KiB SMEM Gated DeltaNet supporting both `silu` and `sigmoid` output gates)
   - `uBf16RescoreCands(H)` (Two-Pass Exact `lm_head`) & `uArgmaxEmbedStep(H)` (GPU-side ZOS argmax + embed + `pos++`)
4. **`arboris_q4.om`, `runtime.om`, `runtime_qwen4.om`, `runtime_qwen4_b8.om` & `cli.om` — Full-APU + Default Live Arboris + True `B=8` Multi-Stream Runtime, First-Fit UVM Slab Allocator, CLI & OpenAI HTTP Server:**
   - **Full-APU (`20` ARMv9 CPU cores + `48` Blackwell SMs = `6,164` cores):** overlaps 16-core parallel NVMe 51B N-gram PLE lookup with Layer 0 on the 48 SMs, and splits the 248,320-vocab `lm_head` across all 20 CPU cores + 48 SMs simultaneously over uncached host-coherent NVLink-C2C shm.
   - **True `B=8` Multi-Stream Batching (`runtime_qwen4_b8.om` + `ecServeB8`):** partitions the existing 6 GiB 128K KV cache into `8 × 16,384` tokens (0 extra KV RAM), runs batched `B=8` dense W4/BF16 GEMMs and a single-pass `B=8` `lm_head` over all `248,320` vocab, and automatically uses the fast zero-overhead `B=1` path at `C=1` and batched `B=8` prefill + decode at `C >= 2`.
   - **Live Arboris (ON by default):** 48-layer × 512-expert × top-10 online adaptive Space-Saving co-activation graph (`arboris_q4.om`), asynchronously overlapped with `lm_head` for zero critical-path latency.
   - `omega-engine inspect <model-dir>` (CPU-only: inspects any checkpoint, prints `ModelSpec`, tensor count, and kernel plan)
   - `omega-engine ktest <model-dir>` (1-second bit-exact SM121 kernel parity verifier for `uW4Gemv4R512`, `uBf16Gemv4R`, `uW4GemmB8`, `uBf16GemmB8`)
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
| Decode Speed (180B model, `N=1`) | **`13.64 tok/s`** (`73.29 ms/tok` self-test) / **`16.38–17.48 tok/s`** sustained HTTP decode |
| Live Arboris Hit Rate (48L × 512E top-10) | **`41.6%`** cold → **`45.7%–47.9%`** sustained (`arboris_hit_pct_x10=457..479`, asynchronously overlapped with `lm_head`, 0 critical-path latency) |

**Sustained Max-Concurrency HTTP Run (`B=8` Batched + 4-Row SMEM Kernels, `C = 1, 4, 8, 16, 32` — `74/74` HTTP 200, `0` errors, `nvrm_errors_delta: 0`, exact `The capital of France is **Paris**.` across every wave, total wall time `129.30 s` vs `184.13 s` unbatched baseline = `1.42×` faster, `92.4%` mean GPU util, `38.15 W` mean GPU power, `29.5 GiB` free RAM)** — exclusive quiet `spark-window` `20261125T043620Z-00f445` (`exit: 0`, binary `988 KB` SHA-256 `f83075f79aa12575437e69716d8d5b17718510230c472d2e3f17b9216f40f093`, [receipt JSON](evidence/gb10-omega-engine-qwen38-flash-next-b8-sustained-c32.bb525e98791130b73aad740cb7b579dbbae44070376522418eef0a39b2d4a0bc.json); earlier unbatched baseline [receipt JSON](evidence/gb10-omega-engine-qwen38-flash-next-sustained-c32.47b49d3e6faf60ccc343c1e9e1c5f103c630a361075712f71bf38ae21e1f6a1a.json)):

| Wave | Concurrency | Reqs OK / Total | Baseline (`B=1`) | **New (`B=8` + 4-Row SMEM)** | Speedup | Live Arboris Hit |
|---|---:|---:|---:|---:|---:|---:|
| `warmup-C1` | `1` | `2 / 2` | `16.38 tok/s` decode | **`19.18 tok/s` decode** | **`+17.1%`** | `41.6%` |
| `conc-C4` | `4` | `8 / 8` | `9.69 agg tok/s` | `7.62 agg tok/s` (`22.96 s`) | — | `46.5%` |
| `conc-C8` | `8` | `16 / 16` | `9.85 agg tok/s` | **`14.83 agg tok/s`** (`23.73 s`) | **`+50.6%`** | `44.2%` |
| `conc-C16` | `16` | `16 / 16` | `10.12 agg tok/s` | **`14.99 agg tok/s`** (`23.48 s`) | **`+48.1%`** | `44.1%` |
| `sustained-C32` | **`32`** | **`32 / 32`** | `10.29 agg tok/s` (`86.4 s`) | **`16.19 agg tok/s`** (`54.91 s`, `30.5 tok/s` decode batch) | **`+57.3%`** | `44.4%` |

#### B. Qwen 3.8 27B Dense — Generic Zero-CUDA Path

Measured on **NVIDIA DGX Spark (GB10, `sm_121`, driver `580.173.02`)** inside exclusive quiet `spark-window` `20261125T093124Z-743d25` (`exit: 0`, `nvrm_errors_delta: 0`) on `2026-11-25` — [receipt JSON](evidence/gb10-omega-engine-27b-sol.8a85e227ccf96e63852857fcfaddb3cae4a8b63224f6e550258dbd7f096a7834.json) (prior baseline: [receipt JSON](evidence/gb10-omega-engine-verify.78669f86b247f11f792f89de9cad471f5de52887f92925d58894805556241e13.json)):

| Metric | Measured Result (`omega-engine run /home/aienalpha/models/qwen3.8-27b-fp8-017b9c7a 64`) |
|---|---|
| Binary | `omega-engine` (`903 KB` static ELF, `ldd: not a dynamic executable`, SHA-256 `03d1067096a76cc6c7b79d632f5de8e95ef314a5c8742f14eacec91b66bad8e6`) |
| Dynamic Tensor Resolution | `1,251` tensors across `65` files resolved purely by canonical name (zero hardcoded offsets) |
| Live Output | `The capital of France is **Paris**.` |
| Decode Speed (unbatched `N=1` generic path) | **`11.94 tok/s`** (`83.71 ms/tok`, up from `11.80 tok/s`) |
| Kernel Driver Errors | **`nvrm_errors_delta: 0`** |

---

## Companion Model-Tuned Repositories (With Full Cross-Stack Benchmarks)

For peak hand-tuned throughput (L2-Swizzled Super-HMMA Tensor-Core chunked prefill, Split-K `S=16` batched FA2, Coalesced HMMA Batch-8, Speculative Verify / Native MTP, Live Arboris, and Full-Chip GB10 APU `4.4+ T ops/s`), see the dedicated model repositories:
- **[omega-qwen36-35b](https://github.com/drakestapleton/omega-qwen36-35b)** — Qwen 3.6 35B-A3B MoE (`67.47 tok/s` prose, `75.99 tok/s` MTP, `241.47 tok/s` Batch-8, `1,841 tok/s` prefill, vs. vLLM & SGLang).
- **[omega-qwen38-27b](https://github.com/drakestapleton/omega-qwen38-27b)** — Qwen 3.8 27B Dense (`17.27 tok/s` Native MTP `PARITY=EXACT`, `50.83 tok/s` Batch-8, `50.76 tok/s` Spec-8, `474 tok/s` 1K / `460 tok/s` 4K / `404 tok/s` 16K prefill, vs. TandemLLM & PyTorch).

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
