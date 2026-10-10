# llama-swap Docker image for AMD GPUs (ROCm + Vulkan)

The [llama-swap](https://github.com/mostlygeek/llama-swap) `unified-vulkan` image rebuilt from source for AMD hardware: the same binary names, config location and port, so a config written for upstream's image works here, but with **both GPU stacks**, everything at its **current upstream revision**, and the open PRs worth having merged in:

- **llama-swap itself built from source** — current `main` plus any open upstream PRs worth having (`LLAMA_SWAP_PATCHES`, empty at the moment — #1099, live per-turn generation stats in the Playground Chat, merged upstream 2026-09-07), web UI embedded, `vllm-wrapper` from the same tree. It is *the* `llama-swap` binary, not a side binary.
- **llama.cpp from current master plus open upstream PRs** (`LLAMA_PATCHES` plus rebased patches: Qwen 3.5-family delta-net fix, exact checkpoint restore for hybrid models, Vulkan MMVQ for speculative decoding on AMD, MTP carrier reset — see [Upstream PRs in the llama.cpp build](#upstream-prs-in-the-llamacpp-build)), the **same tree for the Vulkan and the ROCm build**. There is no un-patched llama.cpp in the image: `llama-server` is the patched build.
- **Vulkan** (Mesa RADV) — works on practically any AMD GPU, including RDNA1/2, iGPUs/APUs and anything ROCm doesn't cover. Built with a modern shader compiler (see [Why build the Vulkan binaries ourselves](#why-build-the-vulkan-binaries-ourselves)) and shipped with a **current Mesa/RADV** instead of Ubuntu 24.04's.
- **ROCm 7.14** (HIP) — the full ROCm userspace runtime (HIP, rocBLAS/hipBLAS, hipBLASLt, `rocminfo`) plus HIP builds of the engines, with flash-attention kernels for every KV-cache quant. ROCm comes from AMD's per-gfx `packages-multi-arch` repository (`ROCM_CHANNEL=multiarch`), so the image carries BLAS kernels only for the gfx targets it is built for instead of ~6 GB of all-arch Tensile blobs; the classic `repo.radeon.com` channel (tops out at ROCm 7.2.4, which still has the HIP-graphs bug fixed in 7.13) remains available as `ROCM_CHANNEL=classic`.
- **llama.cpp-rdna** (Vulkan, all tags) — [SelfRef/llama.cpp-rdna](https://github.com/SelfRef/llama.cpp-rdna) as a third `llama.cpp` install (`*-rdna` binaries): current upstream master **and** the ROCmFPx formats in one tree, plus measured RDNA3/RDNA3.5 Vulkan patches that were never upstreamed. See [The RDNA3 fork](#the-rdna-fork).
- **exllamav3 for RDNA3** (HIP, `:full` tag, gfx1100 only) — [phoenixhaxor/exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm) with [TabbyAPI](https://github.com/theroyallab/tabbyAPI) as `exl3-server`: EXL3 quants with DFlash2 or MTP speculative decoding. Not a llama.cpp. See [exllamav3 for RDNA3](#exllamav3-for-rdna3).

Both llama.cpp backends are built with runtime CPU dispatch (`GGML_CPU_ALL_VARIANTS`), so one image gets AVX2 on Zen 3 and AVX-512/VNNI/BF16 on Zen 4/5 for CPU-offloaded layers. Every engine is built once per backend from one resolved commit, so each ships as a matched Vulkan/ROCm pair — pick the backend per model in your llama-swap config.

## What's inside

| Engine | Vulkan (all tags) | ROCm (`:full`/`:latest` only) |
|---|---|---|
| [llama-swap](https://github.com/mostlygeek/llama-swap) `main` + PRs (`LLAMA_SWAP_PATCHES`) | `llama-swap` (UI embedded), `vllm-wrapper` | (backend-independent) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) master + PRs (`LLAMA_PATCHES`) | `llama-server`, `llama-cli`, `llama-tts`, `llama-bench` | `llama-server-rocm`, `llama-cli-rocm`, `llama-tts-rocm`, `llama-bench-rocm` |
| [llama.cpp-rdna](https://github.com/SelfRef/llama.cpp-rdna) (llama.cpp fork: upstream master + ROCmFPx + RDNA3 patches) | `llama-server-rdna`, `llama-cli-rdna`, `llama-bench-rdna`, `llama-quantize-rdna`, `llama-perplexity-rdna` | — (Vulkan-only) |
| [exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm) + [TabbyAPI](https://github.com/theroyallab/tabbyAPI) (PyTorch venv, `/opt/exl3`) | — (HIP-only) | `exl3-server` (gfx1100 only) |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) master | `whisper-server`, `whisper-cli` | `whisper-server-rocm`, `whisper-cli-rocm` |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) master | `sd-server` (web UI embedded), `sd-cli` | `sd-server-rocm` (web UI embedded), `sd-cli-rocm` |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) main | `audiocpp_server`, `audiocpp_cli`, `audiocpp_gguf` (GGUF converter, upstream ships none) | — (no HIP backend upstream) |
| `benchmark` (this repo, `scripts/benchmark`) | one CLI for server-level / `llama-bench` / standalone-variant benchmarks of the config's text entries, see [Benchmarking](#benchmarking) | (uses the `*-rocm` / `*-rdna` binaries via `--variant`) |
| `rerank-bench` (this repo, `scripts/rerank-bench`) | retrieval quality (nDCG@10) of the config's reranker entries over a cached embedding stage, see [Reranker quality](#reranker-quality-rerank-bench) | — |

**ROCmFPx / ROCmFP4 GGUFs (`Q4_0_ROCMFP4_FAST` and friends) load only in the `*-rdna` binaries.** Stock `llama-server` / `llama-server-rocm` exit with `tensor '…' has invalid ggml type 101`, so benchmark those files with `llama-bench-rdna` or `benchmark --standalone --variant rdna`, not the stock binaries (see [The RDNA fork](#the-rdna-fork)).

llama.cpp lives in self-contained directories `/opt/llama-vulkan`, `/opt/llama-rocm` and `/opt/llama-rdna` (binaries, `libllama`/`libggml*` and the per-CPU-level `libggml-cpu-*.so` variants, RPATH `$ORIGIN`) with symlinks in `/usr/local/bin`; whisper/sd/audio.cpp binaries are static. Exact versions of everything — every commit, every merged PR, the glslc used and the enabled build options — are recorded in `/versions.txt` inside the image.

## Runtime layout

Built from plain `ubuntu:24.04` (see [Why not build on the upstream image](#why-not-build-on-the-upstream-image)); nothing from upstream's `docker/unified` is vendored.

- Binaries in `/usr/local/bin`; config at `/etc/llama-swap/config/config.yaml` (mount the directory or the file); models in `/models` (the working directory); audio.cpp's spec catalog at `/usr/local/share/audiocpp/model_specs`; `/versions.txt`.
- Entrypoint is `llama-swap` itself with `-config /etc/llama-swap/config/config.yaml -listen 0.0.0.0:8080 -watch-config` as the default `CMD`. Any argument passed to the container replaces those defaults (`docker run <image> -version`, `docker run <image> -config /models/my.yaml -listen 0.0.0.0:8080`).
- Also present: `rocm-smi` (llama-swap's GPU monitor in its UI reads it; sysfs-based, works without the ROCm runtime), `ffmpeg` + libav\* (whisper-server input decoding), `curl`, `python3` + PyYAML (for `benchmark`), `uv`/`uvx` (also how `rerank-bench` gets numpy + pyarrow).
- Runs as root, like upstream's root variant. There is no `-rootless` tag.

## Why build the Vulkan binaries ourselves

Upstream builds the Vulkan engines on Ubuntu 24.04 with its stock `glslc` (shaderc 2023.8 / glslang 14). That compiler cannot compile the `GL_EXT_integer_dot_product` and `GL_EXT_bfloat16` shaders, and llama.cpp's CMake then *silently* drops those code paths: at startup the device line reads `int dot: 0 | bf16: 0` even though RADV advertises both extensions. The integer-dot path is llama.cpp's fast quantized path on GPUs **without** cooperative-matrix support — q8_1 MMQ for K-quants (upstream measured ~2x prompt processing on an RX 6800 XT), DP4A flash attention for q8_0/q4_0 KV caches, MMVQ decode. On coopmat GPUs (RDNA3 and newer) llama.cpp keeps its FP16 coopmat matmul for prompts, so on those the rebuild changes little by itself (RX 7900 XTX: prompt speed identical, decode +2%); it still matters for RDNA1/2, older cards and iGPUs, for any future upstream work that needs those extensions, and it removes a silent build-dependent difference between GPUs.

This image builds on the Ubuntu 24.04 ABI, but with `glslc`/`libshaderc1` taken from the Ubuntu 26.04 pocket (glslang 16; `GLSLC_SUITE`). The build fails if the feature tests do not pass, so the extensions cannot silently disappear. Verify at runtime: the `ggml_vulkan: 0 = ...` line must show `int dot: 1`.

The larger measured win on current GPUs is the driver: the final image takes `mesa-vulkan-drivers` from the kisak-mesa PPA (`MESA_PPA`, Mesa 26.1 vs Ubuntu's 25.2). RX 7900 XTX, identical binaries, Qwen3.8-27B Q4_K_XL: pp512 693 → 862 t/s (+24%), decode 37.7 → 37.9 t/s; the newer RADV also exposes `VK_VALVE_shader_mixed_float_dot_product` (`fp16: dot2`).

## Why not build on the upstream image

Until 2026-09-06 this image was `FROM ghcr.io/mostlygeek/llama-swap:unified-vulkan` and replaced the engines. Once llama-swap itself had to be built from source (its open PRs are UI changes, so they cannot be applied to the release binary), the base contributed one apt line, two audio.cpp binaries and three commit hashes — and cost real things: deleting the base's engine binaries in a child layer does not remove them from the image, so every pull carried ~650 MB of unreachable files (the base's llama.cpp, sd and ggml builds plus the stock Mesa shadowed by the PPA one); the daily base rebuild invalidated every final layer whether or not anything relevant changed; and the tags could lag upstream by days. Building from `ubuntu:24.04` removes all of that, and every project is now taken directly from its default branch.

## Get the image

Prebuilt by [GitHub Actions](.github/workflows/build.yml). Two tags from the same Dockerfile (`WITH_ROCM`):

```bash
docker pull ghcr.io/selfref/llama-swap-rdna:vulkan   # Vulkan only (~1.5 GB)
docker pull ghcr.io/selfref/llama-swap-rdna:full     # Vulkan + ROCm
docker pull ghcr.io/selfref/llama-swap-rdna:latest   # alias for :full
```

Both tags are rebuilt on every push and every 3 days by the scheduled run, each time from the then-current default branches of llama-swap, llama.cpp, whisper.cpp, stable-diffusion.cpp, audio.cpp, llama.cpp-rdna and exllamav3-rocm and the current heads of the merged PRs. The ROCm stages are fat multi-gfx HIP builds that take hours of runner time, so on a PR or an ad-hoc run they only happen if you ask for them (Actions → Build image → Run workflow → tick **rocm**). The `*-rocm` binaries, `exl3-server` (see [exllamav3 for RDNA3](#exllamav3-for-rdna3)), the ROCm runtime and `rocminfo` exist only in `:full`/`:latest`.

Or build locally (expect a couple of hours for the fat HIP builds; `WITH_ROCM=false` for the Vulkan-only image in well under an hour):

```bash
docker buildx build -t llama-swap-rdna .
# pin every branch/PR to today's commit first (what CI does) -- see "How versions are pinned"
docker buildx build $(scripts/resolve-refs.sh --docker) -t llama-swap-rdna .
```

Build args:

- `LLAMA_SWAP_COMMIT` — default `main`  
  llama-swap revision to build `llama-swap` and `vllm-wrapper` from (sha, tag such as `v255`, or branch).
- `LLAMA_SWAP_PATCHES` — default *(empty)*  
  Space-separated open upstream llama-swap PR numbers merged on top; same drift rules as `LLAMA_PATCHES`. The version string gets a `+prN` suffix per merged PR, e.g. `v255-2-g1a2b3c+pr1099` (that PR merged upstream 2026-09-07 and was retired).
- `WITH_ROCM` — default `true`  
  `false` builds the Vulkan-only image (no HIP stages, no ROCm runtime); CI uses this for `:vulkan`
- `ROCM_CHANNEL` — default `multiarch`  
  ROCm source: `multiarch` = repo.amd.com per-gfx packages (current releases, small runtime — kernels only for `AMDGPU_TARGETS`); `classic` = repo.radeon.com apt + `rocm/dev-ubuntu-24.04` builder (max 7.2.4, all-arch kernels, HIP-graphs bug — pair it with `GGML_CUDA_DISABLE_GRAPHS=1` at runtime)
- `ROCM_SERIES` — default `7.14`  
  multiarch channel: release series in the package names (`amdrocm-runtime7.14`, ...); apt resolves the newest point release of the series
- `ROCM_VERSION` — default `7.2.4`  
  classic channel: builder image tag and apt repo path
- `AMDGPU_TARGETS` — default `gfx1100;gfx1101;gfx1151;gfx1200;gfx1201`  
  gfx architectures compiled into the HIP binaries: RX 7900, RX 7800/7700, Strix Halo and RDNA4. Everything else (RDNA2, RX 7600, Strix Point, CDNA) is one entry away — see the table in [Choosing ROCm vs Vulkan](#choosing-rocm-vs-vulkan). Trim to just your GPU for a much faster build.
- `LLAMA_COMMIT` — default `master`  
  llama.cpp revision for both the Vulkan and the ROCm build (sha, tag, branch, or `refs/pull/N/head`).
- `LLAMA_PATCHES` — default 2 PRs, see below  
  Space-separated upstream llama.cpp PR numbers merged on top of `LLAMA_COMMIT`, for both backends, fetched over git as `refs/pull/N/head`. A PR that is closed on GitHub is skipped with a notice; one that no longer merges cleanly fails the build — never a silent no-op. `patches/*.patch` (local rebased patches) apply on top. See [Upstream PRs in the llama.cpp build](#upstream-prs-in-the-llamacpp-build).
- `WHISPER_COMMIT` / `SD_COMMIT` / `AUDIOCPP_COMMIT` — default `master` / `master` / `main`  
  Revision of the other engines (sha, tag or branch).
- `GLSLC_SUITE` — default `resolute`  
  Ubuntu release whose `glslc`/`libshaderc1` are used by the Vulkan builder (only those two packages; everything else stays 24.04)
- `LLAMA_FA_ALL_QUANTS` — default `ON`  
  ROCm llama.cpp: compile flash-attention kernels for all K/V cache quant combinations (without it only q8_0/q8_0 and q4_0/q4_0 stay on the GPU, see llama.cpp #27761). Set `OFF` for a faster build.
- `WITH_RDNA` — default `true`  
  Build [llama.cpp-rdna](https://github.com/SelfRef/llama.cpp-rdna) as `*-rdna` binaries. Vulkan-only, in both tags, independent of `WITH_ROCM`. `false` skips the stage.
- `RDNA_REPO` / `RDNA_BRANCH` / `RDNA_COMMIT` — default SelfRef's repo, `rdna`, *(empty)*  
  Fork source. **Always pin `RDNA_COMMIT` to a full 40-character sha** (`scripts/resolve-refs.sh` does): the clone's cache key is the arg values, so passing only the branch silently reuses the binary from whatever the branch pointed at last time, and GitHub rejects an abbreviated sha.
- `WITH_EXL3` — default `true`  
  Build [exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm) + TabbyAPI as `exl3-server`. Only takes effect together with `WITH_ROCM=true`: it is a HIP engine and adds ~8 GB of PyTorch, so `:vulkan` never contains it. `false` skips the stage.
- `EXL3_REPO` / `EXL3_BRANCH` / `EXL3_COMMIT` — default phoenixhaxor's repo, `main`, *(empty)*  
  Fork source; pin `EXL3_COMMIT` to a full sha like the other forks (`scripts/resolve-refs.sh` does). The TabbyAPI revision is the one the fork's own installer pins.
- `MESA_PPA` — default `ppa:kisak/kisak-mesa`  
  Newer Mesa/RADV for the final image; `""` keeps Ubuntu 24.04's stock Mesa 25.2
- `QWEN_TEMPLATE_URL` — default froggeric's `chat_template.jinja`  
  Source of the fixed Qwen chat template shipped at `/etc/llama-swap/templates/qwen-fixed.jinja` (see below)
- `QWEN_SHARP_TEMPLATE_URL` — default peculiar-ragdoll's `chat_template.jinja`  
  Source of the Sharp variant shipped at `/etc/llama-swap/templates/qwen-sharp.jinja` (see below)
- `LLAMA_PATCHES_HEADS` / `LLAMA_SWAP_PATCHES_HEADS` / `BUILD_DATE` — default *(empty)*  
  Cache keys only, set by CI from `scripts/resolve-refs.sh` (see below).

### How versions are pinned

Nothing in the build context changes between two scheduled runs, and "master" in a build arg is the same string today and tomorrow, so on its own BuildKit would serve every cached stage forever. `scripts/resolve-refs.sh` therefore resolves every branch, tag and PR head to a commit *once* per run (a 4-second `git ls-remote` per repository) and CI passes the results as build args: a moved branch or an updated PR is a cache miss for exactly the stages that use it, while the ROCm stages (hours of compile) stay cached as long as nothing they build from moved. It is also what guarantees that the Vulkan and ROCm builds of an engine, compiled in different jobs hours apart, are the same revision. `BUILD_DATE` does the same for the final stage's apt layers (Ubuntu updates, the PPA Mesa, ROCm runtime point releases): they are redone on every run, minutes not hours.

A local `docker buildx build .` resolves per stage at build time, which is fine on one machine — but a *second* local build gets the cached stages, not new commits. Prefix it with `$(scripts/resolve-refs.sh --docker)` to build today's revisions, or pass a `*_COMMIT` build arg to pin one.

## Bundled chat templates

Two fixed Qwen 3.5/3.6/3.8 chat templates ship under `/etc/llama-swap/templates/`, both fetched at build time and refreshed on every rebuild (versions are in `/versions.txt` as `qwen_chat_template:` / `qwen_sharp_chat_template:`):

- `qwen-fixed.jinja` — [froggeric's Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates): sane reasoning-depth default, working `enable_thinking=false`, history `<think>` extraction, robust tool-call arguments — see its model card.
- `qwen-sharp.jinja` — [peculiar-ragdoll's Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates): froggeric's template with a force-appended terseness system prompt (fewer filler/thinking tokens, same kwargs as above). Pass `{"terse": false}` in `chat_template_kwargs` to drop the appended prompt for a request.

Use one per model:

```yaml
cmd: >
  llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --port ${PORT}
    --jinja --chat-template-file /etc/llama-swap/templates/qwen-fixed.jinja
```

## exllamav3 for RDNA3

The `:full` tag ships [phoenixhaxor/exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm) — [exllamav3](https://github.com/turboderp-org/exllamav3) with its EXL3 matmul, decode/verify attention and Gated DeltaNet kernels rewritten for RDNA3 (wave32, WMMA) — served by the fork's pinned, patched [TabbyAPI](https://github.com/theroyallab/tabbyAPI): OpenAI-compatible API with streaming, tool calls, images and reasoning, plus **DFlash2** (block-diffusion draft model) or **MTP** speculative decoding. It is a PyTorch (ROCm 7.2 wheels) venv under `/opt/exl3`, launched as `exl3-server`; PyTorch brings its own HIP libraries, and the image keeps only their gfx1100 kernels. Triton JIT-compiles parts of prefill at runtime, which is why `:full` also carries `gcc` and the Python headers.

**gfx1100 (RX 7900 XTX / 7900 XT / W7900) only.** On anything else use the llama.cpp builds.

Every TabbyAPI config key is also a flag (`exl3-server --help`), so a llama-swap entry needs no config file. The image's Qwen templates are available as `--prompt-template qwen-fixed` / `qwen-sharp`. Example for Qwen3.8-27B ([EXL3 3.5 bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw) + [DFlash2 draft](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw), both downloaded into one directory):

```yaml
qwen38-exl3:
  env: ["HIP_VISIBLE_DEVICES=0"]   # the gfx1100 card; ROCm indices can differ from Vulkan ones, check rocminfo
  checkEndpoint: /health
  cmd: >
    exl3-server --host 127.0.0.1 --port ${PORT} --disable-auth true
      --model-dir /models/exl3 --model-name Qwen3.8-27B-EXL3-3.5bpw
      --max-seq-len 196608 --cache-size 196608 --cache-mode Q8
      --chunk-size 2048 --max-batch-size 1
      --vision true --vision-offload true
      --reasoning true --reasoning-start-token "<think>" --reasoning-end-token "</think>"
      --tool-format qwen3_coder --prompt-template qwen-fixed
      --draft-mode model --draft-model-dir /models/exl3
      --draft-model-name Qwen3.8-27B-DFlash2-EXL3-5.0bpw --draft-cache-mode Q4
      --sysmem-recurrent-cache 4096
```

That is the fork's `config.dflash2-192k.yml` as flags (~23.6 GB VRAM). For the full 262,144 context use `--draft-mode mtp --draft-cache-mode Q8` (the MTP head is inside the main model) and drop the draft model flags. The container needs `/dev/kfd` as well as the render node.

Measured on an RX 7900 XTX against this image's best llama.cpp path for the same model (`llama-server-rdna`, ROCmFP4-FAST + MTP n4), same chat template (`qwen-sharp`), greedy, single stream, client-side timing (t/s, exl3 vs llama.cpp):

| | DFlash2, 192k | MTP, 256k | llama.cpp |
|---|---|---|---|
| prose decode | **77.5** | 66.7 | 67.7 |
| JSON decode | **174.8** | 127.1 | 117.5 |
| code-edit decode | **221.7** | 140.2 | 132.7 |
| reasoning decode | **153.3** | 110.1 | 110.0 |
| decode after an 8k prompt | 79.0 | 69.9 | **80.0** |
| prefill 8k / 30k | 1233 / 1147 | **1243 / 1166** | 947 / 845 |

DFlash2 is the mode to use; MTP only buys the longer context. Both use ~24 GB of the card. The built commits are recorded in `/versions.txt` as `exl3_commit:` / `exl3_tabbyapi_commit:`.

## The RDNA fork

Both tags ship [SelfRef/llama.cpp-rdna](https://github.com/SelfRef/llama.cpp-rdna) (branch `rdna`) as `llama-server-rdna` / `llama-cli-rdna` / `llama-bench-rdna` / `llama-quantize-rdna` / `llama-perplexity-rdna`. Unlike `llama-server` it cannot be replaced by upstream plus a few PRs, because it adds **new GGUF tensor types**. Its base is [LaurentZuijdwijk's ROCmFPx fork](https://github.com/LaurentZuijdwijk/llama.cpp) (shipped here as `*-fpx` until 2026-10-07), which brings:

- **ROCmFPx weight formats** — `Q4_0_ROCMFP4` and its `_FAST` / `_LEAN` / `_COHERENT` / `_STRIX` / `_STRIX_LEAN` recipes plus `Q2/Q3/Q6/Q8_0_ROCMFPX` (ggml type ids 100–107), hand-ported from [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX) ← [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX), where the format originates. A 4-bit codebook with UE4M3 block scales at 4.25–4.6 bpw. Stock `llama-server` **cannot load these files at all** (`Q4_0_ROCMFP4_FAST` is GGUF file type 103), which is the whole reason this build exists. It is a *software* codebook, not hardware FP4 — no RDNA GPU has FP4 matrix instructions.
- **Kernels for the batch widths speculation runs at** — whole-block MMVQ for the FP4 types, branch-free fp6/fp3 dequant, an IQ3_S register-spill fix at `NUM_COLS > 4`. Verifying an MTP draft is a batch-3..8 matmul, and the stock shaders are mistuned exactly there.
- **Adaptive draft sizing** — `--spec-draft-adaptive` with `--spec-draft-n-min`: the draft length follows the measured acceptance rate instead of a fixed `--spec-draft-n-max`.
- **Vulkan prefill tuning that pays on stock K-quants too** — an LDS bank-conflict fix in the coopmat matmul tile stride (driver-gated to RADV ≥ 25.3), an f16 B operand for quantized matmul/matmul_id, a tiled concat-transpose for the delta-net conv state.

On top of that base the branch has **upstream master merged in** (re-ported 2026-09-18, kept current since) — the ROCmFPx fork itself last merged upstream on 2026-08-30 and cannot take `LLAMA_PATCHES` — plus a short list of RDNA3/RDNA3.5 Vulkan patches that circulate between community forks without an upstream PR: coopmat1 flash-attention work and on-device speculative checkpoints, each carried only after a benchmark showed it paying.

Measured here on an **RX 7900 XTX** (gfx1100, RADV/Mesa 26.2.2, 2026-09-09), Qwen3.8-27B with its baked MTP head, greedy decode t/s on prose / structured JSON / a copy-heavy refactor:

| build + weights | decode | prefill | VRAM | wikitext PPL |
|---|---|---|---|---|
| `llama-server`, unsloth UD-Q4_K_XL (16.35 GiB), `--spec-draft-n-max 3` | 60.9 / 83.6 / 94.7 | 423 / 328 / 777 | 22.3 G | 6.637 |
| ROCmFPx fork (the former `llama-server-fpx`), [julianmb ROCmFP4-FAST](https://huggingface.co/julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF) (13.55 GiB), `--spec-draft-n-max 4` | **76.7 / 105.5 / 128.0** | 457 / 343 / 921 | **19.0 G** | 6.921 |

So **+26 / +26 / +35 % decode and −3.3 GiB** for **+4.3 % perplexity** — the FP4 file is the lossiest ROCmFP4 recipe (single scale per 32 weights); `Q4_0_ROCMFP4_STRIX_LEAN` trades ~0.1 GiB for less of that loss. On the *same* K-quant file the fork's engine work alone is +5–7 % prefill and neutral decode, so most of the win is the weight format, and both parts are needed. Draft depth matters more than usual: `n-max 4` beat 3 by +9/+11/+15 % and 5–6 by ~11 % on prose (acceptance falls from 47 % to 34 %); `--spec-draft-adaptive` only won on the 100 %-acceptance copy task.

`llama-quantize-rdna` produces these files from a BF16/F16 source (`llama-quantize-rdna in.gguf out.gguf Q4_0_ROCMFP4_STRIX_LEAN`), and `llama-perplexity-rdna` scores them — no other binary in the image knows the types.

Targets are gfx1100 (RX 7900 XTX), gfx1101 (RX 7800 XT) and gfx1151 (Strix Halo); it is Vulkan-only. The build fails if a merge ever drops the ROCmFPx types or adaptive drafting, so the `-rdna` name never silently becomes a plain llama.cpp; the built commit and type list are in `/versions.txt` as `llama_rdna_commit:` / `llama_rdna_types:`. What the branch carries, what was tried and dropped, and the measurements are in the fork's own README.

## Upstream PRs in the llama.cpp build

`LLAMA_PATCHES` fetches each PR's branch over git (`refs/pull/<N>/head`, blobless) and merges it into `LLAMA_COMMIT` before building — the same tree for Vulkan and ROCm (`scripts/checkout-with-prs.sh`). The `.patch` HTTP endpoint is deliberately not used — GitHub rate-limits it (HTTP 429) from shared CI runner IPs. GitHub keeps `refs/pull/<N>/merge` only while a PR is open, so a closed PR (merged or rejected) is detected and skipped with a notice — drop it from the list; a PR that no longer merges cleanly fails the build (it drifted; re-check it). Merged PRs and the base llama.cpp commit are listed in `/versions.txt` (`llama_patches:`, `llama.cpp:`). Dry-run a changed set before committing it:

```bash
scripts/checkout-with-prs.sh https://github.com/ggml-org/llama.cpp.git master /tmp/llama.cpp 28457 28243 ...
```

Until 2026-09-06 there were two Vulkan builds — a "pure" `llama-server` with only #27952 and a `llama-server-next` with the full set. The full set had become the one actually serving models, and none of the PRs touches CUDA/HIP sources, so they were merged into one build that the ROCm side now shares.

The set was last revised on **2026-10-05** against master `e117148a4`. Since the 09-09 trim most of the list merged upstream or was superseded — `#27952` (int8 coopmat1 MMQ, the one with the largest measured RDNA3 gain) merged on 2026-09-24, `#28243` gave way to `#29761` (Qwen4Exp MTP), `#28213` to master's own sparse attention — and two PRs (`#28699`, `#27210`) stopped merging and were dropped. The Dockerfile comment above `ARG LLAMA_PATCHES` records every decision with its date. The current set:

| PR | area | what | how | status upstream |
|---|---|---|---|---|
| [#28265](https://github.com/ggml-org/llama.cpp/pull/28265) | models | Qwen3.5-family delta-net out-proj 2D (+6-9% batched TG on Strix Halo) | `LLAMA_PATCHES` | open |
| [#25592](https://github.com/ggml-org/llama.cpp/pull/25592) | server | exact-position checkpoint restore (hybrid/recurrent) | `LLAMA_PATCHES` | open |
| [#25666](https://github.com/ggml-org/llama.cpp/pull/25666) | Vulkan | no MMVQ for speculative-decode steps on AMD | `patches/25666-rebased.patch` | open, conflicts with master |
| [#28333](https://github.com/ggml-org/llama.cpp/pull/28333) | spec | zero the MTP carrier at sequence start (deterministic MTP output) | `patches/28333-rebased.patch` | open |

Why each patch is carried as a file rather than a PR number is in [patches/README.md](patches/README.md).

Measured on an RX 7900 XTX with this image (Mesa 26.1, llama-bench, q8_0/q4_0 KV, ub 512; dense = Qwen3.8-27B UD-Q4_K_XL, MoE = Qwen3.6-35B-A3B UD-Q4_K_M):

| PR | what | dense pp512 | MoE pp512 | decode | verdict |
|---|---|---|---|---|---|
| [#27952](https://github.com/ggml-org/llama.cpp/pull/27952) | Vulkan int8 coopmat1 matmul for RDNA3/4 | 858 → **898** (+4.6%) | 3127 → **3704** (+18.5%) | unchanged | **merged upstream 2026-09-24**, in every build since |
| [#25483](https://github.com/ggml-org/llama.cpp/pull/25483) | Vulkan: skip unneeded MoE work in coopmat1 mul_mm | +0.3% | +0.2% | unchanged | not worth a patch |
| [#26284](https://github.com/ggml-org/llama.cpp/pull/26284) + [#26301](https://github.com/ggml-org/llama.cpp/pull/26301) | HIP: RDNA3 MMQ tuning + dequant-float matvec | ROCm 977 → 1000 (+2%, within noise) | +2% | unchanged | not adopted (#26284 still carries RDNA4 changes its reviewer wants removed; re-test when merged) |
| [#22970](https://github.com/ggml-org/llama.cpp/pull/22970) | Vulkan K-quant A-matrix transpose | – | – | – | stale, conflicts with master |
| [#28507](https://github.com/ggml-org/llama.cpp/pull/28507) + [#28489](https://github.com/ggml-org/llama.cpp/pull/28489) | Vulkan FA shmem staging (RDNA scalar path) + batch-independent MMVQ | 908.7 → 906.5 | 4018 → 4029 | kernel unchanged; server MoE+MTP decode +2-5 % | #28489 was in the set, **dropped** when it stopped merging (not worth a rebase for +2-5 % decode); #28507 not adopted (neutral) |

The same mechanism serves llama-swap: `LLAMA_SWAP_PATCHES` is merged into `LLAMA_SWAP_COMMIT` and the result is compiled with the UI embedded, exactly as upstream's release build does (`make linux-amd64`). The list is empty right now — its one entry, [#1099](https://github.com/mostlygeek/llama-swap/pull/1099) (live generation stats in the Playground Chat), merged upstream on 2026-09-07. Retire PRs from either list as they merge — the build says so.

## Run

```bash
docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  -p 8080:8080 \
  -v "$PWD/models:/models" \
  -v "$PWD/config:/etc/llama-swap/config" \
  ghcr.io/selfref/llama-swap-rdna:latest
```

Or `docker compose up` — see [compose.yml](compose.yml). The llama-swap UI is at http://localhost:8080. The host only needs the `amdgpu` kernel driver (ROCm userspace lives in the image); Vulkan-only use works without `/dev/kfd`.

Edit [config/config.yaml](config/config.yaml) to define your models — it shows the pattern: the same engine as `*-rocm` (ROCm) or plain (Vulkan), chosen per model. The container watches the config and reloads on change. To use a config elsewhere, pass the flags as the container command (they replace the defaults, see [Runtime layout](#runtime-layout)): `... ghcr.io/selfref/llama-swap-rdna:latest -config /models/my.yaml -listen 0.0.0.0:8080 -watch-config`.

## Choosing ROCm vs Vulkan

The Vulkan binaries run on any AMD GPU with a current RADV driver. The HIP (ROCm) binaries contain GPU code only for the architectures in `AMDGPU_TARGETS`: every target multiplies their compile time, so the published `:full` image builds just these five:

| GPU | gfx | ROCm in the published `:full` | Vulkan |
|---|---|---|---|
| RX 7900 XTX / XT / GRE, W7900 / W7800 | gfx1100 | **yes** (also the only target of `exl3-server`) | yes |
| RX 7800 XT / 7700 XT | gfx1101 | **yes** | yes |
| Strix Halo — Ryzen AI Max (Radeon 8060S / 8050S) | gfx1151 | **yes** | yes |
| RX 9070 XT / 9070 | gfx1201 | **yes** | yes |
| RX 9060 XT | gfx1200 | **yes** | yes |
| RX 7600 / 7600 XT | gfx1102 | no — add `gfx1102`, or try `HSA_OVERRIDE_GFX_VERSION=11.0.0` (runs the gfx1100 code) | yes |
| Strix Point — Ryzen AI 300 (Radeon 890M / 880M) | gfx1150 | no — add `gfx1150` | yes |
| RX 6800 / 6900 (RDNA2) | gfx1030 | no — add `gfx1030` | yes |
| other RDNA2 (RX 6700 / 6600 …) | gfx1031–1036 | no — add `gfx1030` and run with `HSA_OVERRIDE_GFX_VERSION=10.3.0` | yes |
| Instinct MI100–MI350 | gfx908 / 90a / 942 / 950 | no — add the target; AMD's own ROCm containers are the usual choice | — |
| RDNA1, Vega, older iGPUs | — | no (outside current ROCm) | yes |

"Add" means building the image yourself with the target appended, e.g. `--build-arg AMDGPU_TARGETS="gfx1100;gfx1101;gfx1151;gfx1200;gfx1201;gfx1030"` (build args are listed under [Get the image](#get-the-image)); the `:full` runtime then also installs that target's ROCm math-library kernels. The override variables make the runtime treat an unlisted chip as a listed one and are not tested here.

On a card that has both, prefer ROCm for dense models and compare: on the RX 7900 XTX the Vulkan build of the same llama.cpp commit is usually as fast or faster at decode, and the `*-rdna` binaries (ROCmFPx formats) are Vulkan-only.

GPU selection on multi-GPU hosts: `HIP_VISIBLE_DEVICES=0` for `*-rocm` binaries, `GGML_VK_VISIBLE_DEVICES=0` for Vulkan ones.

## Verifying the stacks inside the container

```bash
# ROCm sees the GPU (needs /dev/kfd + /dev/dri)
docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
  --entrypoint rocminfo ghcr.io/selfref/llama-swap-rdna:latest

# Each backend's device list
docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
  --entrypoint llama-server-rocm ghcr.io/selfref/llama-swap-rdna:latest --list-devices
docker run --rm --device /dev/dri \
  --entrypoint llama-server ghcr.io/selfref/llama-swap-rdna:latest --list-devices

# Vulkan feature line -- must say "int dot: 1" (see "Why build the Vulkan binaries ourselves")
docker run --rm --device /dev/dri --entrypoint llama-bench \
  ghcr.io/selfref/llama-swap-rdna:latest -m /dev/null 2>&1 | grep "ggml_vulkan: 0"

# Versions baked into the image (every commit, merged PR and build option)
docker run --rm --entrypoint cat ghcr.io/selfref/llama-swap-rdna:latest /versions.txt

# llama-swap version (e.g. v255-2-g1a2b3c, plus a +prN suffix per merged PR)
docker run --rm ghcr.io/selfref/llama-swap-rdna:latest -version
```

## Benchmarking

`--ppl` is the odd mode out: it measures **quality, not speed**, so its rows do not belong in a
speed log. It runs `llama-perplexity` over the wikitext-2 raw test split — fetched once from the
same URL `llama.cpp`'s own `scripts/get-wikitext-2.sh` uses and cached next to the model hub, so
the numbers are comparable with those quoted upstream (`--ppl-file` overrides, and is the fallback
on a host that cannot reach the URL). The KV cache is deliberately left at f16 rather than taking
the entry's `--cache-type-k/v`: quantised KV would fold a *serving* choice into a measurement of
the *weights*.

It is the cheap, sensitive screen for a quant or requant — minutes, and a continuous number,
where a task benchmark needs hundreds of samples to resolve the same difference. It does not say
what the drift *costs*: a measured +4.3 % PPL on one model landed entirely on maths and code and
not at all on instruction following, which only a task benchmark could show. Use it to **reject**
candidates cheaply, not to accept them. `--kld-base` adds KL-divergence against a reference model,
which is closer to what a quant recipe is tuned against; note the base file is large (hundreds of
MB even at a couple of chunks) and must be produced from the full-precision file.

`benchmark` (in `/usr/local/bin`, source `scripts/benchmark`) measures the llama.cpp text entries of
the mounted llama-swap `config.yaml` and prints one table: model, the parameters that matter
(quant, context, KV types, speculative type / draft length, placement, batch sizes) and the numbers.

```sh
docker compose exec llama-swap benchmark my-model                # production sampling + filters (default): what users get
docker compose exec llama-swap benchmark --greedy my-model my-model-q8   # deterministic + hash, for A/Bs
docker compose exec llama-swap benchmark --all                   # every listed text entry (hours -- see below)
docker compose exec llama-swap benchmark my-model:l              # a reasoning alias, with its filters
docker compose exec llama-swap benchmark --prompt text my-model   # all text presets (fast is the default)
docker compose exec llama-swap benchmark --prompt prose,prefill --prefill-tokens 65536 my-model
docker compose exec llama-swap benchmark --kernel --std --unload my-model    # llama-bench, community-comparable line
docker compose exec llama-swap benchmark --standalone --variant rocm --unload my-model # this entry on the ROCm build
docker compose exec llama-swap benchmark --ppl --unload <model> <other-quant>   # perplexity over wikitext-2
docker compose exec llama-swap benchmark --ppl --kld-base ref.dat --unload <model>   # KL-divergence vs a reference
docker compose exec llama-swap benchmark --prompt vision --vision-answers my-vlm other-vlm   # vision: accuracy + speed
docker compose exec llama-swap benchmark --list                  # parsed entries, no requests
```

| Mode | What runs | What it tells you |
|---|---|---|
| sampled (**default** since 2026-09-15; `--sampled`) | llama-swap's `/v1/chat/completions` with the entry's filters (production sampling, aliases like `:l` allowed) | what users actually get; hash blank, not comparable with greedy rows |
| `--greedy` | greedy requests through `/upstream/<model>/` (real entry flags: MTP, mmproj, `--parallel`, KV types; llama-swap filters bypassed, aliases resolve to the parent) | deterministic throughput + output hash for build/config A/Bs. **Overstates real-world decode** — draft acceptance falls at production temperature (Qwen3.8-27B + MTP, 2026-09-12: prose +26 %, tools +7 %, json +2 %) |
| `--kernel` | `llama-bench[-<variant>]` on the entry's cached GGUF with the entry's KV/batch/placement flags, or `--std` for `-fa 1 -ctk q8_0 -ctv q4_0 -b 2048 -ub 512 -p 512 -n 128 -d 0,8192` | hardware / driver / build (no speculative decoding, no mmproj) |
| `--standalone --variant V` | spawns `llama-server-<V>` with the entry's exact cmd (host/port substituted), benches it like `--greedy`, kills it | binary A/Bs without a temporary config entry |

**A model name is required** since 2026-09-12 — pass `--all` for every listed text entry. The full sweep is every preset x every entry, and on a multi-GPU box the `vram` entries swap one at a time, so it is a ~20 GB load per entry plus a 32k-prompt pass (`prefill`, `depth`): hours of work that used to start by typing `benchmark` with no arguments. Model names may be entry keys or their aliases / `:l:m:x` tiers.

Variants are `vulkan` (plain `llama-server`), `rocm` and `rdna`. Presets (`--prompt`, **default group `fast`** = prose, json, tools, the ~1 min smoke test; `text` = all of them): `prose` (free text, speculative worst case), `json` (structured
output, best case), `refactor` (copy-heavy code edit, ~2k-token module), `agent` (tool-calling
transcript, 6 tools, ~2.5k prompt tokens; column `call`), `tools` (single tool call, natural stop;
`call`/`finish`), `reasoning` (`enable_thinking` on; `think`), `prefill` (pure big-context prefill,
`--prefill-tokens`, default 32768 capped to the entry's context; `tok` is the real prompt length,
`ttft` the prompt time), `depth` (decode speed for 256 tokens after a `--depth-tokens` prompt, default 32768 --
where attention variants and speculative acceptance differ most; column `depth`). `--format md` prints rows for a markdown log, `--format json` for diffing.

### Vision presets

Two presets measure the multimodal path, and they are the only ones that grade whether the answer
is *right* rather than just how fast it arrived. They are **not** in the default `text` group (most entries
take no image); ask for them by name, as the group `vision`, or take everything with `all`.
The group `fast` (`prose`, `json`, `tools`) is the smoke test: no 32k prompts and no 1k-token
generation, about a minute per GPU entry, still showing both speculative extremes and a tool call. An entry
without `image` in `capabilities.in` skips them with a warning.

| Preset | Request | Columns |
|---|---|---|
| `vision-ocr` | one image, "transcribe this receipt", natural stop (cap 512 tokens) | `acc` = character F1 against the printed text (recall punishes misreads, precision punishes preamble/hallucination) |
| `vision-vqa` | eight short questions over four images: receipt total, a line-item quantity, counting shapes, a shape's colour, one big glyph, a bar chart's tallest bar, a bar's value, a cell in a 4x4 digit grid | `acc` = graded answers, `miss` = which cases failed |

Both also report `ptok`, the median prompt tokens the model spent on the same image — a
dynamic-resolution ViT with `--image-min-tokens 2048` and a NaFlex encoder differ by several
thousand tokens on identical input, which is most of the explanation for their `pp t/s`.

The four test images (a receipt, a shapes scene, a bar chart, a digit grid) are **rendered by the
script itself** from tables in its source — a hand-rolled PNG encoder plus a 5x7 bitmap font, pure
stdlib, no PIL in the image and no binary assets in this repo — so they are bit-identical on every
box and every build. `--vision-dump DIR` writes them out to look at, together with each case's
question and ground truth. `--vision-answers` prints what each model actually replied, per case,
which is the part worth reading when a score drops.

Accuracy rows are comparable only *within one version of this file*: editing an image, a question
or a grader invalidates every earlier `acc` number, so re-run the sweep rather than comparing
across a change.

Notes: the tool holds the box while it runs -- a watcher thread polls llama-swap's `/running` and
immediately unloads any model that is not the one under test (other services' requests fail for
that moment; rows that saw an intruder are flagged `EVICTED:<n>`; `--no-guard` disables it and a
stranger then only gets `CONTAMINATED`). Concurrent requests to the *same* model cannot be blocked
and are detected through llama-server's `/metrics` counters (`SHARED`). `--kernel` and
`--standalone` refuse to start while models are resident unless `--unload` is given. `--kernel` passes the entry's
`--load-mode` to llama-bench (`-lm`); this matters on Vulkan, where llama.cpp's default is mmap *off* and every
CPU-side tensor would be copied into pinned memory instead of page cache (77 GiB GTT measured on Strix Halo). `--fit on` entries map to `-fitt 1024` in kernel mode (llama-server's margin), `-ngl all`
to llama-bench's default. `gtt` growth above `--spill-threshold` (default 1.5 GiB) on a
VRAM-resident entry flags `SPILL`; on UMA GPUs (Strix Halo) the check is off. The `hash` is the
sha256 of the first measured output and is reproducible for the same request sequence across
loads and builds. Needs `LLAMA_SWAP_API_KEY` in the environment when llama-swap has `apiKeys`.
The `-rocm` variant exists only in the `:full`/`:latest` tag; `-rdna` is in both tags.

### Reranker quality (`rerank-bench`)

`rerank-bench` (in `/usr/local/bin`, source `scripts/rerank-bench`) measures what `benchmark` cannot:
how much a **reranker** entry improves retrieval. It is a quality tool, nDCG@10, against the served
entries through llama-swap:

```sh
docker compose exec llama-swap rerank-bench --embed embeddinggemma \
    --query-prefix 'task: search result | query: ' \
    --doc-template 'title: {title} | text: {text}' --empty-title none \
    qwen3-reranker jina-reranker-v2              # model ids from your config.yaml
docker compose exec llama-swap rerank-bench --embed embeddinggemma   # stage-1 baseline only
```

Stage 1 is the `--embed` model retrieving the top 25 per query over the whole corpus; it is
computed once per embedder / prompt / task set and cached, so adding a reranker later costs only
that reranker's pass. Every reranker then re-orders the **same** candidates, so arms differ only in
the reranker, and the `stage 1 only` row is the baseline to beat — with a strong embedder, a
reranker that makes things worse is common. Default tasks are the MTEB NFCorpus and SciFact test
splits in English and Polish (same corpora, two languages, so an EN/PL gap is multilingual quality);
`--task <hf dataset repo>` takes any MTEB-layout retrieval set. Datasets (~25 MB) are fetched once
and cached next to the model hub; numpy + pyarrow come through `uv` on first use.

Example, RX 7800 XT, EmbeddingGemma-300M with both prompts as stage 1 (wall time for ~31k pairs):

| reranker | NFCorpus-PL | SciFact-PL | NFCorpus | SciFact | mean | time |
|---|---:|---:|---:|---:|---:|---:|
| stage 1 only | 0.3008 | 0.6882 | 0.3889 | 0.7911 | 0.5423 | — |
| bge-reranker-v2-m3 Q8_0 | 0.3191 | 0.6994 | 0.3619 | 0.7475 | 0.5320 | 812 s |
| jina-reranker-v2-base-multilingual Q8_0 | 0.3149 | 0.7226 | 0.3839 | 0.7765 | 0.5495 | 332 s |
| Qwen3-Reranker-0.6B Q8_0 | 0.3139 | 0.7188 | 0.3995 | 0.7743 | 0.5516 | 4366 s |

Serving notes the tool surfaces:

- **Encoder rerankers (XLM-R based: Jina v2, BGE m3) must fit a whole pair in one physical batch.**
  At llama-server's default `-ub 512` every pair over 512 tokens is a 500, not a slow path (the
  Qwen3 decoders can split a pair across batches; encoders cannot). Set `-b`/`-ub` to the model's
  trained context and `--ctx-size` to that times `--parallel`, so an oversized pair is a clean 400
  `exceed_context_size` instead. `--ctx-size` is shared by the slots: 8192 with `--parallel 4` is a
  2048-token limit per pair.
- **Qwen3-Reranker is slow on short chunks.** It wraps every pair in a ~80-token instruction
  template, so on 45-word chunks two thirds of the tokens it processes are template — ~13x the
  wall time of Jina v2 above. More slots or a unified KV cache did not help (it is prefill-bound).
- **Scores are not comparable across models.** Qwen3 returns a 0-1 probability, XLM-R encoders raw
  logits (mostly negative). Ranking is unaffected, but a client-side relevance threshold of 0 would
  drop most of an encoder's results.
- GGUFs that are plain causal conversions of a decoder reranker (most community Qwen3-Reranker-4B
  / Qwen3-VL-Reranker files) return ~0 for every document: run the stage-1 baseline plus that model
  and an arm at or below the baseline is the tell.

## Notes

- The full image is large: the ROCm runtime with rocBLAS/hipBLASLt kernel libraries accounts for most of it. The Vulkan-only image is about 650 MB smaller than it was when it extended the upstream image (nothing unreachable is shipped any more).
- `sd-server` (both backends) serves its web UI at `/` — upstream builds it without the frontend (no `pnpm` in its builder), so there `/` is a text placeholder and you need `--serve-html-path`. Here the pinned `sdcpp-webui` is built once (Node stage) and embedded; `/versions.txt` lists its commit as `sd_server_webui:`.
- `audiocpp_gguf` is audio.cpp's GGUF converter, which upstream's image does not ship. It is built from the same tree as `audiocpp_server` (so its model-spec catalog matches the server that loads its output; `/versions.txt`: `audiocpp_commit:`) for the community models whose licence forbids redistributing converted weights — e.g. [audio8_asr](https://github.com/0xShug0/audio.cpp/blob/main/docs/community_models/audio8_asr.md) (Audio8-ASR-0.1B, CC-BY-NC-4.0) is absent from `audio-cpp/audio.cpp-gguf` and has to be converted from the HF checkpoint you download yourself:

  ```sh
  hf download Audio8/Audio8-ASR-0.1B --local-dir /models/Audio8-ASR-0.1B-hf
  audiocpp_gguf --input /models/Audio8-ASR-0.1B-hf/model.safetensors --root /models/Audio8-ASR-0.1B-hf \
      --family audio8_asr --type q8_0 --output /models/Audio8-ASR-0.1B-GGUF/audio8-asr-0.1b-q8_0.gguf
  ```

  The output is self-contained (tokenizer/config sidecars and the model spec are embedded); point the `path` of an `audio8_asr` entry in your audio.cpp server config at the `.gguf` file.
- The container runs as root (standard for ROCm images — device access works without any `--group-add`). Files created in `/models` will be root-owned on the host. If you run as a non-root user, pass the *numeric* host GIDs of your `video`/`render` groups (`--group-add $(getent group render | cut -d: -f3)`); the image has no `render` group, so adding it by name fails.
- Verified on a Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, gfx1151): both `llama-server --list-devices` (Vulkan/RADV) and `llama-server-rocm --list-devices` (ROCm) see the GPU. A gfx1100-only build also works on it with `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- `llama-swap` is built from `main`, not from the latest release archive: releases are cut from `main` every few days and the merged PRs are written against it. Set `LLAMA_SWAP_COMMIT=v255` (any tag) to build a release instead.

## Sources

- [llama-swap unified container docs](https://github.com/mostlygeek/llama-swap/tree/main/docker/unified) (the image this one mirrors in layout)
- [llama.cpp ROCm Dockerfile](https://github.com/ggml-org/llama.cpp/blob/master/.devops/rocm.Dockerfile) (ROCm version + gfx target list followed here)
- [whisper.cpp ROCm build docs](https://github.com/ggml-org/whisper.cpp#amd-rocm-gpu-support)
- [ROCm apt installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager-index.html)
