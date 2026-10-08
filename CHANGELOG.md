# Changelog

The image has no version numbers: every push and the scheduled rebuild publish
from the then-current upstream branches, so changes are grouped by date. Newest
first. The exact revisions inside any given image are in its `/versions.txt`.

## 2026-10-07

- **New `exl3-server`** (`:full` tag, gfx1100 only):
  [phoenixhaxor/exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm),
  exllamav3 with RDNA3 kernels, served by its pinned, patched TabbyAPI
  (OpenAI API, tools, images, reasoning, DFlash2 or MTP speculative decoding).
  A PyTorch 2.13 (ROCm 7.2 wheels) venv under `/opt/exl3`, with the bundled
  kernels pruned to gfx1100. It adds ~9 GB to `:full` and is not in `:vulkan`.
  On an RX 7900 XTX with Qwen3.8-27B (EXL3 3.5 bpw + DFlash2 draft) against
  `llama-server-rdna` (ROCmFP4 + MTP) it measured +14 % to +67 % decode and
  +30-36 % prefill. `:full` also gains `gcc` and the Python headers, because
  Triton compiles its kernel launchers at runtime. New build args `WITH_EXL3`,
  `EXL3_REPO`, `EXL3_BRANCH`, `EXL3_COMMIT`; `/versions.txt` keys `exl3_*`.
  See the README section "exllamav3 for RDNA3".
- **Removed the `*-fpx` binaries** (LaurentZuijdwijk's ROCmFPx fork). The
  `*-rdna` build carries the same ROCmFPx formats, adaptive drafting and
  `quantize`/`perplexity` tools on current upstream master, while the ROCmFPx
  fork had not merged upstream since 2026-08-30. Use `llama-server-rdna` /
  `llama-quantize-rdna` / `llama-perplexity-rdna` instead. Build args
  `WITH_FPX`, `FPX_REPO`, `FPX_BRANCH`, `FPX_COMMIT` are gone.
- **Removed the `*-engram` binaries** (EngramHalo.cpp, `:full` tag, gfx1151
  only). Upstream master now has Qwen3.8-Flash-Next MTP (#29761) and its sparse
  attention, and `llama-server-rocm` already measured faster than the fork.
  Build args `WITH_ENGRAM`, `ENGRAM_REPO`, `ENGRAM_BRANCH`, `ENGRAM_COMMIT`,
  `ENGRAM_TARGETS` are gone, and the `:full` build is one HIP compile shorter.
- **Renamed the RDNA fork** to [SelfRef/llama.cpp-rdna](https://github.com/SelfRef/llama.cpp-rdna)
  (branch `rdna`): binaries `*-rdna3` → `*-rdna`, `/opt/llama-rdna`, build args
  `WITH_RDNA` / `RDNA_REPO` / `RDNA_BRANCH` / `RDNA_COMMIT`, `/versions.txt`
  keys `llama_rdna_*`, `benchmark` variant `rdna`. Configs calling
  `llama-server-rdna3` must be updated.
- `benchmark`: the SPILL check now also runs for entries with a numeric
  `-ngl` ≥ 999, not only `-ngl all`, so a model spilling into GTT is flagged
  instead of silently reporting 4-5x lower numbers.
- `benchmark`: non-JSON responses from the llama-swap API are handled.
- Example config: the `qwen38-fp4` entry sets `--ctx-size 131072` and
  f16 K / q4_0 V; the default context with f16 K and V does not fit in 24 GB.
- README: build args as a list instead of a table, the current patch set, a
  section on the RDNA fork; stale PR history condensed in the Dockerfile.

## 2026-10-05

- `LLAMA_PATCHES`: retired #28956, #28876 (merged), #28243 (superseded by
  #29761), #28927 (superseded by #28751), #28213 (closed); dropped #28699 and
  #27210, which no longer merge; #28333 is carried as
  `patches/28333-rebased.patch`. This fixed the failing scheduled builds of
  2026-10-01 and 2026-10-04.
- New `rerank-bench` CLI: nDCG@10 of the config's reranker entries over a cached
  embedding stage on MTEB retrieval sets.
- `benchmark`: comments are stripped when parsing an entry's `cmd`; tool calls
  are hashed without their random ids.

## 2026-09-24

- `LLAMA_PATCHES`: retired #27952 (int8 coopmat1 MMQ for RDNA3/4, merged
  upstream) and #28943 (closed upstream).

## 2026-09-20

- Image renamed to `ghcr.io/selfref/llama-swap-rdna`; the OCI title is derived
  from the repository name.
- CI: a manual run publishes `:full`/`:latest` by default (the `rocm` checkbox
  is ticked).
- `benchmark`: every preset except `reasoning` sends `enable_thinking: false`.

## 2026-09-19

- License changed from GPL-3.0 to AGPL-3.0.

## 2026-09-18

- **New `*-rdna3` binaries** from SelfRef's own fork: the ROCmFPx formats on top
  of current upstream master, plus measured RDNA3/RDNA3.5 Vulkan patches
  (build args `WITH_RDNA3`, `RDNA3_REPO`, `RDNA3_BRANCH`, `RDNA3_COMMIT`).
- #25666 carried as `patches/25666-rebased.patch`; the local #28243 patch
  removed after the PR was rebased upstream.

## 2026-09-16

- `LLAMA_PATCHES`: added #25666, #28927, #28956, #28876, #28901, #28943.

## 2026-09-15

- `benchmark`: sampled requests are the default again; new `backend` column
  (`<variant>@<device>`).

## 2026-09-12

- `benchmark`: a model name is required (`--all` for every entry); greedy
  default (reverted on 09-15).

## 2026-09-11

- `LLAMA_PATCHES`: #27952 restored after upstream rebased it; #28699 added;
  #28489 dropped (no longer merges).

## 2026-09-10

- `benchmark`: `--ppl` mode (perplexity over wikitext-2, optional KL divergence
  against a reference).

## 2026-09-09

- **New `*-fpx` binaries** from LaurentZuijdwijk's ROCmFPx fork: ROCmFP4/FPx
  weight formats stock llama.cpp cannot load, adaptive draft sizing.
- `LLAMA_PATCHES` trimmed: #27952 and #28136 dropped (no longer merge),
  merged PRs retired.

## 2026-09-08

- `benchmark`: vision presets (accuracy + speed on synthetic images).
- llama-swap #1099 retired from `LLAMA_SWAP_PATCHES` (merged upstream).

## 2026-09-07

- `benchmark` improvements.

## 2026-09-06

- **Built from `ubuntu:24.04`** instead of on top of upstream's
  `unified-vulkan` image; llama-swap itself is built from source with optional
  open PRs (`LLAMA_SWAP_PATCHES`), and is the entrypoint.
- One patched llama.cpp tree for Vulkan and ROCm (`LLAMA_PATCHES`); the separate
  `-next`/`-qwen4exp` builds are gone.
- Scheduled rebuild every 3 days (was weekly); `:vulkan`, `:full` and `:latest`
  tags.

## 2026-09-05

- New `benchmark` CLI for the config's text entries.

## 2026-09-04

- llama.cpp patch set updated (#28330 added).

## 2026-09-03

- `audiocpp_gguf`, audio.cpp's GGUF converter, added to the image.

## 2026-09-02

- ROCm from AMD's per-gfx multi-arch repository (`ROCM_CHANNEL=multiarch`,
  ROCm 7.14), classic channel kept as an option.
- Separate `-qwen4exp` llama.cpp build for Qwen3.8-Flash-Next with its open PRs.

## 2026-09-01

- **New `*-engram` binaries** (EngramHalo.cpp, Strix Halo, `:full` tag only).

## 2026-08-31

- CI: retry step for GHCR rate limits on large images.

## 2026-08-30

- Second bundled chat template, `qwen-sharp.jinja`.

## 2026-08-29

- Vulkan built with a modern `glslc` (integer-dot and bf16 shaders enabled) and
  shipped with kisak Mesa; ROCm 7.2.4; flash-attention kernels for all KV quant
  types; llama.cpp master + #27952.
- `WITH_ROCM=false` builds a Vulkan-only image; sd-server web UI embedded;
  bundled `qwen-fixed.jinja` chat template.

## 2026-08-28

- First version: upstream's `unified-vulkan` llama-swap image with ROCm builds
  of the engines added.
