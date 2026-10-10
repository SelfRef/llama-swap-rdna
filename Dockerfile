# llama-swap for AMD GPUs — ROCm + Vulkan in one image, built from source
#
# The same shape as ghcr.io/mostlygeek/llama-swap:unified-vulkan (binary names
# in /usr/local/bin, config at /etc/llama-swap/config/config.yaml, models in
# /models, port 8080) so a config written for that image works here, but built
# from plain ubuntu:24.04 with no upstream image in the chain. Everything in it
# is compiled here from the projects' current default branches:
#
#   - llama-swap itself, from source (LLAMA_SWAP_COMMIT + the open upstream PRs
#     in LLAMA_SWAP_PATCHES, web UI embedded) and vllm-wrapper from the same
#     tree -- so an open llama-swap PR ships in THE llama-swap binary, not as a
#     side binary.
#   - llama.cpp from current master + the open upstream PRs in LLAMA_PATCHES
#     (Qwen fixes, checkpoint restore, MTP fixes, ... -- the list with
#     rationale is at the LLAMA_PATCHES arg), the
#     SAME tree for the Vulkan and the ROCm build. There is no un-patched
#     llama.cpp in the image any more: llama-server IS the patched build.
#   - the ROCm userspace runtime (HIP runtime, rocBLAS/hipBLAS + Tensile
#     kernels, hipBLASLt, rocminfo) from AMD's apt repository
#   - HIP builds of llama.cpp, whisper.cpp and stable-diffusion.cpp, installed
#     as *-rocm binaries next to the Vulkan ones:
#       llama-server-rocm, llama-cli-rocm, llama-tts-rocm, llama-bench-rocm,
#       whisper-server-rocm, whisper-cli-rocm, sd-server-rocm, sd-cli-rocm
#   - SelfRef/llama.cpp-rdna as *-rdna binaries: the ROCmFPx formats
#     (ROCmFP4 & co., GGUF types stock llama.cpp cannot load) and adaptive
#     draft sizing on current upstream master, plus unupstreamed RDNA3/RDNA3.5
#     Vulkan patches
#     measured on gfx1100/1101/1151 -- see the WITH_RDNA arg below
#   - exllamav3-rocm (exllamav3 with RDNA3 kernels) + TabbyAPI in a PyTorch
#     venv under /opt/exl3, launched as `exl3-server` (gfx1100 only, :full
#     only) -- see the WITH_EXL3 arg below
#   - Vulkan builds of llama.cpp, whisper.cpp, sd.cpp and audio.cpp with a
#     MODERN glslc. Upstream builds them on Ubuntu 24.04 with its stock glslc
#     (shaderc 2023.8 / glslang 14), which cannot compile the
#     GL_EXT_integer_dot_product and GL_EXT_bfloat16 shaders, so llama.cpp's
#     CMake silently drops those code paths (the device line at startup shows
#     "int dot: 0 | bf16: 0" even though RADV advertises both). The integer
#     dot path is the fast quantized prompt/matvec path on GPUs without
#     cooperative-matrix support (q8_1 MMQ for K-quants: ~2x pp on RDNA2 in
#     upstream's numbers, DP4A flash attention for q8_0/q4_0 KV caches, MMVQ
#     decode); on coopmat GPUs (RDNA3+) llama.cpp keeps its FP16 coopmat
#     matmul for prompts, so measured on an RX 7900 XTX the rebuild is
#     neutral for prompt speed (+~2% decode) -- there the big win is the
#     newer Mesa below. We build on the Ubuntu 24.04 ABI but with glslc taken
#     from the Ubuntu 26.04 pocket (see vulkan-builder), and verify at build
#     time that the extensions are compiled in, so nothing is silently left
#     out for any GPU.
#   - llama.cpp (both backends) built with GGML_BACKEND_DL +
#     GGML_CPU_ALL_VARIANTS: the CPU backend is compiled once per x86 feature
#     level and the best one is picked at runtime, so a single image gets
#     AVX2 on Zen 3 (e.g. 5950X) and AVX-512/VNNI/BF16 on Zen 4/5 (e.g. Strix
#     Halo) for CPU-offloaded experts. Upstream ships one generic AVX2 build.
#   - llama.cpp ROCm built with GGML_CUDA_FA_ALL_QUANTS so flash attention has
#     kernels for every K/V cache type combination; without it only q8_0/q8_0
#     and q4_0/q4_0 stay on the GPU and e.g. q8_0/q4_0 falls back to the CPU
#     (upstream issue #27761: pp512 drops ~68%).
#   - sd-server with its web UI embedded (upstream builds it without, see the
#     sd-frontend stage).
#   - audio.cpp's server and CLI (Vulkan, as upstream) PLUS its GGUF converter
#     audiocpp_gguf (upstream ships no converter), all from one tree so the
#     converter's model-spec catalog matches the server that loads its output.
#   - a current Mesa/RADV from the kisak PPA instead of Ubuntu 24.04's.
#
# Why not FROM the upstream image any more: once llama-swap is built here the
# base contributed one apt line, two audio.cpp binaries and three commit
# hashes -- and cost ~650 MB per pull in binaries that were deleted in a child
# layer but still shipped in the parent layers, plus a daily base rebuild that
# invalidated every final-stage layer whether or not anything relevant changed.
# Nothing of upstream's docker/unified is vendored either: the entrypoint is
# llama-swap itself (defaults in CMD, so container arguments replace them).
#
# Versions: every project is built from the ref in its *_COMMIT arg (default:
# the current default branch). Because nothing in the build context changes
# between scheduled runs, CI resolves those refs to commits first
# (scripts/resolve-refs.sh) and passes them as build args -- a moved branch is
# then a cache miss for exactly the stages that use it. A local
# `docker buildx build .` resolves per stage at build time (fine on one
# machine; use `$(scripts/resolve-refs.sh --docker)` to pin). Everything that
# was built, with the PRs merged, is recorded in /versions.txt.
#
# Layout: llama.cpp is installed as self-contained directories
# /opt/llama-vulkan, /opt/llama-rocm and /opt/llama-rdna
# (binaries + their shared libs, RPATH $ORIGIN, ggml backends discovered next
# to the executable) with symlinks in /usr/local/bin, so the builds never
# share a libggml.
# whisper/sd/audio.cpp binaries are static.
#
# Build:
#   docker buildx build -t llama-swap-rdna .
#
# Run (container is root, so no --group-add is needed for device access):
#   docker run -it --rm --device /dev/kfd --device /dev/dri \
#     --security-opt seccomp=unconfined \
#     -p 8080:8080 -v $PWD/models:/models llama-swap-rdna
#
# See README.md for build args, GPU support and runtime env vars.

# ── llama-swap ─────────────────────────────────────────────────────────
# Revision of mostlygeek/llama-swap to build llama-swap and vllm-wrapper from
# (branch, tag such as v255, or sha). main: releases are cut from it every few
# days and the open PRs below are written against it.
ARG LLAMA_SWAP_COMMIT="main"

# Open upstream llama-swap PRs merged on top, same rules as LLAMA_PATCHES
# (closed PRs are skipped with a notice, conflicts fail the build). Empty since
# 2026-09-07: #1099 (ui/playground live per-turn generation stats in the Chat
# tab) merged upstream on 2026-09-07 and is in main.
ARG LLAMA_SWAP_PATCHES=""

# Cache key only: CI sets it to the PR heads' shas (scripts/resolve-refs.sh) so
# an updated PR rebuilds llama-swap even though the PR list did not change.
ARG LLAMA_SWAP_PATCHES_HEADS=""

# ── ROCm ───────────────────────────────────────────────────────────────
# ROCm source channel. AMD now ships ROCm through two repositories:
#   multiarch (default) — repo.amd.com/rocm/packages-multi-arch/ubuntu2404:
#     the current releases (ROCM_SERIES, e.g. 7.14 -> apt picks 7.14.1), split
#     into per-gfx packages, so the runtime image carries BLAS kernels only
#     for AMDGPU_TARGETS (~50 MB per arch) instead of the classic ~6 GB
#     all-arch rocBLAS/hipBLASLt Tensile blobs. Installs under
#     /opt/rocm/core-<series>/, symlinked back to the classic /opt/rocm
#     layout in the stages below.
#   classic — repo.radeon.com/rocm/apt + the rocm/dev-ubuntu-24.04 builder
#     images: tops out at 7.2.4 (checked 2026-09-01) and carries the HIP
#     graphs bug fixed in 7.13 (needs GGML_CUDA_DISABLE_GRAPHS=1, see
#     llama.cpp discussion #27950). Kept as a fallback.
ARG ROCM_CHANNEL=multiarch

# classic channel only: builder image tag + apt repo path.
ARG ROCM_VERSION=7.2.4

# multiarch channel only: the release series embedded in the package names
# (amdrocm-runtime7.14, amdrocm-blas7.14-gfx1151, ...); apt then resolves the
# newest point release of that series. Bump when AMD publishes the next one.
ARG ROCM_SERIES=7.14

# Build the ROCm side at all? true = full image (Vulkan + ROCm runtime + *-rocm
# binaries); false = Vulkan-only image, the HIP builder stages are not even
# started (BuildKit only builds stages the final one references). CI publishes
# the Vulkan-only image as :vulkan and the full one as :full / :latest.
ARG WITH_ROCM=true

# gfx architectures compiled into the HIP binaries: RX 7900 (gfx1100), RX 7800/
# 7700 (gfx1101), Strix Halo (gfx1151) and RDNA4 RX 9070/9060 (gfx1201/gfx1200).
# Every target multiplies the HIP compile time (all-quant FA kernels x every
# target), so the list is kept to the RDNA3/3.5 cards this image is tuned on
# plus current RDNA4.
# Not compiled in, but each is one entry away (the table in the README's
# "Choosing ROCm vs Vulkan"): RDNA2 gfx1030, RX 7600 gfx1102, Strix Point
# gfx1150, and the CDNA data-center targets of llama.cpp's official ROCm image
# (gfx908;gfx90a;gfx942).
# GPUs not listed can still use the Vulkan binaries.
ARG AMDGPU_TARGETS="gfx1100;gfx1101;gfx1151;gfx1200;gfx1201"

# Compile flash-attention kernels for all K/V cache quant combinations in the
# ROCm llama.cpp build (see header). Costs build time and binary size; set to
# OFF to build faster.
ARG LLAMA_FA_ALL_QUANTS=ON

# ── Vulkan ─────────────────────────────────────────────────────────────
# Ubuntu release whose glslc/libshaderc1 are installed into the (24.04) Vulkan
# builder. Only those two packages come from it (per-package release selection
# + low pin), everything else stays 24.04 so the binaries run on the runtime
# image's glibc. 26.04 "resolute" ships shaderc 2026.1 / glslang 16.
ARG GLSLC_SUITE=resolute

# Newer Mesa (RADV, the Vulkan driver) for the final image. Ubuntu 24.04's
# stock Mesa 25.2 is a year behind; kisak-mesa tracks the current stable
# release (26.1 at the time of writing). Measured on an RX 7900 XTX with the
# SAME Vulkan binaries: pp512 693 -> 856 t/s (+24%), decode unchanged, and the
# newer RADV exposes VK_VALVE_shader_mixed_float_dot_product (fp16 "dot2").
# Set to "" to keep Ubuntu's stock Mesa.
ARG MESA_PPA="ppa:kisak/kisak-mesa"

# ── llama.cpp ──────────────────────────────────────────────────────────
# llama.cpp revision for BOTH llama.cpp builds (Vulkan and ROCm): sha, tag,
# branch, or refs/pull/N/head. Default: current master, so that the open PRs
# below apply and the image carries the newest backend work.
ARG LLAMA_COMMIT="master"

# Upstream llama.cpp pull requests merged on top of LLAMA_COMMIT, for BOTH
# backends, as a space-separated list of PR numbers (fetched over git as
# refs/pull/N/head and merged in list order; the .patch endpoint is
# rate-limited on CI runners). A PR that is closed on GitHub (merged or
# rejected) is skipped with a notice; one that no longer merges cleanly FAILS
# the build, so drift is never a silent no-op. Dry-run the whole set with
# scripts/checkout-with-prs.sh on a local clone before changing it.
#
# One tree for both backends since 2026-09-06 (none of the PRs touches
# CUDA/HIP sources, so the ROCm build takes the same tree).
#
# Current set, revised 2026-10-05 against master e117148a4:
#   #28265 keep the Qwen3.5-family delta-net out-proj 2D (Strix Halo: +6-9 %
#          TG at batch 4-8, i.e. --parallel 2 plus MTP verify batches)
#   #25592 exact-position checkpoint restore for hybrid/recurrent models
#          (agentic multi-turn @130k: 35 s -> 1.3 s turn restore)
#   #30283 clamp the A prefetch to end_k in the int8 coopmat1 matmul (RDNA3/4),
#          as the B prefetch already is; without it a row whose K is not a
#          multiple of 128 reads the next row's blocks (NaN on a non-finite
#          scale). Added 2026-10-10; MoE pp2048 -0.4 %, decode flat. The
#          llama.cpp-rdna binaries carry the same fix.
# plus two PRs carried as rebased patches because they no longer merge as-is
# (see patches/README.md): #25666 (no MMVQ for speculative-decode steps on
# AMD; only gfx1151 numbers exist upstream -- delete it if a discrete-GPU
# benchmark shows no gain) and #28333 (zero the MTP carrier at sequence start).
#
# Retired, merged upstream: #27952 (int8 coopmat1 MMQ for RDNA3/4, 09-24;
#   pp512 +4.6 % dense / +18.5 % MoE on an RX 7900 XTX. Its final form also
#   brings the IQ4_XS MMQ/MMV pair #28440/#28415, which produced subtly broken
#   UD-Q4_K_XL output on RDNA3 when tested on 09-09 -- check the text, not
#   only t/s, on IQ4_XS-bearing quants), #28024, #27220, #28253, #28068,
#   #28457, #28330, #28901, #28956, #28876.
# Retired, superseded or closed: #28243 (-> #29761 Qwen4Exp MTP), #28927
#   (-> #28751), #28213 (master's sparse FA and k-pool cache match it),
#   #28943 (closed without merge).
# Dropped because they no longer merge: #28699 (master's own k-pool cache
#   took the same lines), #27210 (draft-mtp-adaptive; conflicts with the
#   rejection-sampling rewrite #27694 -- the *-rdna binaries still carry it),
#   #28489 (MMVQ path selection, +2-5 % MoE decode only), #28136 (lazy PLE
#   direct reads; this removed the `--lazy-mode on-direct` value, master has
#   only on/auto/off).
# Evaluated and left out: #28092 (--cache-disk; conflicts with #25592, the
#   bigger TTFT win), #28528 (stream-k MUL_MAT; the cm1 heuristic is only
#   enabled on coopmat2, a no-op on RADV), #28611 (RDNA3 L-tile warp
#   micro-dimension; gated to UMA on purpose, does not merge), #27332
#   (MUL_MAT_VEC_ID density gate; batch > 8 MoE decode only), #28507 (FA shmem
#   staging; neutral on a 7900 XTX), #25483 (+0.3 %), #26284 + #26301 (HIP MMQ;
#   +2 %), #22970 (stale), #28849 (auto-fit over ctx x parallel slots; a
#   behaviour change, not a speed-up).
#
# Test a changed set locally before pushing: `docker build --target
# llama-vulkan .` (and `--target llama-rocm --build-arg AMDGPU_TARGETS=gfx1100`)
# takes minutes instead of hours of runner time. The dated history of every
# decision above is in this file's git log.
# patches/*.patch (local rebased patches) apply after the merges to both
# backends -- so a patch has to be generated against the tree with ALL the
# other merges in it, not just against master (see patches/README.md).
# Retire PRs from the list as they merge (the build says so).
ARG LLAMA_PATCHES="28265 25592 30283"

# Cache key only (see LLAMA_SWAP_PATCHES_HEADS).
ARG LLAMA_PATCHES_HEADS=""

# ── RDNA fork (SelfRef/llama.cpp-rdna) ───────────────────────────────
# SelfRef's fork of ggml-org/llama.cpp, built as another Vulkan llama.cpp
# install (/opt/llama-rdna, *-rdna binaries). Created 2026-09-18 because the
# two things RDNA3 hardware needs had never been in one tree:
#
#   - the ROCmFPx weight formats and tooling from LaurentZuijdwijk's
#     llama.cpp fork (the base of this branch), which upstream does not carry
#     and probably will not until 0cc4m's #28898 lands FP8/NVFP4 quant scales
#     in ggml;
#   - a set of RDNA3/RDNA3.5 Vulkan patches that were written against upstream,
#     measured, and then NEVER submitted -- they exist only as patch files
#     passed between community forks (voidsurfer/llama.cpp-nudge <- Nathan
#     Wilson's strix-halo-vulkan, plus Gaetan Puleo's server fixes).
#
# LaurentZuijdwijk's fork cannot host either: it last merged upstream on
# 2026-08-30 and its owner decides what it carries (the image shipped it as
# *-fpx until 2026-10-07). Branch `rdna` = that ROCmFPx base plus upstream
# master plus only the patches that measured as a win on the target cards,
# every patch on its own `carry/*` branch so one bad upstream merge does not
# take the rest with it.
#
# What the ROCmFPx base brings:
#
#   - the ROCmFPx weight formats (Q4_0_ROCMFP4 / _FAST / Q2/Q3/Q6/Q8_0_ROCMFPX,
#     ggml type ids 100-107, hand-ported from ciru-ai/ROCmFPX <- charlie12345/
#     ROCmFPX where the format originates) with CPU codecs plus Vulkan dequant,
#     mat-vec, matmul and integer-dot kernels. Stock llama.cpp cannot LOAD these
#     files at all (`Q4_0_ROCMFP4_FAST` is GGUF file type 103).
#   - a reworked batch-3..8 mat-vec path (whole-block MMVQ for the FP4 types,
#     branch-free fp6/fp3 dequant, an IQ3_S register-spill fix) -- exactly the
#     widths speculative decoding verifies at.
#   - `--spec-draft-adaptive` / `--spec-draft-n-min`: draft length follows the
#     measured acceptance rate instead of a fixed `--spec-draft-n-max`.
#   - Vulkan prefill tuning that also pays on stock K-quants: an LDS bank-conflict
#     fix in the coopmat matmul tile stride (driver-gated to RADV >= 25.3), f16 B
#     operand for quantized matmul/matmul_id, a tiled concat-transpose for the
#     delta-net conv state.
#
# Measured on an RX 7900 XTX (gfx1100, RADV/Mesa 26.2.2, 2026-09-09) -- Qwen3.8-27B with its
# baked MTP head, greedy prose / json / refactor decode t/s:
#   stock llama-server, unsloth UD-Q4_K_XL (16.35 GiB), n-max 3
#       60.9 / 83.6 / 94.7   prefill 423 / 328 / 777   VRAM 22.3 G  PPL 6.637
#   ROCmFPx fork (the former llama-server-fpx), julianmb ROCmFP4-FAST
#   (13.55 GiB, 4.25 bpw), n-max 4
#       76.7 / 105.5 / 128.0 prefill 457 / 343 / 921   VRAM 19.0 G  PPL 6.921
# i.e. +26 / +26 / +35 % decode and -3.3 GiB for +4.3 % perplexity. The FP4
# format is a software codebook, NOT hardware FP4 (no RDNA GPU has FP4 matrix
# instructions): the decode win is bytes moved per token plus those batch-3..8
# kernels. On the same card the fork's engine work alone, on the same K-quant
# file, is +5-7 % prefill and neutral decode -- so the file, not the binary, is
# where most of it comes from; both are needed.
#
# Targets, and the reason for the name: gfx1100 (RX 7900 XTX), gfx1101
# (RX 7800 XT) and gfx1151 (Strix Halo / Ryzen AI Max+ 395) -- one architecture
# family (RDNA3 + RDNA3.5). Nothing else is accepted into the branch.
#
# Vulkan-only, and LLAMA_PATCHES is NOT applied here: the branch merges
# upstream master itself (re-ported onto it 2026-09-18, last merged 2026-10-07
# at 36a73916) and carries the upstream PRs it wants as its own commits. What
# it carries and why is in the fork's README.
#
# ALWAYS build with RDNA_COMMIT pinned -- scripts/resolve-refs.sh resolves it for
# you. The clone happens inside a RUN whose cache key is the ARG VALUES, so a
# build that passes only RDNA_BRANCH silently reuses the layer from whatever the
# branch pointed at last time: measured 2026-09-18, a rebuild after a force-push
# returned the PREVIOUS tip's binary and reported the old commit in
# /versions.txt.
# Pass the FULL 40-char sha, never an abbreviation: the clone is a
# `git fetch --depth=1 origin <ref>`, and GitHub rejects a short sha in a want
# line -- the stage then fails with a bare `exit code: 128`.
# Canaries guard it: if a merge ever drops the ROCmFPx types or adaptive
# drafting, the build FAILS instead of shipping a plain llama.cpp under the
# -rdna name.
ARG WITH_RDNA=true
ARG RDNA_REPO=https://github.com/SelfRef/llama.cpp-rdna.git
ARG RDNA_BRANCH=rdna
ARG RDNA_COMMIT=""

# ── exllamav3 for RDNA3 (TabbyAPI) ─────────────────────────────────────
# phoenixhaxor/exllamav3-rocm: exllamav3 with its EXL3 matmul, decode/verify
# attention and Gated DeltaNet kernels rewritten for RDNA3 (wave32, WMMA),
# served by TabbyAPI (OpenAI API: streaming, tools, images, reasoning) with
# DFlash2 or MTP speculative decoding. Not a llama.cpp: a PyTorch (ROCm wheels)
# venv in /opt/exl3 with an `exl3-server` launcher, so a llama-swap entry runs
# it like any other server (`exl3-server --port ${PORT} --model-dir ...`).
# Measured on an RX 7900 XTX with Qwen3.8-27B EXL3 3.5 bpw + DFlash2 draft vs
# llama.cpp ROCmFP4 + MTP, same chat template: decode +14 % prose, +49 % JSON,
# +67 % code, equal at 8k depth; prefill +30-36 % (README has the table).
# gfx1100 ONLY (the fork's kernels are wave32/RDNA3 and validated on the XTX),
# so EXL3_TARGETS is not a list. Only with WITH_ROCM=true AND WITH_EXL3=true:
# it is a HIP engine and adds ~8 GB (PyTorch's bundled ROCm libraries, already
# pruned to gfx1100), which the Vulkan-only image must not carry. Built on the
# classic ROCm toolchain whatever ROCM_CHANNEL says: the PyTorch wheels are
# ROCm 7.2 builds and the fork is tested on 7.2.4 (= ROCM_VERSION).
ARG WITH_EXL3=true
ARG EXL3_REPO=https://github.com/phoenixhaxor/exllamav3-rocm.git
ARG EXL3_BRANCH=main
ARG EXL3_COMMIT=""

# ── whisper.cpp, stable-diffusion.cpp, audio.cpp ───────────────────────
# Revisions (branch, tag or sha) of the other engines; their default branches.
ARG WHISPER_COMMIT="master"
ARG SD_COMMIT="master"
ARG AUDIOCPP_COMMIT="main"

# ── Chat templates ─────────────────────────────────────────────────────
# Sources of the fixed Qwen chat templates shipped under
# /etc/llama-swap/templates/ (fetched at build time):
#   qwen-fixed.jinja -- froggeric's Qwen-Fixed-Chat-Templates (the base fix)
#   qwen-sharp.jinja -- peculiar-ragdoll's Qwen-Sharp-Chat-Templates: froggeric's
#                       template rebased with a terseness system prompt spliced in
#                       (opt out per request with chat_template_kwargs {"terse": false})
ARG QWEN_TEMPLATE_URL="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja"
ARG QWEN_SHARP_TEMPLATE_URL="https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/resolve/main/chat_template.jinja"

# ── Final-stage cache key ──────────────────────────────────────────────
# Declared in the final stage right before its apt layers: CI passes the run's
# timestamp so Ubuntu updates, the PPA Mesa and the ROCm runtime are refreshed
# on every run (minutes). Locally, leave it empty and those layers stay cached.
ARG BUILD_DATE=""

# ══════════════════════════════════════════════════════════════════════
# ── Vulkan builder: Ubuntu 24.04 ABI + modern glslc ────────────────────

FROM ubuntu:24.04 AS vulkan-builder
ARG GLSLC_SUITE

ENV DEBIAN_FRONTEND=noninteractive
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=5G

# libav*-dev only for whisper.cpp's WHISPER_FFMPEG=ON; the final stage installs
# the matching Ubuntu 24.04 libav* runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ccache curl ca-certificates \
        pkg-config libssl-dev \
        libvulkan-dev spirv-headers spirv-tools \
        libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# glslc + libshaderc1 from the newer Ubuntu pocket, nothing else (pin 100 keeps
# apt from preferring that release; the pkg/suite syntax selects it explicitly).
# Their only dependencies are libc6 >= 2.38 / libstdc++6 >= 13.1, satisfied by
# 24.04. The three feature tests in the llama.cpp stage are the ones llama.cpp's
# CMake runs; integer_dot and bfloat16 FAIL with 24.04's own glslc, which is the
# whole reason this stage exists -- so a regression there must fail the build.
RUN echo "deb http://archive.ubuntu.com/ubuntu ${GLSLC_SUITE} main universe" \
        > /etc/apt/sources.list.d/glslc.list \
    && printf 'Package: *\nPin: release n=%s\nPin-Priority: 100\n' "${GLSLC_SUITE}" \
        > /etc/apt/preferences.d/glslc-suite \
    && apt-get update \
    && apt-get install -y --no-install-recommends "glslc/${GLSLC_SUITE}" "libshaderc1/${GLSLC_SUITE}" \
    && rm -rf /var/lib/apt/lists/* \
    && glslc --version

COPY --chmod=0755 scripts/checkout-with-prs.sh /usr/local/bin/checkout-with-prs.sh
WORKDIR /build

# ── Build llama.cpp (Vulkan) ───────────────────────────────────────────

FROM vulkan-builder AS llama-vulkan
ARG LLAMA_COMMIT
ARG LLAMA_PATCHES
ARG LLAMA_PATCHES_HEADS
COPY patches/ /build/patches/
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

# master + the open PRs + local patches; see scripts/checkout-with-prs.sh for
# the skip/fail rules.
LOCAL_PATCHES=/build/patches checkout-with-prs.sh \
    https://github.com/ggml-org/llama.cpp.git "${LLAMA_COMMIT}" /src/llama.cpp ${LLAMA_PATCHES:-}
cd /src/llama.cpp

echo "=== glslc feature tests (llama.cpp's own) ==="
for t in integer_dot bfloat16 coopmat; do
    f="ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/$t.comp"
    [ -f "$f" ] || { echo "(no feature test $t in this revision, skipping)"; continue; }
    if glslc -o /dev/null -fshader-stage=compute --target-env=vulkan1.3 "$f" >/dev/null 2>&1; then
        echo "  $t: OK"
    else
        echo "FATAL: glslc cannot compile $f -- the Vulkan build would lose that code path" >&2
        exit 1
    fi
done

echo "=== Building llama.cpp (Vulkan) ==="
# BACKEND_DL + CPU_ALL_VARIANTS need shared libs; building with the install
# RPATH ($ORIGIN, nothing else) makes binaries + libs relocatable as one
# directory (ggml also searches for its backend libs next to the executable).
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_VULKAN=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' \
    2>&1 | tee /tmp/configure.log
for ext in GL_EXT_integer_dot_product GL_EXT_bfloat16 GL_KHR_cooperative_matrix; do
    line=$(grep -i "$ext" /tmp/configure.log || true)
    echo "  cmake: ${line:-<no message for $ext>}"
    if grep -qi "not supported" <<<"$line"; then
        echo "FATAL: CMake reports $ext unsupported by glslc" >&2; exit 1; fi
done
# "all" so the per-feature-level ggml-cpu variants and the backend libs get
# built too (they are not link-time dependencies of the executables).
cmake --build build --config Release -j"$(nproc)"

echo "=== Collecting ==="
OUT=/install/llama-vulkan
mkdir -p "$OUT" /install/build-info
for bin in llama-server llama-cli llama-tts llama-bench; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    cp "build/bin/$bin" "$OUT/"
done
cp -P build/bin/*.so* "$OUT/"
ls "$OUT"/libggml-cpu-*.so >/dev/null 2>&1 || { echo "FATAL: no ggml-cpu variants built" >&2; exit 1; }
ls "$OUT"/libggml-vulkan.so >/dev/null 2>&1 || { echo "FATAL: libggml-vulkan.so not built" >&2; exit 1; }
# Relocatable check: every ELF's run path must start with $ORIGIN and must not
# point into the build tree.
for f in "$OUT"/*; do
    [ -L "$f" ] && continue
    rp=$(readelf -d "$f" 2>/dev/null | awk '/RUNPATH|RPATH/ {gsub(/[\[\]]/,"",$NF); print $NF}')
    if [ -n "$rp" ] && { [[ "$rp" != '$ORIGIN'* ]] || [[ "$rp" == */src/* ]]; }; then
        echo "FATAL: $f has run path '$rp' (expected \$ORIGIN[:...])" >&2; exit 1; fi
    if ldd "$f" 2>/dev/null | grep -q "not found"; then
        echo "FATAL: $f has unresolved libraries" >&2; ldd "$f" | grep "not found" >&2; exit 1; fi
done
# A backend .so with an UNDEFINED SYMBOL links fine, then fails dlopen at runtime and
# ggml silently falls back to the CPU -- measured 2026-09-18: a lost definition in the
# Vulkan backend produced a binary that passed every canary above, answered correctly,
# and ran Qwen3.8-27B at 2.7 t/s on 16 CPU threads. ldd -r resolves symbols, ldd does not.
# APPEND to LD_LIBRARY_PATH, never replace it: the multiarch ROCm toolchain reaches
# /opt/rocm/lib ONLY through that variable (it adds no ld.so.conf entry -- the runtime
# image does, which is why the shipped libs resolve there), so overwriting it hides
# libamdhip64.so and reports every HIP symbol as undefined. Treat "not found" as fatal
# too, so a dependency that cannot be located can never masquerade as a clean run.
LDD_PATH="$OUT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for lib in "$OUT"/libggml-*.so; do
    if LD_LIBRARY_PATH="$LDD_PATH" ldd -r "$lib" 2>&1 | grep -qE "undefined symbol|not found"; then
        echo "FATAL: $lib has undefined symbols (would fail dlopen -> silent CPU fallback):" >&2
        LD_LIBRARY_PATH="$LDD_PATH" ldd -r "$lib" 2>&1 | grep -E "undefined symbol|not found" | head -5 >&2; exit 1; fi
done
{ echo "llama_vulkan_commit: $(cat .base-commit) (requested: ${LLAMA_COMMIT}; merged tree $(git rev-parse --short HEAD))";
  echo "llama_patches: $(cat .merged-prs)";
  echo "llama_local_patches: $(cat .local-patches)";
  echo "vulkan_glslc: $(glslc --version | head -1)"; } > /install/build-info/llama-vulkan
BUILD

# ── Build the RDNA fork (Vulkan) ──────────────────────────────────────
# SelfRef/llama.cpp-rdna -> /opt/llama-rdna, *-rdna binaries. Same
# vulkan-builder and ccache as llama-vulkan; see the WITH_RDNA arg above for
# what the branch carries.

FROM vulkan-builder AS llama-rdna
ARG RDNA_REPO
ARG RDNA_BRANCH
ARG RDNA_COMMIT
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

REF="${RDNA_COMMIT:-${RDNA_BRANCH}}"
echo "=== Cloning the RDNA3 fork (${RDNA_BRANCH} @ ${REF}) ==="
mkdir -p /src/llama-rdna && cd /src/llama-rdna
git init -q
git remote add origin "${RDNA_REPO}"
git fetch --depth=1 origin "${REF}"
git checkout -q FETCH_HEAD
echo "fork at $(git rev-parse HEAD)"

echo "=== glslc feature tests (llama.cpp's own) ==="
for t in integer_dot bfloat16 coopmat; do
    f="ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/$t.comp"
    [ -f "$f" ] || { echo "(no feature test $t in this revision, skipping)"; continue; }
    if glslc -o /dev/null -fshader-stage=compute --target-env=vulkan1.3 "$f" >/dev/null 2>&1; then
        echo "  $t: OK"
    else
        echo "FATAL: glslc cannot compile $f -- the Vulkan build would lose that code path" >&2
        exit 1
    fi
done

echo "=== Building the RDNA3 fork (Vulkan) ==="
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_VULKAN=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' \
    2>&1 | tee /tmp/configure-rdna.log
for ext in GL_EXT_integer_dot_product GL_EXT_bfloat16 GL_KHR_cooperative_matrix; do
    line=$(grep -i "$ext" /tmp/configure-rdna.log || true)
    echo "  cmake: ${line:-<no message for $ext>}"
    if grep -qi "not supported" <<<"$line"; then
        echo "FATAL: CMake reports $ext unsupported by glslc" >&2; exit 1; fi
done
cmake --build build --config Release -j"$(nproc)"

echo "=== Collecting ==="
OUT=/install/llama-rdna
mkdir -p "$OUT" /install/build-info
for bin in llama-server llama-cli llama-bench llama-quantize llama-perplexity; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    cp "build/bin/$bin" "$OUT/${bin}-rdna"
done
cp -P build/bin/*.so* "$OUT/"
ls "$OUT"/libggml-cpu-*.so >/dev/null 2>&1 || { echo "FATAL: no ggml-cpu variants built" >&2; exit 1; }
ls "$OUT"/libggml-vulkan.so >/dev/null 2>&1 || { echo "FATAL: libggml-vulkan.so not built" >&2; exit 1; }
# Same relocatable check as the llama-vulkan stage.
for f in "$OUT"/*; do
    [ -L "$f" ] && continue
    rp=$(readelf -d "$f" 2>/dev/null | awk '/RUNPATH|RPATH/ {gsub(/[\[\]]/,"",$NF); print $NF}')
    if [ -n "$rp" ] && { [[ "$rp" != '$ORIGIN'* ]] || [[ "$rp" == */src/* ]]; }; then
        echo "FATAL: $f has run path '$rp' (expected \$ORIGIN[:...])" >&2; exit 1; fi
    if ldd "$f" 2>/dev/null | grep -q "not found"; then
        echo "FATAL: $f has unresolved libraries" >&2; ldd "$f" | grep "not found" >&2; exit 1; fi
done
# A backend .so with an UNDEFINED SYMBOL links fine, then fails dlopen at runtime and
# ggml silently falls back to the CPU -- measured 2026-09-18: a lost definition in the
# Vulkan backend produced a binary that passed every canary above, answered correctly,
# and ran Qwen3.8-27B at 2.7 t/s on 16 CPU threads. ldd -r resolves symbols, ldd does not.
# APPEND to LD_LIBRARY_PATH, never replace it: the multiarch ROCm toolchain reaches
# /opt/rocm/lib ONLY through that variable (it adds no ld.so.conf entry -- the runtime
# image does, which is why the shipped libs resolve there), so overwriting it hides
# libamdhip64.so and reports every HIP symbol as undefined. Treat "not found" as fatal
# too, so a dependency that cannot be located can never masquerade as a clean run.
LDD_PATH="$OUT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for lib in "$OUT"/libggml-*.so; do
    if LD_LIBRARY_PATH="$LDD_PATH" ldd -r "$lib" 2>&1 | grep -qE "undefined symbol|not found"; then
        echo "FATAL: $lib has undefined symbols (would fail dlopen -> silent CPU fallback):" >&2
        LD_LIBRARY_PATH="$LDD_PATH" ldd -r "$lib" 2>&1 | grep -E "undefined symbol|not found" | head -5 >&2; exit 1; fi
done
# The whole point of this stage: the ROCmFPx tensor types and adaptive
# drafting must be present. If a later upstream merge in the fork drops
# either, this build fails instead of silently shipping a plain llama.cpp
# under the -rdna name (config entries that reference these files would then
# fail to load a model at runtime).
# (both binaries print their help to stdout and EXIT 1, so capture first --
# a `cmd | grep` would fail the build through `pipefail`, not through grep.)
QHELP=$("$OUT/llama-quantize-rdna" --help 2>&1 || true)
grep -q 'Q4_0_ROCMFP4_FAST' <<<"$QHELP" || {
    echo "FATAL: llama-quantize-rdna does not know the ROCmFPx types -- the fork lost them" >&2
    head -5 <<<"$QHELP" >&2; exit 1; }
SHELP=$("$OUT/llama-server-rdna" --help 2>&1 || true)
grep -q -- '--spec-draft-adaptive' <<<"$SHELP" || {
    echo "FATAL: llama-server-rdna has no --spec-draft-adaptive -- the fork lost adaptive drafting" >&2; exit 1; }
{ echo "llama_rdna_commit: $(git rev-parse HEAD) (${RDNA_REPO} @ ${RDNA_BRANCH})";
  echo "llama_rdna_types: $(sed -n 's/^ *[0-9]* *or *\(Q[0-9]_[0-9]_ROCM[A-Z0-9_]*\) .*/\1/p' <<<"$QHELP" | sort -u | tr '\n' ' ')"; } \
  > /install/build-info/llama-rdna
BUILD

# ── Build whisper.cpp (Vulkan) ─────────────────────────────────────────

FROM vulkan-builder AS whisper-vulkan
ARG WHISPER_COMMIT
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning whisper.cpp at ${WHISPER_COMMIT} ==="
mkdir -p /src/whisper.cpp && cd /src/whisper.cpp
git init -q
git remote add origin https://github.com/ggml-org/whisper.cpp.git
git fetch --depth=1 origin "${WHISPER_COMMIT}"
git checkout -q FETCH_HEAD
echo "whisper.cpp at $(git rev-parse HEAD)"

echo "=== Building whisper.cpp (Vulkan, static) ==="
# Static: no shared libggml in /usr/local/lib to collide with anything.
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_VULKAN=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DWHISPER_FFMPEG=ON
cmake --build build --config Release -j"$(nproc)" \
    --target whisper-server whisper-cli

mkdir -p /install/bin /install/build-info
for bin in whisper-server whisper-cli; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    if readelf -d "build/bin/$bin" | grep -q 'libggml\|libwhisper'; then
        echo "FATAL: $bin is not statically linked against ggml/whisper" >&2; exit 1; fi
    cp "build/bin/$bin" /install/bin/
done
echo "whisper_vulkan_commit: $(git rev-parse HEAD) (requested: ${WHISPER_COMMIT})" > /install/build-info/whisper-vulkan
BUILD

# ── sd-server web UI (built once, embedded into both sd-server builds) ──
# sd.cpp embeds its frontend (the sdcpp-webui submodule, pinned per revision)
# when CMake finds pnpm -- or a pre-generated frontend/dist/gen_index_html.h.
# Upstream's builder has neither, so its sd-server answers "/" with a text
# placeholder. Build the header here with Node and hand it to the C++ stages.

FROM node:22-alpine AS sd-frontend
RUN apk add --no-cache git && corepack enable
ARG SD_COMMIT
RUN <<'BUILD'
#!/bin/sh
set -eu
mkdir -p /src/sd && cd /src/sd
git init -q && git remote add origin https://github.com/leejet/stable-diffusion.cpp.git
git fetch --depth=1 origin "${SD_COMMIT}" && git checkout -q FETCH_HEAD
git submodule update --init --depth=1 examples/server/frontend
cd examples/server/frontend
echo "sd_server_webui: embedded ($(git rev-parse HEAD))" > /src/frontend-version
pnpm install --frozen-lockfile
pnpm run build
pnpm run build:header
[ -f dist/gen_index_html.h ] || { echo "FATAL: gen_index_html.h not produced" >&2; exit 1; }
BUILD

# ── Build stable-diffusion.cpp (Vulkan) ────────────────────────────────

FROM vulkan-builder AS sd-vulkan
ARG SD_COMMIT
COPY --from=sd-frontend /src/sd/examples/server/frontend/dist/gen_index_html.h /src/frontend-version /tmp/sd-frontend/
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning stable-diffusion.cpp at ${SD_COMMIT} ==="
mkdir -p /src/stable-diffusion.cpp && cd /src/stable-diffusion.cpp
git init -q
git remote add origin https://github.com/leejet/stable-diffusion.cpp.git
git fetch --depth=1 origin "${SD_COMMIT}"
git checkout -q FETCH_HEAD
git submodule update --init --recursive --depth=1
echo "stable-diffusion.cpp at $(git rev-parse HEAD)"
# Pre-built web UI header (see the sd-frontend stage) -> embedded frontend
mkdir -p examples/server/frontend/dist
cp /tmp/sd-frontend/gen_index_html.h examples/server/frontend/dist/

echo "=== Building stable-diffusion.cpp (Vulkan) ==="
mkdir -p build
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_VULKAN=ON \
    -DSD_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DSD_BUILD_EXAMPLES=ON \
    2>&1 | tee /tmp/configure.log
grep -q "using pre-built frontend header" /tmp/configure.log \
    || { echo "FATAL: sd-server would be built WITHOUT its web UI (pre-built header not picked up)" >&2; exit 1; }
cmake --build build --config Release -j"$(nproc)" \
    --target sd-server sd-cli

mkdir -p /install/bin /install/build-info
for bin in sd-server sd-cli; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    if readelf -d "build/bin/$bin" | grep -q 'libggml\|libstable'; then
        echo "FATAL: $bin expects shared ggml/sd libs" >&2; exit 1; fi
    cp "build/bin/$bin" /install/bin/
done
{ echo "sd_vulkan_commit: $(git rev-parse HEAD) (requested: ${SD_COMMIT})";
  cat /tmp/sd-frontend/frontend-version; } > /install/build-info/sd-vulkan
BUILD

# ── Build audio.cpp (Vulkan): audiocpp_server, audiocpp_cli, audiocpp_gguf ──
# Server and CLI exactly as upstream's install-audio.sh builds them for the
# vulkan flavour, plus the GGUF converter that upstream does not ship:
#   AUDIOCPP_DEPLOYMENT_BUILD=ON compiles the model_specs/*.json catalog into
#   the binaries (a bare binary otherwise fails with "model spec not found"
#   for anything that is not a GGUF with an embedded spec); the on-disk catalog
#   is installed too so --model-spec-override has a path to point at.
#   ENGINE_ENABLE_NATIVE_CPU=OFF: portable CPU kernels, and it keeps the build
#   static (audio.cpp only switches to shared libs under CPU_ALL_VARIANTS).
# audiocpp_gguf turns a HF safetensors checkpoint into an audio.cpp GGUF
# package (weights + embedded tokenizer/config sidecars + the family's spec
# from the compiled-in catalog):
#   audiocpp_gguf --input <ckpt>/model.safetensors --root <ckpt> \
#       --family audio8_asr --type q8_0 --output <out>/audio8-asr-0.1b-q8_0.gguf
# Needed for community models whose licence forbids redistributing converted
# weights (audio8_asr, CC-BY-NC-4.0), absent from audio-cpp/audio.cpp-gguf.
# One tree for all three, so the converter's catalog matches the server.

FROM vulkan-builder AS audiocpp
ARG AUDIOCPP_COMMIT
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning audio.cpp at ${AUDIOCPP_COMMIT} ==="
mkdir -p /src/audio.cpp && cd /src/audio.cpp
git init -q
git remote add origin https://github.com/0xShug0/audio.cpp.git
git fetch --depth=1 origin "${AUDIOCPP_COMMIT}"
git checkout -q FETCH_HEAD
echo "audio.cpp at $(git rev-parse HEAD)"

echo "=== Building audio.cpp (Vulkan, static) ==="
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
    -DAUDIOCPP_MODEL_SET=full \
    -DENGINE_ENABLE_NATIVE_CPU=OFF \
    -DENGINE_ENABLE_OPENMP=ON \
    -DENGINE_BUILD_EXAMPLES=OFF \
    -DENGINE_BUILD_TESTS=OFF \
    -DENGINE_BUILD_WARMBENCH=OFF \
    -DENGINE_ENABLE_CUDA=OFF \
    -DENGINE_ENABLE_HIP=OFF \
    -DENGINE_ENABLE_VULKAN=ON
cmake --build build --config Release -j"$(nproc)" \
    --target audiocpp_cli audiocpp_server audiocpp_gguf

mkdir -p /install/bin /install/share/audiocpp /install/build-info
for bin in audiocpp_cli audiocpp_server audiocpp_gguf; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    needed=$(readelf -d "build/bin/$bin" | grep NEEDED || true)
    # A backend that silently failed to enable still produces working binaries
    # that fall back to CPU at runtime -- catch it here.
    if [ "$bin" != audiocpp_gguf ] && ! grep -q 'libvulkan\.so' <<<"$needed"; then
        echo "FATAL: $bin is not linked against libvulkan:" >&2; echo "$needed" >&2; exit 1; fi
    if grep -qE 'libggml|libengine' <<<"$needed"; then
        echo "FATAL: $bin expects audio.cpp shared libraries; only static builds are installed" >&2
        echo "$needed" >&2; exit 1; fi
    cp "build/bin/$bin" /install/bin/
done
cp -r model_specs /install/share/audiocpp/model_specs
# Usage exits non-zero; the point is that the converter loads and runs.
{ /install/bin/audiocpp_gguf 2>&1 || true; } | grep -q '^Usage: audiocpp_gguf' \
    || { echo "FATAL: audiocpp_gguf does not run" >&2; exit 1; }
echo "audiocpp_commit: $(git rev-parse HEAD) (requested: ${AUDIOCPP_COMMIT})" > /install/build-info/audiocpp
BUILD

# ── Build llama-swap + vllm-wrapper from source ────────────────────────
# Three stages: fetch + merge PRs (git), build the Svelte UI (node), build the
# Go binaries with the UI embedded (`-tags embed_ui`, see the upstream Makefile
# and internal/server/embed.go). vllm-wrapper is not in upstream's release
# archives either, so it comes from the same tree.

FROM golang:1.27-bookworm AS llama-swap-src
ARG LLAMA_SWAP_COMMIT
ARG LLAMA_SWAP_PATCHES
ARG LLAMA_SWAP_PATCHES_HEADS
COPY --chmod=0755 scripts/checkout-with-prs.sh /usr/local/bin/checkout-with-prs.sh
RUN <<'FETCH'
#!/bin/bash
set -euo pipefail
FETCH_TAGS=1 checkout-with-prs.sh \
    https://github.com/mostlygeek/llama-swap.git "${LLAMA_SWAP_COMMIT}" /src/llama-swap ${LLAMA_SWAP_PATCHES:-}
cd /src/llama-swap
# Version string as upstream's Makefile derives it (git describe on the base
# commit) plus a +prN suffix per merged PR, e.g. v255-2-g1a2b3c+pr1099.
BASE=$(cat .base-commit)
VERSION=$(git describe --tags --abbrev=6 "$BASE" 2>/dev/null || echo devel)
for pr in $(cat .merged-prs); do VERSION="${VERSION}+pr${pr}"; done
COMMIT=$(git rev-parse --short "$BASE")
[ -z "$(cat .merged-prs)" ] || COMMIT="${COMMIT}+"
{ echo "LS_VERSION=${VERSION}"; echo "LS_COMMIT=${COMMIT}"; } > .version
mkdir -p /install/build-info
{ echo "llama_swap_version: ${VERSION}";
  echo "llama_swap_commit: ${BASE} (requested: ${LLAMA_SWAP_COMMIT}; merged tree $(git rev-parse --short HEAD))";
  echo "llama_swap_patches: $(cat .merged-prs)"; } > /install/build-info/llama-swap
FETCH

FROM node:24-bookworm-slim AS llama-swap-ui
COPY --from=llama-swap-src /src/llama-swap/ui /src/ui
# vite.config.ts writes to ../internal/server/ui_dist
RUN --mount=type=cache,id=npm,target=/root/.npm \
    cd /src/ui && npm ci --no-audit --no-fund && npm run build \
    && test -f /src/internal/server/ui_dist/index.html

FROM golang:1.27-bookworm AS llama-swap-build
COPY --from=llama-swap-src /src/llama-swap /src/llama-swap
COPY --from=llama-swap-src /install/build-info /install/build-info
COPY --from=llama-swap-ui /src/internal/server/ui_dist /src/llama-swap/internal/server/ui_dist
RUN --mount=type=cache,id=go-build,target=/root/.cache/go-build \
    --mount=type=cache,id=go-mod,target=/go/pkg/mod <<'BUILD'
#!/bin/bash
set -euo pipefail
cd /src/llama-swap
. ./.version
DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== Building llama-swap ${LS_VERSION} (${LS_COMMIT}) ==="
mkdir -p /install/bin
CGO_ENABLED=0 go build -trimpath -tags embed_ui \
    -ldflags="-s -w -X main.version=${LS_VERSION} -X main.commit=${LS_COMMIT} -X main.date=${DATE}" \
    -o /install/bin/llama-swap .
echo "=== Building vllm-wrapper ==="
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /install/bin/vllm-wrapper ./cmd/vllm-wrapper
/install/bin/llama-swap -version
# The UI must actually be inside the binary (the embed_ui tag with an empty
# ui_dist would build a server that 404s on /).
grep -q '<!doctype html' /install/bin/llama-swap || grep -qi '<!DOCTYPE html' /install/bin/llama-swap \
    || { echo "FATAL: llama-swap binary does not contain the embedded UI" >&2; exit 1; }
BUILD

# ══════════════════════════════════════════════════════════════════════
# ── ROCm toolchain (selected by ROCM_CHANNEL, see the arg) ─────────────

# classic: AMD's prebuilt dev image; the build scripts derive HIPCXX/HIP_PATH
# from its hipconfig.
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete AS rocm-toolchain-classic

# multiarch: plain Ubuntu 24.04 + the dev packages from
# repo.amd.com/rocm/packages-multi-arch (AMD publishes no prebuilt dev image
# for these releases). Everything lands under /opt/rocm/core-<series>/; the
# classic layout is symlinked back so ROCM_PATH=/opt/rocm and the cmake
# configs keep working. amdrocm-runtime-dev pulls the HIP headers/cmake and
# the LLVM toolchain (amdclang++, device bitcode); blas-host/-dev +
# hipblas-common-dev provide librocblas/libhipblas(lt) with headers and cmake
# configs for ggml-hip's find_package(hipblas/rocblas); solver-host provides
# librocsolver, which libhipblas.so references since 7.14 (static links fail
# without it and the runtime needs it on the NEEDED chain); amdrocm-base has
# rocminfo/rocm_agent_enumerator. There is no hipconfig here — HIPCXX and
# HIP_PATH are exported instead and the build scripts prefer them when set.
FROM ubuntu:24.04 AS rocm-toolchain-multiarch
ARG ROCM_SERIES
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg \
        | gpg --dearmor -o /etc/apt/keyrings/rocm-multiarch.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm-multiarch.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main" \
        > /etc/apt/sources.list.d/rocm-multiarch.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        "amdrocm-runtime-dev${ROCM_SERIES}" \
        "amdrocm-blas-host${ROCM_SERIES}" \
        "amdrocm-blas-dev${ROCM_SERIES}" \
        "amdrocm-hipblas-common-dev${ROCM_SERIES}" \
        "amdrocm-solver-host${ROCM_SERIES}" \
        "amdrocm-base${ROCM_SERIES}" \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s "core-${ROCM_SERIES}" /opt/rocm/core \
    && for d in bin include lib libexec share; do \
        ln -s "core-${ROCM_SERIES}/$d" "/opt/rocm/$d"; done \
    && ln -s "core-${ROCM_SERIES}/lib/llvm" /opt/rocm/llvm \
    && ln -s "core-${ROCM_SERIES}/lib/llvm/amdgcn" /opt/rocm/amdgcn \
    && test -x /opt/rocm/lib/llvm/bin/amdclang++ \
    && test -f /opt/rocm/lib/cmake/hip/hip-config.cmake
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIPCXX=/opt/rocm/lib/llvm/bin/amdclang++ \
    PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib/rocm_sysdeps/lib:/opt/rocm/lib

# ── ROCm builder base ──────────────────────────────────────────────────

FROM rocm-toolchain-${ROCM_CHANNEL} AS rocm-builder
ARG AMDGPU_TARGETS

ENV DEBIAN_FRONTEND=noninteractive
ENV AMDGPU_TARGETS=${AMDGPU_TARGETS}
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=5G

# libav*-dev only for whisper.cpp's WHISPER_FFMPEG=ON; the final stage installs
# the matching Ubuntu 24.04 libav* runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ccache curl ca-certificates \
        pkg-config libssl-dev \
        libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --chmod=0755 scripts/checkout-with-prs.sh /usr/local/bin/checkout-with-prs.sh
WORKDIR /build

# ── Build llama.cpp (HIP) ──────────────────────────────────────────────

FROM rocm-builder AS llama-rocm
ARG LLAMA_COMMIT
ARG LLAMA_FA_ALL_QUANTS
ARG LLAMA_PATCHES
ARG LLAMA_PATCHES_HEADS
COPY patches/ /build/patches/
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

# Same tree as the Vulkan build: master + the open PRs + local patches.
LOCAL_PATCHES=/build/patches checkout-with-prs.sh \
    https://github.com/ggml-org/llama.cpp.git "${LLAMA_COMMIT}" /src/llama.cpp ${LLAMA_PATCHES:-}
cd /src/llama.cpp

echo "=== Building llama.cpp (HIP) for ${AMDGPU_TARGETS}, FA_ALL_QUANTS=${LLAMA_FA_ALL_QUANTS} ==="
# Shared + BACKEND_DL + CPU_ALL_VARIANTS like llama.cpp's own ROCm image; the
# HIP backend lives in libggml-hip.so next to the binaries (RPATH $ORIGIN).
# POSITION_INDEPENDENT_CODE: ggml-hip's device-stub objects are non-PIC by
# default and fail to link into Ubuntu's default-PIE executables.
HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}" HIP_PATH="${HIP_PATH:-$(hipconfig -R)}" \
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
    -DGGML_CUDA_FA_ALL_QUANTS="${LLAMA_FA_ALL_QUANTS}" \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN'
cmake --build build --config Release -j"$(nproc)"

echo "=== Collecting ==="
OUT=/install/llama-rocm
mkdir -p "$OUT" /install/build-info
for bin in llama-server llama-cli llama-tts llama-bench; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    cp "build/bin/$bin" "$OUT/${bin}-rocm"
done
cp -P build/bin/*.so* "$OUT/"
[ -f "$OUT/libggml-hip.so" ] || { echo "FATAL: libggml-hip.so not built" >&2; exit 1; }
readelf -d "$OUT/libggml-hip.so" | grep -q 'libamdhip64\.so' || {
    echo "FATAL: libggml-hip.so is not linked against the HIP runtime" >&2; exit 1; }
ls "$OUT"/libggml-cpu-*.so >/dev/null 2>&1 || { echo "FATAL: no ggml-cpu variants built" >&2; exit 1; }
# Relocatable check: every ELF's run path must start with $ORIGIN and must not
# point into the build tree (CMake may append toolchain lib dirs such as
# /opt/rocm-*/lib for the HIP backend -- those exist in the runtime image).
for f in "$OUT"/*; do
    [ -L "$f" ] && continue
    rp=$(readelf -d "$f" 2>/dev/null | awk '/RUNPATH|RPATH/ {gsub(/[\[\]]/,"",$NF); print $NF}')
    if [ -n "$rp" ] && { [[ "$rp" != '$ORIGIN'* ]] || [[ "$rp" == */src/* ]]; }; then
        echo "FATAL: $f has run path '$rp' (expected \$ORIGIN[:...])" >&2; exit 1; fi
    if ldd "$f" 2>/dev/null | grep -q "not found"; then
        echo "FATAL: $f has unresolved libraries" >&2; ldd "$f" | grep "not found" >&2; exit 1; fi
done
# A backend .so with an UNDEFINED SYMBOL links fine, then fails dlopen at runtime and
# ggml silently falls back to the CPU -- measured 2026-09-18: a lost definition in the
# Vulkan backend produced a binary that passed every canary above, answered correctly,
# and ran Qwen3.8-27B at 2.7 t/s on 16 CPU threads. ldd -r resolves symbols, ldd does not.
# APPEND to LD_LIBRARY_PATH, never replace it: the multiarch ROCm toolchain reaches
# /opt/rocm/lib ONLY through that variable (it adds no ld.so.conf entry -- the runtime
# image does, which is why the shipped libs resolve there), so overwriting it hides
# libamdhip64.so and reports every HIP symbol as undefined. Treat "not found" as fatal
# too, so a dependency that cannot be located can never masquerade as a clean run.
LDD_PATH="$OUT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for lib in "$OUT"/libggml-*.so; do
    if LD_LIBRARY_PATH="$LDD_PATH" ldd -r "$lib" 2>&1 | grep -qE "undefined symbol|not found"; then
        echo "FATAL: $lib has undefined symbols (would fail dlopen -> silent CPU fallback):" >&2
        LD_LIBRARY_PATH="$LDD_PATH" ldd -r "$lib" 2>&1 | grep -E "undefined symbol|not found" | head -5 >&2; exit 1; fi
done
{ echo "llama_rocm_commit: $(cat .base-commit) (requested: ${LLAMA_COMMIT}; merged tree $(git rev-parse --short HEAD))";
  echo "llama_rocm_patches: $(cat .merged-prs)";
  echo "llama_rocm_local_patches: $(cat .local-patches)";
  echo "rocm_fa_all_quants: ${LLAMA_FA_ALL_QUANTS}"; } > /install/build-info/llama-rocm
BUILD

# ── Build exllamav3-rocm + TabbyAPI (see the WITH_EXL3 arg) ────────────

FROM rocm-toolchain-classic AS exl3
ARG EXL3_REPO
ARG EXL3_BRANCH
ARG EXL3_COMMIT
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*
RUN <<'BUILD'
#!/bin/bash
set -euo pipefail
REF="${EXL3_COMMIT:-${EXL3_BRANCH}}"
echo "=== Cloning exllamav3-rocm (${EXL3_BRANCH} @ ${REF}) ==="
mkdir -p /opt/exl3/exllamav3 && cd /opt/exl3/exllamav3
git init -q
git remote add origin "${EXL3_REPO}"
git fetch --depth=1 origin "${REF}"
git checkout -q FETCH_HEAD
COMMIT=$(git rev-parse HEAD)
rm -rf .git

# The fork's rocm/scripts/setup_env.sh, minus huggingface_hub (its httpx2-based
# releases pin a tokenizers range with no wheel, so pip falls back to building
# tokenizers 0.13 from source and fails without Rust; nothing at runtime needs it).
python3.12 -m venv /opt/exl3/venv
. /opt/exl3/venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install --no-cache-dir tokenizers "numpy>=2.1" rich typing_extensions safetensors ninja \
    pillow pyyaml marisa_trie pydantic "llguidance>=1.7.0" setuptools wheel

# ROCm 7.2.4's clang 22 rejects the device-only rsqrtf() in host code; harmless
# no-op once the fork fixes it.
sed -i 's/sm_scale == 0.0f ? rsqrtf((float) dim)/sm_scale == 0.0f ? 1.0f \/ sqrtf((float) dim)/' \
    exllamav3/exllamav3_ext/attention.cu

echo "=== Building exllamav3_ext for gfx1100 ==="
# = rocm/scripts/build.sh without its final import check (that one needs a GPU)
CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ PYTORCH_ROCM_ARCH=gfx1100 \
MAX_JOBS="$(nproc)" ROCM_HOME=/opt/rocm python setup.py build_ext --inplace --force
EXLLAMA_NOCOMPILE=1 pip install --no-cache-dir -e . --no-deps --no-build-isolation
rm -rf build

echo "=== TabbyAPI ==="
# The fork's installer pins the tested TabbyAPI commit and applies its RDNA3
# patch. TabbyAPI ships its own models/ dir, so the installer's symlink lands
# inside it; drop both, models are passed with --model-dir.
mkdir -p /tmp/models
bash rocm/scripts/install_tabbyapi.sh /opt/exl3/tabbyAPI /tmp/models
TABBY_COMMIT=$(git -C /opt/exl3/tabbyAPI rev-parse HEAD)
rm -rf /opt/exl3/tabbyAPI/.git /opt/exl3/tabbyAPI/models /opt/exl3/tabbyAPI/config.yml /tmp/models
mkdir -p /opt/exl3/tabbyAPI/models
# Stream deltas carry no "role"; LangChain's OpenAI client (LibreChat, n8n)
# then builds a generic chunk and drops the delta's tool_calls, so a tool call
# streams as an empty reply. Add the role OpenAI and llama-server send.
sed -i 's/^\(        delta\["tool_calls"\] = delta_tool\)$/\1\n    if delta:\n        delta["role"] = "assistant"/' \
    /opt/exl3/tabbyAPI/endpoints/OAI/utils/chat_completion.py
grep -q 'delta\["role"\] = "assistant"' /opt/exl3/tabbyAPI/endpoints/OAI/utils/chat_completion.py \
    || { echo "FATAL: TabbyAPI stream-delta role patch no longer applies" >&2; exit 1; }
# The image's Qwen templates, selectable with --prompt-template qwen-fixed / qwen-sharp.
ln -s /etc/llama-swap/templates/qwen-fixed.jinja /opt/exl3/tabbyAPI/templates/qwen-fixed.jinja
ln -s /etc/llama-swap/templates/qwen-sharp.jinja /opt/exl3/tabbyAPI/templates/qwen-sharp.jinja

# PyTorch's wheels bundle rocBLAS/hipBLASLt/MIOpen/AOTriton kernels for every
# ROCm GPU (~7 GB); the extension only ever runs on gfx1100.
SP=/opt/exl3/venv/lib/python3.12/site-packages
before=$(du -sm "$SP" | cut -f1)
find "$SP/torch/lib" "$SP/torch/share" -regextype posix-extended -type f \
    -regex '.*gfx[0-9]+[a-z0-9]*.*' ! -regex '.*gfx(1100|110x)([^0-9a-z].*|$)' -delete
find "$SP/torch/lib" "$SP/torch/share" -type d -empty -delete
echo "pruned $(( before - $(du -sm "$SP" | cut -f1) )) MB of non-gfx1100 kernels"
find /opt/exl3 -name __pycache__ -type d -prune -exec rm -rf {} +

# Canary: the extension must import (no GPU needed to load it) and TabbyAPI must
# accept the RDNA3 backend.
cd /
python -c "import torch; from exllamav3.ext import exllamav3_ext; print('exllamav3_ext loads, torch', torch.__version__, torch.version.hip)"
(cd /opt/exl3/tabbyAPI && python main.py --help >/dev/null)

mkdir -p /opt/exl3/bin
cat > /opt/exl3/bin/exl3-server <<'EOF'
#!/bin/sh
# TabbyAPI on exllamav3-rocm. Every TabbyAPI config key is also a flag
# (`exl3-server --help`); --config is optional. EXL3_NOGRAPH=mlp,gdn is the
# fork's default (eager MLP/DeltaNet decode is ~0.5 % faster than HIP graphs).
export PYTHONPATH="/opt/exl3/exllamav3${PYTHONPATH:+:$PYTHONPATH}"
export EXL3_NOGRAPH="${EXL3_NOGRAPH-mlp,gdn}"
cd /opt/exl3/tabbyAPI
exec /opt/exl3/venv/bin/python main.py "$@"
EOF
chmod 755 /opt/exl3/bin/exl3-server

mkdir -p /install/build-info
{ echo "exl3_commit: ${COMMIT} (${EXL3_REPO} @ ${EXL3_BRANCH})";
  echo "exl3_tabbyapi_commit: ${TABBY_COMMIT}";
  echo "exl3_torch: $(python -c 'import torch; print(torch.__version__)')";
  echo "exl3_targets: gfx1100"; } > /install/build-info/exl3
BUILD

# ── Build whisper.cpp (HIP) ────────────────────────────────────────────

FROM rocm-builder AS whisper-rocm
ARG WHISPER_COMMIT
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning whisper.cpp at ${WHISPER_COMMIT} ==="
mkdir -p /src/whisper.cpp && cd /src/whisper.cpp
git init -q
git remote add origin https://github.com/ggml-org/whisper.cpp.git
git fetch --depth=1 origin "${WHISPER_COMMIT}"
git checkout -q FETCH_HEAD
echo "whisper.cpp at $(git rev-parse HEAD)"

echo "=== Building whisper.cpp (HIP) for ${AMDGPU_TARGETS} ==="
# POSITION_INDEPENDENT_CODE: see llama.cpp stage
HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}" HIP_PATH="${HIP_PATH:-$(hipconfig -R)}" \
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DWHISPER_FFMPEG=ON \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}"
cmake --build build --config Release -j"$(nproc)" \
    --target whisper-server whisper-cli

mkdir -p /install/bin /install/build-info
for bin in whisper-server whisper-cli; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    needed=$(readelf -d "build/bin/$bin" | grep NEEDED || true)
    grep -q 'libamdhip64\.so' <<<"$needed" || {
        echo "FATAL: $bin is not linked against the HIP runtime:" >&2
        echo "$needed" >&2; exit 1; }
    if grep -q 'libggml' <<<"$needed"; then
        echo "FATAL: $bin expects shared ggml libs" >&2; exit 1; fi
    cp "build/bin/$bin" "/install/bin/${bin}-rocm"
done
echo "whisper_rocm_commit: $(git rev-parse HEAD) (requested: ${WHISPER_COMMIT})" > /install/build-info/whisper-rocm
BUILD

# ── Build stable-diffusion.cpp (HIP) ───────────────────────────────────

FROM rocm-builder AS sd-rocm
ARG SD_COMMIT
COPY --from=sd-frontend /src/sd/examples/server/frontend/dist/gen_index_html.h /src/frontend-version /tmp/sd-frontend/
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning stable-diffusion.cpp at ${SD_COMMIT} ==="
mkdir -p /src/stable-diffusion.cpp && cd /src/stable-diffusion.cpp
git init -q
git remote add origin https://github.com/leejet/stable-diffusion.cpp.git
git fetch --depth=1 origin "${SD_COMMIT}"
git checkout -q FETCH_HEAD
git submodule update --init --recursive --depth=1
echo "stable-diffusion.cpp at $(git rev-parse HEAD)"
# Pre-built web UI header (see the sd-frontend stage) -> embedded frontend
mkdir -p examples/server/frontend/dist
cp /tmp/sd-frontend/gen_index_html.h examples/server/frontend/dist/

echo "=== Building stable-diffusion.cpp (HIP) for ${AMDGPU_TARGETS} ==="
# POSITION_INDEPENDENT_CODE: see llama.cpp stage (sd.cpp also sets it itself)
HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}" HIP_PATH="${HIP_PATH:-$(hipconfig -R)}" \
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DSD_BUILD_EXAMPLES=ON \
    -DSD_HIPBLAS=ON \
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
    2>&1 | tee /tmp/configure.log
grep -q "using pre-built frontend header" /tmp/configure.log \
    || { echo "FATAL: sd-server would be built WITHOUT its web UI (pre-built header not picked up)" >&2; exit 1; }
cmake --build build --config Release -j"$(nproc)" \
    --target sd-server sd-cli

mkdir -p /install/bin /install/build-info
for bin in sd-server sd-cli; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    needed=$(readelf -d "build/bin/$bin" | grep NEEDED || true)
    grep -q 'libamdhip64\.so' <<<"$needed" || {
        echo "FATAL: $bin is not linked against the HIP runtime:" >&2
        echo "$needed" >&2; exit 1; }
    if grep -q 'libggml' <<<"$needed"; then
        echo "FATAL: $bin expects shared ggml libs" >&2; exit 1; fi
    cp "build/bin/$bin" "/install/bin/${bin}-rocm"
done
echo "sd_rocm_commit: $(git rev-parse HEAD) (requested: ${SD_COMMIT})" > /install/build-info/sd-rocm
BUILD

# ── ROCm stage selection (WITH_ROCM) ───────────────────────────────────
# Alias stages: the final stage copies from `<name>-sel`, which FROMs
# `<name>-${WITH_ROCM}`; with false that resolves to this empty stand-in and the
# HIP builders are never started.

FROM alpine:3 AS rocm-none
RUN mkdir -p /install/bin /install/llama-rocm /install/build-info

FROM llama-rocm   AS llama-rocm-true
FROM whisper-rocm AS whisper-rocm-true
FROM sd-rocm      AS sd-rocm-true
FROM rocm-none    AS llama-rocm-false
FROM rocm-none    AS whisper-rocm-false
FROM rocm-none    AS sd-rocm-false
# COPY --from cannot expand variables, FROM can: select here.
FROM llama-rocm-${WITH_ROCM}   AS llama-rocm-sel
FROM whisper-rocm-${WITH_ROCM} AS whisper-rocm-sel
FROM sd-rocm-${WITH_ROCM}      AS sd-rocm-sel

# ── RDNA fork selection (WITH_RDNA) ───────────────────────────────────
# Vulkan-only, so this switch is independent of WITH_ROCM and the stage is
# built for BOTH published tags. Its own empty stand-in keeps the two knobs
# separable (rocm-none exists only to serve the ROCm side).

FROM alpine:3 AS rdna-none
RUN mkdir -p /install/llama-rdna /install/build-info

FROM llama-rdna AS llama-rdna-true
FROM rdna-none  AS llama-rdna-false
FROM llama-rdna-${WITH_RDNA} AS llama-rdna-sel

# ── exllamav3-rocm selection (WITH_ROCM x WITH_EXL3) ──────────────────
# Needs BOTH switches on (it links the ROCm runtime, which only the WITH_ROCM
# image installs), so the selection key is the concatenated pair.
FROM alpine:3 AS exl3-none
RUN mkdir -p /opt/exl3 /install/build-info

FROM exl3      AS exl3-true-true
FROM exl3-none AS exl3-true-false
FROM exl3-none AS exl3-false-true
FROM exl3-none AS exl3-false-false
FROM exl3-${WITH_ROCM}-${WITH_EXL3} AS exl3-sel

# ── Stage outputs (CI handover) ────────────────────────────────────────
# Exactly what the final stage copies out of each builder, nothing else. CI
# builds these as `<stage>-out` in the parallel stage jobs, pushes them as small
# images and hands them to the final build as named build contexts
# (`--build-context llama-rocm=docker-image://...`), which REPLACE the builder
# stage of that name: the final job then cannot recompile anything, whether or
# not a cache matches. Relying on the stage jobs' registry caches alone did not
# work — the final job missed them and rebuilt every engine itself.
# A local `docker buildx build .` never touches these stages.
FROM scratch AS llama-vulkan-out
COPY --from=llama-vulkan /install/ /install/
FROM scratch AS llama-rdna-out
COPY --from=llama-rdna /install/ /install/
FROM scratch AS whisper-vulkan-out
COPY --from=whisper-vulkan /install/ /install/
FROM scratch AS sd-vulkan-out
COPY --from=sd-vulkan /install/ /install/
FROM scratch AS audiocpp-out
COPY --from=audiocpp /install/ /install/
FROM scratch AS llama-swap-build-out
COPY --from=llama-swap-build /install/ /install/
FROM scratch AS llama-rocm-out
COPY --from=llama-rocm /install/ /install/
FROM scratch AS whisper-rocm-out
COPY --from=whisper-rocm /install/ /install/
FROM scratch AS sd-rocm-out
COPY --from=sd-rocm /install/ /install/
FROM scratch AS exl3-out
COPY --from=exl3 /opt/exl3/ /opt/exl3/
COPY --from=exl3 /install/ /install/

# ══════════════════════════════════════════════════════════════════════
# ── Final image: Ubuntu 24.04 runtime (+ ROCm) + everything built above ──

FROM ubuntu:24.04 AS final
ARG ROCM_CHANNEL
ARG ROCM_VERSION
ARG ROCM_SERIES
ARG AMDGPU_TARGETS
ARG MESA_PPA
ARG QWEN_TEMPLATE_URL
ARG QWEN_SHARP_TEMPLATE_URL
ARG WITH_ROCM
ARG WITH_RDNA
ARG WITH_EXL3

LABEL org.opencontainers.image.source="https://github.com/SelfRef/llama-swap-rdna" \
      org.opencontainers.image.description="llama-swap unified image for AMD GPUs (ROCm + Vulkan)"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"

# Cache key for everything below (see the arg's comment at the top).
ARG BUILD_DATE

# Runtime packages: Vulkan loader + RADV, libgomp for the CPU backends, libav*
# + ffmpeg for whisper-server's WHISPER_FFMPEG input decoding, rocm-smi for
# llama-swap's GPU monitor in its UI (sysfs-based, works without the ROCm
# runtime), curl for healthchecks, python3 + PyYAML for the bundled `benchmark`
# CLI. Mesa: only mesa-vulkan-drivers (+ deps) is taken from the PPA, not the
# whole GL stack; software-properties-common is only needed to add it and is
# purged again.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libvulkan1 mesa-vulkan-drivers \
        rocm-smi \
        curl ca-certificates \
        libavcodec60 libavformat60 libavutil58 libswresample4 \
        ffmpeg \
        python3 python3-yaml \
    && if [ -n "${MESA_PPA}" ]; then \
        apt-get install -y --no-install-recommends software-properties-common \
        && add-apt-repository -y "${MESA_PPA}" \
        && apt-get install -y --no-install-recommends --only-upgrade mesa-vulkan-drivers \
        && apt-get purge -y --auto-remove software-properties-common; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# uv/uvx (static binaries from the official image) for `uvx`-launched tools in
# config.yaml commands and maintenance scripts (e.g. `uvx --from huggingface_hub hf`).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ROCm userspace matching the builder's channel (see ROCM_CHANNEL).
# classic: hipblas/rocblas pull in the HIP runtime (libamdhip64), hsa-rocr,
# comgr etc. via package dependencies; hipblaslt is explicit — rocBLAS dlopens
# it on some architectures (gfx90a/gfx942/RDNA4), which no NEEDED-entry check
# can catch. This is the all-arch variant (~6 GB of kernels).
# multiarch: HIP runtime + host-side BLAS libs + rocminfo (amdrocm-base) and
# ONLY the per-gfx kernel packages for AMDGPU_TARGETS (~50 MB per arch);
# layout is /opt/rocm/core-<series>/ symlinked to the classic paths, and
# rocm_sysdeps (AMD's vendored deps the libs link against) needs its own
# ld.so.conf entry.
RUN if [ "${WITH_ROCM}" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends gnupg \
    && mkdir -p /etc/apt/keyrings \
    && if [ "${ROCM_CHANNEL}" = "classic" ]; then \
        curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
            | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
        && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
            > /etc/apt/sources.list.d/rocm.list \
        && printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
            > /etc/apt/preferences.d/rocm-pin-600 \
        && apt-get update \
        && apt-get install -y --no-install-recommends hipblas rocblas hipblaslt rocminfo \
        && echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf; \
    else \
        curl -fsSL https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg \
            | gpg --dearmor -o /etc/apt/keyrings/rocm-multiarch.gpg \
        && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm-multiarch.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main" \
            > /etc/apt/sources.list.d/rocm-multiarch.list \
        && apt-get update \
        && PKGS="amdrocm-runtime${ROCM_SERIES} amdrocm-blas-host${ROCM_SERIES} amdrocm-solver-host${ROCM_SERIES} amdrocm-base${ROCM_SERIES}" \
        && for t in $(echo "${AMDGPU_TARGETS}" | tr ';' ' '); do \
            PKGS="$PKGS amdrocm-blas${ROCM_SERIES}-${t}"; done \
        && apt-get install -y --no-install-recommends $PKGS \
        && ln -s "core-${ROCM_SERIES}" /opt/rocm/core \
        && for d in bin include lib libexec share; do \
            ln -s "core-${ROCM_SERIES}/$d" "/opt/rocm/$d"; done \
        && { echo /opt/rocm/lib; echo /opt/rocm/lib/rocm_sysdeps/lib; } \
            > /etc/ld.so.conf.d/rocm.conf; \
    fi \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig; \
    fi

# exllamav3-rocm's prefill attention is Triton, which compiles a small C launcher
# per kernel signature at runtime with the system C compiler + Python headers.
RUN if [ "${WITH_ROCM}" = "true" ] && [ "${WITH_EXL3}" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev python3.12-dev \
    && rm -rf /var/lib/apt/lists/*; \
    fi

ENV PATH="/opt/rocm/bin:${PATH}"

RUN mkdir -p /etc/llama-swap/config /models

# ── Binaries ──
COPY --from=llama-vulkan   /install/llama-vulkan/ /opt/llama-vulkan/
COPY --from=whisper-vulkan /install/bin/ /usr/local/bin/
COPY --from=sd-vulkan      /install/bin/ /usr/local/bin/
COPY --from=audiocpp       /install/bin/ /usr/local/bin/
COPY --from=audiocpp       /install/share/audiocpp/ /usr/local/share/audiocpp/
COPY --from=llama-swap-build /install/bin/ /usr/local/bin/
COPY --from=llama-rocm-sel   /install/llama-rocm/ /opt/llama-rocm/
COPY --from=whisper-rocm-sel /install/bin/ /usr/local/bin/
COPY --from=sd-rocm-sel      /install/bin/ /usr/local/bin/
COPY --from=llama-rdna-sel  /install/llama-rdna/ /opt/llama-rdna/
COPY --from=exl3-sel         /opt/exl3/ /opt/exl3/
# build-info of every stage -> /versions.txt below
COPY --from=llama-vulkan     /install/build-info/ /tmp/build-info/
COPY --from=whisper-vulkan   /install/build-info/ /tmp/build-info/
COPY --from=sd-vulkan        /install/build-info/ /tmp/build-info/
COPY --from=audiocpp         /install/build-info/ /tmp/build-info/
COPY --from=llama-swap-build /install/build-info/ /tmp/build-info/
COPY --from=llama-rocm-sel   /install/build-info/ /tmp/build-info/
COPY --from=whisper-rocm-sel /install/build-info/ /tmp/build-info/
COPY --from=sd-rocm-sel      /install/build-info/ /tmp/build-info/
COPY --from=llama-rdna-sel  /install/build-info/ /tmp/build-info/
COPY --from=exl3-sel         /install/build-info/ /tmp/build-info/
RUN for bin in llama-server llama-cli llama-tts llama-bench; do \
        ln -sf "/opt/llama-vulkan/$bin" "/usr/local/bin/$bin"; \
        if [ "${WITH_ROCM}" = "true" ]; then \
            ln -sf "/opt/llama-rocm/$bin-rocm" "/usr/local/bin/$bin-rocm"; \
        fi; \
    done \
    && { [ "${WITH_ROCM}" = "true" ] || rmdir /opt/llama-rocm; } \
    && if [ "${WITH_RDNA}" = "true" ]; then \
        for bin in llama-server llama-cli llama-bench llama-quantize llama-perplexity; do \
            ln -sf "/opt/llama-rdna/$bin-rdna" "/usr/local/bin/$bin-rdna"; \
        done; \
    else rmdir /opt/llama-rdna; fi \
    && if [ "${WITH_ROCM}" = "true" ] && [ "${WITH_EXL3}" = "true" ]; then \
        ln -sf /opt/exl3/bin/exl3-server /usr/local/bin/exl3-server; \
    else rmdir /opt/exl3; fi \
    && ldconfig

# Example config with both backends; override by mounting /etc/llama-swap/config
COPY config/config.yaml /etc/llama-swap/config/config.yaml

# `benchmark` CLI (scripts/benchmark): server-level (via llama-swap), kernel-level
# (llama-bench[-variant]) and standalone (llama-server-<variant>) benchmarks of the
# config.yaml text entries, one table. Pure python3 + PyYAML (apt: the image is
# PEP-668 externally managed).
COPY --chmod=0755 scripts/benchmark /usr/local/bin/benchmark

# `rerank-bench` CLI (scripts/rerank-bench): nDCG@10 of reranker entries over a
# cached embedding stage 1 on MTEB retrieval sets. numpy + pyarrow are not in the
# apt set; it re-execs itself through uv on first use (a few seconds, needs net
# once), like benchmark's PyYAML fallback.
COPY --chmod=0755 scripts/rerank-bench /usr/local/bin/rerank-bench

# Fixed Qwen 3.5/3.6/3.8 chat templates for `--chat-template-file`:
#   qwen-fixed.jinja -- froggeric's (reasoning-depth default, enable_thinking=false,
#                       history <think> extraction, tool-call wire format -- see the
#                       model card)
#   qwen-sharp.jinja -- peculiar-ragdoll's Sharp variant: the same template with a
#                       force-appended terseness system prompt (fewer filler tokens,
#                       same kwargs; {"terse": false} in chat_template_kwargs drops it)
# ADD from the URL: BuildKit re-checks the remote file on every build
# (ETag/Last-Modified), so a rebuild picks up a new template version even when
# the layer would otherwise be cached. Paths are stable; the version strings
# are recorded in /versions.txt.
ADD --chmod=0644 ${QWEN_TEMPLATE_URL} /etc/llama-swap/templates/qwen-fixed.jinja
ADD --chmod=0644 ${QWEN_SHARP_TEMPLATE_URL} /etc/llama-swap/templates/qwen-sharp.jinja
# --chmod also applies to the directory ADD creates; make it traversable.
RUN chmod 755 /etc/llama-swap/templates

# Fail the build if any binary or backend library has unresolved shared
# libraries (catches a missing ROCm runtime package or a broken RPATH), and
# smoke-test that each llama-server starts, finds its ggml backends next to
# itself and lists devices (no GPU here, so the list is empty -- the point is
# that backend loading does not fail), and that llama-swap runs and accepts
# the bundled config.
RUN <<'CHECK'
#!/bin/bash
set -euo pipefail
BINS="llama-server llama-cli llama-tts llama-bench whisper-server whisper-cli sd-server sd-cli audiocpp_server audiocpp_cli audiocpp_gguf"
SERVERS="llama-server"
if [ "${WITH_ROCM}" = "true" ]; then
    BINS="$BINS llama-server-rocm llama-cli-rocm llama-tts-rocm llama-bench-rocm whisper-server-rocm whisper-cli-rocm sd-server-rocm sd-cli-rocm"
    SERVERS="$SERVERS llama-server-rocm"
fi
if [ "${WITH_RDNA}" = "true" ]; then
    BINS="$BINS llama-server-rdna llama-cli-rdna llama-bench-rdna llama-quantize-rdna llama-perplexity-rdna"
    SERVERS="$SERVERS llama-server-rdna"
fi
if [ "${WITH_ROCM}" = "true" ] && [ "${WITH_EXL3}" = "true" ]; then
    # Python, so no ldd: the extension must import on the RUNTIME image (its HIP
    # libraries come from the PyTorch wheels, not from the ROCm runtime above).
    (cd / && PYTHONPATH=/opt/exl3/exllamav3 /opt/exl3/venv/bin/python -c "from exllamav3.ext import exllamav3_ext") \
        || { echo "FATAL: exllamav3_ext does not import in the runtime image" >&2; exit 1; }
    exl3-server --help >/dev/null || { echo "FATAL: exl3-server (TabbyAPI) does not start" >&2; exit 1; }
fi
for bin in $BINS; do
    out=$(ldd "$(readlink -f "$(command -v "$bin")")")
    if grep -q 'not found' <<<"$out"; then
        echo "FATAL: $bin has unresolved libraries:" >&2
        grep 'not found' <<<"$out" >&2
        exit 1
    fi
done
for lib in /opt/llama-vulkan/*.so* $([ "${WITH_ROCM}" = "true" ] && echo /opt/llama-rocm/*.so*) $([ -d /opt/llama-rdna ] && echo /opt/llama-rdna/*.so*); do
    if ldd "$lib" | grep -q 'not found'; then
        echo "FATAL: $lib has unresolved libraries" >&2; ldd "$lib" | grep 'not found' >&2; exit 1; fi
done
echo "All binaries and libraries resolve their shared libraries."
for bin in $SERVERS; do
    "$bin" --version
    out=$("$bin" --list-devices 2>&1 || true)
    if ! grep -q "Available devices" <<<"$out"; then
        echo "FATAL: $bin --list-devices did not run (backend libs not found?):" >&2
        echo "$out" >&2; exit 1
    fi
done
# Vulkan feature line is checked at build time in the builder; here only that
# the driver is the PPA one when asked for.
if [ -n "${MESA_PPA}" ]; then
    dpkg-query -W -f '${Version}\n' mesa-vulkan-drivers | grep -q kisak \
        || { echo "FATAL: mesa-vulkan-drivers is not the ${MESA_PPA} build: $(dpkg-query -W -f '${Version}' mesa-vulkan-drivers)" >&2; exit 1; }
fi
llama-swap -version
vllm-wrapper --help >/dev/null 2>&1 || vllm-wrapper -h >/dev/null 2>&1 || true
llama-swap -config /etc/llama-swap/config/config.yaml -validate
uv --version && uvx --version
python3 -c "import yaml" || { echo "FATAL: PyYAML missing" >&2; exit 1; }
benchmark --list --config /etc/llama-swap/config/config.yaml >/dev/null \
    || { echo "FATAL: benchmark --list failed on the bundled config" >&2; exit 1; }
test -d /usr/local/share/audiocpp/model_specs
ls /opt/llama-vulkan/libggml-cpu-*.so | sed 's|.*/libggml-cpu-||; s|\.so||' | tr '\n' ' ' | sed 's/^/cpu variants: /; s/ $/\n/'
CHECK

# /versions.txt: one-line summary per project first, then every stage's full
# build-info (base commit, merged PRs, build options).
RUN <<'VERSIONS'
#!/bin/bash
set -euo pipefail
first() { awk -v k="$1" '$1==k {print $2; exit}' "/tmp/build-info/$2"; }
{
  echo "llama.cpp: $(first llama_vulkan_commit: llama-vulkan)"
  echo "whisper.cpp: $(first whisper_vulkan_commit: whisper-vulkan)"
  echo "stable-diffusion.cpp: $(first sd_vulkan_commit: sd-vulkan)"
  echo "audio.cpp: $(first audiocpp_commit: audiocpp)"
  echo "llama-swap: $(first llama_swap_version: llama-swap)"
  if [ "${WITH_ROCM}" = "true" ]; then echo "backend: vulkan rocm"; else echo "backend: vulkan"; fi
  echo "build_timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "${WITH_ROCM}" = "true" ]; then
    if [ "${ROCM_CHANNEL}" = "classic" ]; then echo "rocm: ${ROCM_VERSION} (classic)"
    else echo "rocm: $(dpkg-query -W -f '${Version}' "amdrocm-runtime${ROCM_SERIES}") (multiarch, series ${ROCM_SERIES})"; fi
    echo "amdgpu_targets: ${AMDGPU_TARGETS}"
  fi
  echo "mesa_vulkan_drivers: $(dpkg-query -W -f '${Version}' mesa-vulkan-drivers) (${MESA_PPA:-ubuntu})"
  echo "cpu_variants: $(ls /opt/llama-vulkan/libggml-cpu-*.so | sed 's|.*/libggml-cpu-||; s|\.so||' | tr '\n' ' ')"
  for f in llama-swap llama-vulkan llama-rocm llama-rdna exl3 whisper-vulkan whisper-rocm sd-vulkan sd-rocm audiocpp; do
    [ -f "/tmp/build-info/$f" ] && cat "/tmp/build-info/$f"
  done
  echo "qwen_chat_template: $(grep -o 'template_version = "[^"]*"' /etc/llama-swap/templates/qwen-fixed.jinja | head -1 | cut -d'"' -f2) (${QWEN_TEMPLATE_URL})"
  echo "qwen_sharp_chat_template: $(grep -o 'template_version = "[^"]*"' /etc/llama-swap/templates/qwen-sharp.jinja | head -1 | cut -d'"' -f2) (${QWEN_SHARP_TEMPLATE_URL})"
} > /versions.txt
rm -rf /tmp/build-info
cat /versions.txt
VERSIONS

# Root (device access without --group-add), /models as the working directory.
# llama-swap is the entrypoint and its defaults live in CMD, so any container
# argument replaces them: `docker run <image> -version`, or
# `docker run <image> -config /models/my.yaml -listen 0.0.0.0:8080 -watch-config`.
WORKDIR /models
USER 0
ENTRYPOINT ["llama-swap"]
CMD ["-config", "/etc/llama-swap/config/config.yaml", "-listen", "0.0.0.0:8080", "-watch-config"]
