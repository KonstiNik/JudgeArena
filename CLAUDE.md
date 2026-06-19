# Claude Context — JudgeArena

Fork of [OpenEuroLLM/JudgeArena](https://github.com/OpenEuroLLM/JudgeArena) at [KonstiNik/JudgeArena](https://github.com/KonstiNik/JudgeArena). Used for ELO evaluation of fine-tuned OLMo-3 checkpoints. Git remotes: `origin` → the fork (our patches live on `konstinik/work`), `upstream` → OpenEuroLLM (pull updates with `git fetch upstream && git merge upstream/main`).

## Local patches (4 files)

### `judgearena/instruction_dataset/__init__.py` — circular import fix

`utils.py` -> `instruction_dataset/__init__.py` -> `m_arenahard.py` -> `utils.data_root` (not yet initialized). Fixed by making the `m_arenahard` and `utils` imports lazy (moved from module-level into `load_instructions()`).

### `judgearena/estimate_elo_ratings.py` — three upstream bugs

1. **`engine_kwargs` not forwarded to judge model**: `judge_extra_kwargs` was built from scratch as `{}`, ignoring `args.engine_kwargs`. The candidate model received `enforce_eager`, `gpu_memory_utilization`, etc., but the judge (the model that actually needs them) did not. Fixed by initializing `judge_extra_kwargs = dict(args.engine_kwargs)`.

2. **`swap_mode="both"` DataFrame crash**: `judge_and_parse_prefs` returns `prefs` of length 2N (original + reversed) but `annotations` of length N. The position arrays (`use_model_a_as_opponent`, `our_model_is_position_a`, `opponent_models`) are also length N. Constructing a DataFrame with mixed lengths crashes. Fixed by duplicating annotations and position arrays for the swapped batch, flipping `our_model_is_position_a` for the reversed entries.

3. **Judge cache key did not include the judge model**: `judge_cache_suffix = f"judge_{cache_suffix}"` reused the candidate-model cache key, so swapping the judge (e.g. Qwen3.5-27B-FP8 → gemma-4-26b-a4b-it) silently replayed the prior judge's verdicts from disk. Symptom: re-runs return in minutes instead of hours, with ELO scores that match the old judge's distribution rather than the new one. Fixed by including the judge model name in the suffix: `f"judge_{replace_slash(args.judge_model)}_{cache_suffix}"`. The candidate-model completion cache (line ~317) is intentionally left judge-independent, since completions don't depend on the judge.

### `judgearena/generate_and_evaluate.py` — result-folder NAME_MAX overflow

Pairwise tasks build the result subdirectory as `f"{task}-{model_A}-{model_B}-{judge}-{swap_mode}-{ts}".replace("/", "_")`. When `model_A` / `model_B` are absolute filesystem paths to local checkpoints (common for fine-tuned models), the resulting single path component exceeds Linux's 255-byte `NAME_MAX` and `mkdir` raises `OSError: [Errno 36] File name too long`. Fixed by taking only the basename of each model id when constructing the folder/file name; full ids are still preserved in `annotations.csv` (model_A/model_B columns) and `args-*.json` for traceability.

### `pyproject.toml` — documentation only

Comments explaining the CUDA/glibc constraints on Leonardo (see below).

## Environment

- **Venv**: `.venv/` managed by `uv sync` (**without** `--extra vllm`) — Python 3.12, langchain, pandas, sklearn, fast_langdetect
- **vLLM + PyTorch**: provided at runtime by shared container `/leonardo_work/OELLM_prod2026/vllm-v0.19.0.sif` (vLLM 0.19.0, PyTorch 2.10+cu129)
- **Do not install vLLM into the venv** — it conflicts with the container's version. `uv sync` (no extras) is intentional.

### Why the container is needed

The prebuilt vLLM 0.10.2 wheel ships CUDA 12.8 kernels, but Leonardo compute nodes have CUDA 12.2 (driver 535). MoE models crash with `_moe_C.topk_softmax` missing. Upgrading to vLLM 0.17+ is not possible either — those wheels require glibc 2.31, but Leonardo (RHEL 8) has glibc 2.28. The shared container was built from source with the correct CUDA, solving both issues.

## Judge model

Current default: `google/gemma-4-26b-a4b-it` — 26B-total MoE with 4B active parameters, 256k context (capped to 32k via `max_model_len`).

Historical:
- `Qwen/Qwen3.5-27B-FP8` — dense FP8, ~29 GiB on A100 64GB, ran with `gpu_memory_utilization=0.90`.
- `Qwen/Qwen3-30B-A3B-Instruct-2507` — original choice, 30B-total MoE with ~3B active params, ~56.9 GiB on A100 64GB, required `enforce_eager=true` + `gpu_memory_utilization=0.98`.

`max_model_len=32768` overrides each model's much larger native context (actual need ~12k tokens).

Eval outputs are namespaced by judge under `outputs/eval/<judge-name>/...` so results from different judges coexist without collision — see `scripts/eval/run_elo_eval.sh` (`JUDGE_NAME=$(basename $JUDGE_MODEL)`).

## Known issues

- **`download_all()` fails on arena-hard datasets**: upstream HF dataset `lmarena-ai/arena-hard-auto` changed its schema. Does not affect ELO evaluation (uses LMArena data). Setup script downloads only the 3 LMArena datasets needed.
- **`estimate_elo_ratings.py` does not save results to files** — prints ELO scores to stdout. SLURM script tees output to `elo_output.log`.

## Eval parameters (matching Fabio's open-instruct setup)

| Parameter | Value |
|---|---|
| `arena` | `LMArena` |
| `swap_mode` | `both` (position bias correction) |
| `max_out_tokens_models` | 8192 |
| `max_out_tokens_judge` | 8192 |
| `truncate_all_input_chars` | 8192 |
| `provide_explanation` | yes |
| `languages` | `en` (expandable to EU set) |
| `enforce_eager` | `true` (memory constraint) |
| `gpu_memory_utilization` | `0.98` |
| `max_model_len` | `32768` |
