# vLLM benchmark scripts

These scripts mirror the `llama-cpp-tests` workload matrix while launching the
OpenAI-compatible server with vLLM 0.26.0 conventions.

The default model cache can be supplied through `MODEL_CACHE_ROOT`. For example,
when a Hugging Face cache is stored on Windows:

```text
C:\Users\<user>\.cache\huggingface\hub\models--openai--gpt-oss-20b
```

Inside WSL, set `MODEL_CACHE_ROOT` to the corresponding
`/mnt/c/Users/<user>/.cache/huggingface/hub/` path. The scripts resolve
`refs/main` to the concrete `snapshots/<revision>` directory; the cache root
itself is not passed to `vllm serve`.

Run a workload from WSL:

```bash
cd /path/to/llm-serving-benchmark
scripts/vllm/01-startup.sh
scripts/vllm/02-context-scaling.sh
scripts/vllm/03-concurrency-scaling.sh
scripts/vllm/04-prefix-cache.sh
scripts/vllm/05-interactive-aiperf.sh
scripts/vllm/06-rag-aiperf.sh
scripts/vllm/07-long-context-aiperf.sh
```

Results are stored below `results/vllm/`. Useful environment overrides include
`MODEL_CACHE_ROOT`, `MODEL_PATH`, `SERVER_PORT`, `SERVER_MAX_MODEL_LEN`,
`SERVER_MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, `TENSOR_PARALLEL_SIZE`, and
`SERVER_READY_TIMEOUT_SECONDS`.

For vLLM, `--max-model-len` is a per-request limit and `--max-num-seqs` controls
scheduler concurrency. It is not equivalent to llama.cpp's shared
`--ctx-size`/`--parallel` allocation.
