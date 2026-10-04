# llama.cpp benchmark scripts

The llama.cpp half of the micro-benchmark matrix. The
[`../vllm/`](../vllm/) scripts mirror this same workload sequence, which is what
makes the two backends comparable at all.

The server is `llama-server` from a pinned llama.cpp build, serving the
GPT-OSS-20B MXFP4 GGUF behind an OpenAI-compatible API.

## Running

```bash
cd /path/to/llm-serving-benchmark
rtx4090-wsl2/scripts/llama-cpp-tests/01-startup.sh
rtx4090-wsl2/scripts/llama-cpp-tests/02-context-scaling.sh
rtx4090-wsl2/scripts/llama-cpp-tests/03-concurrency-scaling.sh
rtx4090-wsl2/scripts/llama-cpp-tests/04-prefix-cache.sh
rtx4090-wsl2/scripts/llama-cpp-tests/05-interactive-aiperf.sh
rtx4090-wsl2/scripts/llama-cpp-tests/06-rag-aiperf.sh
rtx4090-wsl2/scripts/llama-cpp-tests/07-long-context-aiperf.sh
```

Results are written under `results/llama-cpp/`.

| Script | Workload |
|---|---|
| `01-startup.sh` | Server startup time and readiness |
| `02-context-scaling.sh` | Prefill/decode across context depths |
| `03-concurrency-scaling.sh` | Throughput and latency vs concurrency |
| `04-prefix-cache.sh` | Cold vs warm prefix cache |
| `05-interactive-aiperf.sh` | AIPerf interactive chat workload |
| `06-rag-aiperf.sh` | AIPerf RAG workload |
| `07-long-context-aiperf.sh` | AIPerf long-context workload |
| `aiperf-workload-common.sh` | Shared AIPerf invocation and SLO settings |
| `plot-*.py` | Charts for 02, 03 and 04 |

Scripts 02–04 use llama-benchy; 05–07 use AIPerf. Both are third-party tools —
see [`../../../docs/EXTERNAL_TOOLS.md`](../../../docs/EXTERNAL_TOOLS.md).

## Environment overrides

| Variable | Purpose |
|---|---|
| `LLAMA_CPP_ROOT` | llama.cpp checkout, used to locate `llama-server` |
| `LLAMA_SERVER` | Override the `llama-server` binary path directly |
| `MODEL_FILE` | GGUF path (GPT-OSS-20B MXFP4 by default) |
| `LLAMA_BENCHY_ROOT` | llama-benchy checkout |
| `SERVER_HOST`, `SERVER_PORT` | Bind address |
| `SERVER_PARALLEL` | `--parallel`, the number of server slots |
| `SERVER_READY_TIMEOUT_SECONDS` | How long to wait for readiness before failing |
| `RESULT_DIR_NAME` | Override the output directory name |

## One thing that is not equivalent to vLLM

llama.cpp allocates `--ctx-size` **shared across** `--parallel` slots, so raising
concurrency shrinks the context available per slot. vLLM's `--max-model-len` is a
per-request limit and `--max-num-seqs` bounds scheduler concurrency independently.

The two are therefore not interchangeable knobs, and the comparison here is
between **deployment configurations**, not between engines under identical
settings. This is stated again in
[`../../../docs/RESULTS.md`](../../../docs/RESULTS.md#比較邊界).
