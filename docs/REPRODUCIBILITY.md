# 重現測試

## 1. 參考環境

- Windows + WSL2/Linux shell
- NVIDIA GeForce RTX 4090 24 GB
- GPT-OSS-20B
- OpenAI-compatible chat endpoint
- Python 3.12
- AIPerf 0.12.0，commit `03c9c6ddc5e6227782e53ded177f1227d332af48`
- llama.cpp commit `312cf03328496df9c56fe843952cf02a8964c59a`
- llama-benchy 0.4.1.dev1，commit `e9be344578cec17745066b220798b80a0d2686d3`
- Tokenizer snapshot `6cee5e81ee83917806bbde320786a8fb61efebee`
- Random seed 42、temperature 0

Driver、CUDA、模型 hash 與 Python 套件仍應以每次執行產生的 environment manifest 為準，不要只依賴本文。

## 2. Python 環境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/dataset-tests/requirements.txt
```

正式 runner 需要可用的 AIPerf 環境；AIPerf 與 llama-benchy 的取得方式、授權與固定 commit 見 [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md)。llama.cpp、llama-benchy、模型與 vLLM container 路徑可依腳本中的環境變數覆寫；執行前請先檢查本機路徑。

可選的 benchlab CLI 另需外部專案與 `BENCHLAB_PROJECT` 設定，見 [腳本索引](../scripts/README.md#可選的-benchlab-cli)。正式 dataset campaign 不需要 benchlab。

## 3. 離線驗證

```bash
python -m compileall -q scripts rtx4090-wsl2/analysis rtx4090-wsl2/scripts dgx-spark-cuda/scripts dgx-spark-cuda/analysis
python -m unittest discover -s scripts/dataset-tests -p 'test_*.py'
python scripts/dataset-tests/benchmark_matrix.py
python scripts/dataset-tests/validate_prepared.py
bash scripts/dataset-tests/00-preflight.sh
```

`00-preflight.sh` 會產生環境 manifest，並驗證模型、工具、GPU、revision 與資料完整性。正式執行不應略過此步驟。

## 4. 微基準

```bash
for script in rtx4090-wsl2/scripts/llama-cpp-tests/0{1,2,3,4}-*.sh; do bash "$script"; done
for script in rtx4090-wsl2/scripts/vllm/0{1,2,3,4}-*.sh; do bash "$script"; done
```

執行 AIPerf workload：

```bash
bash rtx4090-wsl2/scripts/llama-cpp-tests/05-interactive-aiperf.sh
bash rtx4090-wsl2/scripts/llama-cpp-tests/06-rag-aiperf.sh
bash rtx4090-wsl2/scripts/llama-cpp-tests/07-long-context-aiperf.sh

bash rtx4090-wsl2/scripts/vllm/05-interactive-aiperf.sh
bash rtx4090-wsl2/scripts/vllm/06-rag-aiperf.sh
bash rtx4090-wsl2/scripts/vllm/07-long-context-aiperf.sh
```

## 5. 正式資料集 campaign

公平比較建議以 suite block 交替執行：

```bash
bash scripts/dataset-tests/run_ab_campaign.sh
bash scripts/dataset-tests/run_vllm_stress_campaign.sh
bash scripts/dataset-tests/run_prefix_campaign.sh
```

完整流程約需 45–55 GPU 小時，且 prefix cache on/off 需要以正確 server mode 分開啟動。詳細控制方式、單格重跑與 RULER accuracy 請見 [`../scripts/dataset-tests/README.md`](../scripts/dataset-tests/README.md)。

## 6. 最小發布資訊

每批結果至少保存：

- Git commit 與是否有未提交變更。
- GPU 型號、VRAM、driver、CUDA 與 WSL/kernel。
- 模型識別、revision、檔案 SHA256、量化與 tokenizer revision。
- Backend/AIPerf/llama-benchy 版本與完整 server/client 命令。
- Dataset revision、SHA256、seed、sampling、warm-up、profile duration。
- 每個 cell 的 repeats、valid/invalid、OOM、error rate、CV 與 SLO 定義。

