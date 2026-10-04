# 資料集 Manifest

正式 campaign 使用的 prepared 資料檔清單與 SHA256。原始與 prepared 資料集**不隨本
repository 散布**（見 [DATASETS.md](DATASETS.md)）；本表的用途是讓重現者在自行執行
`prepare_datasets.py` 之後，能比對產出是否與本專案所用的完全相同。

> SHA256 不符代表資料不同——通常是 tokenizer revision 或來源 revision 有差異。
> 此時的量測結果**不可**與本專案的數字並列比較。

## 共同的準備條件

| 項目 | 值 |
|---|---|
| Tokenizer snapshot | `6cee5e81ee83917806bbde320786a8fb61efebee`（GPT-OSS-20B HF revision）|
| AIPerf revision | `03c9c6ddc5e6227782e53ded177f1227d332af48` |
| Random seed | 42 |
| Temperature | 0 |

兩平台使用同一份 tokenizer snapshot；正式測試不使用 approximate tokenizer。

## 檔案清單

RTX 4090 的 `formal-split-65k-20260814` 記錄 15 筆，DGX Spark 的
`formal-split-65k-20260816-dgx` 記錄 17 筆，15 筆共同檔案的
SHA256 **全部相同**——這是兩平台結果可直接對比的前提。

| 檔案 | SHA256 | 兩平台 |
|---|---|---|
| `datasets/prepared/prefix/squad-prefix-16384-off.jsonl` | `0b95cfe12ffa57ae6cd008d87193748c48bced4d3a1288c2979ece85f5906665` | 僅 DGX |
| `datasets/prepared/prefix/squad-prefix-4096-off.jsonl` | `17da05d94400dc6cc5562ffd3975d9f2590d6272fedadace46bd086c906a140a` | 僅 DGX |
| `datasets/prepared/rag/hotpotqa-1024.jsonl` | `29090ee3772abcd41eeabd894794446286c5cd2ef76e11ff29c231c81b2638d8` | ✓ 相同 |
| `datasets/prepared/rag/hotpotqa-16384.jsonl` | `5f5f7610bc398f618f19482ea1a404eb9a8d0f6318976e0b27c9ad82ad7ff67c` | ✓ 相同 |
| `datasets/prepared/rag/hotpotqa-4096.jsonl` | `eb4a1b90f3a743a26bf72d79708b5a1c1a9e4829257cd6752c02f5135fead711` | ✓ 相同 |
| `datasets/prepared/rag/hotpotqa-8192.jsonl` | `ed9d4778948db3da955d5d3679f8b6905476fe1fec640140946a2390371b82ed` | ✓ 相同 |
| `datasets/prepared/rag/squad-1024.jsonl` | `bf608364fc2e8a24f4375ebf29c41bb5b6215a742f6de44063a9e61e8818368c` | ✓ 相同 |
| `datasets/prepared/rag/squad-16384.jsonl` | `e680a76c79e2cdc2cc27bf1adf0ea7b37132ef28177b6301181ad01883b37442` | ✓ 相同 |
| `datasets/prepared/rag/squad-4096.jsonl` | `135859c9edc4ce28b4d9519a1f66a252aa8eb1bfbed0aec34893775927c76ffc` | ✓ 相同 |
| `datasets/prepared/rag/squad-8192.jsonl` | `5a5c6c569a48cbf2ba1037ed270adb5171c4647343a1e2599c209f474c20f298` | ✓ 相同 |
| `datasets/prepared/ruler/ruler-16384.jsonl` | `c21162f61261984fc3d4b29c8d8d48b3466bf61ba841c90dc4bf7fec66d736aa` | ✓ 相同 |
| `datasets/prepared/ruler/ruler-32768.jsonl` | `2d93a69199859b065e32ccf3e8a207ed0c903e0318dd6b123100123499e8c39c` | ✓ 相同 |
| `datasets/prepared/ruler/ruler-4096.jsonl` | `dd7fa351bb6ab432449c28bac799748624431aa7def2988b60aefd8d505404f8` | ✓ 相同 |
| `datasets/prepared/ruler/ruler-61440.jsonl` | `8d8c689d7e79353d7d3c39df4463c2fd305b4e5bd93214f8d835c6e3bb46a337` | ✓ 相同 |
| `datasets/prepared/ruler/ruler-8192.jsonl` | `886a725f64a095c7df288bc7fd661922a36b021cc1501d9e44ec1a58bd4523b5` | ✓ 相同 |
| `datasets/prepared/sharegpt/sharegpt-fixed-128.jsonl` | `73f3dee1c18b28af09253d6effb73dfa3b7cdade28eace8dc4b63ebb1c2db33d` | ✓ 相同 |
| `datasets/prepared/sharegpt/sharegpt-natural-osl.jsonl` | `347e50be2a42d794b974622f8f686d313ce2c13d116cca5a5607586f509dc2d1` | ✓ 相同 |

DGX 多出的 2 筆是該平台額外執行的格子所需，4090 側未使用。

## 環境 Manifest

每次 campaign 由 `00-preflight.sh` 產生一份環境 manifest，其 SHA256 記錄在
`campaign.json` 的 `environment_manifest` 欄位：

| 平台 | environment_manifest SHA256 |
|---|---|
| RTX 4090 / WSL2 | `82d6457c3db4fe9c4804566c1b42b9cca24f563c6910f97e1ead31ed1239c36e` |
| DGX Spark | `112fbbdadafe3d983255894c02a75a96ed4c60b859acb5052e626655d22ea10c` |

完整環境資訊（driver、CUDA、kernel、GPU 型號、套件版本）內嵌於各 campaign 的
`campaign.json`；DGX 側收錄於 [`../dgx-spark-cuda/campaigns/`](../dgx-spark-cuda/campaigns/)。

## 部署組態

| 平台 | 組態 |
|---|---|
| RTX 4090 | GPT-OSS-20B HF revision 6cee5e81, max-model-len=65536, max-num-seqs=16; client load up to concurrency=64 |
| DGX Spark | vllm/vllm-openai@sha256:928dde7d6c19478c9c6ab975fed12e6466f6d9b7cb3695d218e287c0c88ad608 (linux/arm64, tag nightly pulled 2026-08-07, vLLM 0.26.1rc1.dev457+gc810e5ee9, build commit c810e5ee9976ad86b81 |

兩邊的 `max-model-len` 皆為 65,536、`max-num-seqs` 皆為 16、客戶端併發上限 64。

## 重新產生

```bash
python scripts/dataset-tests/prepare_datasets.py download
python scripts/dataset-tests/prepare_datasets.py sharegpt --source ... --tokenizer ...
python scripts/dataset-tests/prepare_datasets.py rag      --squad-source ... --tokenizer ...
python scripts/dataset-tests/prepare_datasets.py prefix   --squad-source ... --tokenizer ...
python scripts/dataset-tests/validate_prepared.py
```

完整步驟與參數見 [DATASETS.md](DATASETS.md) 與 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
