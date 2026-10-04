#!/usr/bin/env bash
set -uo pipefail

workspace="${LLM_INFRA_ROOT:-$HOME/LLM_infra}"
passes=0
warnings=0
failures=0

pass() {
    printf '[PASS] %s\n' "$1"
    passes=$((passes + 1))
}

warn() {
    printf '[WARN] %s\n' "$1"
    warnings=$((warnings + 1))
}

fail() {
    printf '[FAIL] %s\n' "$1"
    failures=$((failures + 1))
}

section() {
    printf '\n== %s ==\n' "$1"
}

require_command() {
    local command_name="$1"
    if command -v "$command_name" >/dev/null 2>&1; then
        pass "$command_name: $(command -v "$command_name")"
    else
        fail "找不到必要指令：$command_name"
    fi
}

check_version_contains() {
    local label="$1"
    local expected="$2"
    shift 2
    local output
    if ! output=$("$@" 2>&1); then
        fail "$label 執行失敗：$output"
        return
    fi
    if [[ "$output" == *"$expected"* ]]; then
        pass "$label 包含預期值：$expected"
    else
        fail "$label 預期包含 $expected，實際輸出：$output"
    fi
}

check_commit_prefix() {
    local label="$1"
    local directory="$2"
    local expected="$3"
    local actual
    if ! actual=$(git -C "$directory" rev-parse HEAD 2>/dev/null); then
        fail "$label 不是可讀取的 Git checkout：$directory"
        return
    fi
    if [[ "$actual" == "$expected"* ]]; then
        pass "$label commit：$actual"
    else
        fail "$label 預期 commit 前綴為 $expected，實際為 $actual"
    fi
}

printf 'RTX 4090 / WSL2 環境驗證\n'
printf '工作區：%s\n' "$workspace"

section 'WSL 與系統工具'
kernel=$(uname -r)
if [[ "${kernel,,}" == *microsoft* ]]; then
    pass "WSL kernel：$kernel"
else
    fail "目前不是 Microsoft WSL kernel：$kernel"
fi

if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    pass "作業系統：${PRETTY_NAME:-未知}"
else
    fail '無法讀取 /etc/os-release'
fi

for command_name in git gcc g++ cmake ninja ccache curl wget rsync uv nvcc nvidia-smi; do
    require_command "$command_name"
done

section 'NVIDIA GPU 與 CUDA Toolkit'
if gpu_line=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
    --format=csv,noheader,nounits 2>/dev/null); then
    IFS=',' read -r gpu_name driver_version memory_total memory_free <<< "$gpu_line"
    gpu_name="${gpu_name# }"
    driver_version="${driver_version# }"
    memory_total="${memory_total// /}"
    memory_free="${memory_free// /}"
    if [[ "$gpu_name" == *'RTX 4090'* ]]; then
        pass "GPU：$gpu_name"
    else
        fail "預期 GPU 為 RTX 4090，實際為 $gpu_name"
    fi
    if [[ "$memory_total" =~ ^[0-9]+$ ]] && (( memory_total >= 24000 )); then
        pass "VRAM 總量：${memory_total} MiB"
    else
        fail "預期至少 24000 MiB VRAM，實際為 ${memory_total:-未知}"
    fi
    pass "WSL 可見的 Windows NVIDIA driver：$driver_version"
    if [[ "$memory_free" =~ ^[0-9]+$ ]] && (( memory_free >= 20000 )); then
        pass "載入模型前可用 VRAM：${memory_free} MiB"
    else
        warn "可用 VRAM 僅 ${memory_free:-未知} MiB；啟動服務前請找出其他 GPU 程序"
    fi
else
    fail 'nvidia-smi GPU 查詢失敗'
fi

nvcc_path=$(command -v nvcc 2>/dev/null || true)
if [[ "$(readlink -f "$nvcc_path" 2>/dev/null || true)" == '/usr/local/cuda-12.9/bin/nvcc' ]]; then
    pass "nvcc 路徑：$nvcc_path"
else
    fail "nvcc 未指向 CUDA 12.9：${nvcc_path:-找不到}"
fi
check_version_contains 'nvcc' 'release 12.9' nvcc --version

section 'uv managed Python 與原始碼版本'
if managed_python=$(uv python find --managed-python 3.12 2>/dev/null) && [[ -x "$managed_python" ]]; then
    pass "uv managed Python：$managed_python"
else
    fail '找不到 uv managed Python 3.12'
fi

check_commit_prefix 'AIPerf' "$workspace/aiperf" '03c9c6ddc'
check_commit_prefix 'llama-benchy' "$workspace/llama-benchy" 'e9be344'
check_commit_prefix 'llama.cpp' "$workspace/llama.cpp" '312cf03328496df9c56fe843952cf02a8964c59a'

if llama_tag=$(git -C "$workspace/llama.cpp" describe --tags --exact-match 2>/dev/null) \
    && [[ "$llama_tag" == 'b8407' ]]; then
    pass 'llama.cpp tag：b8407'
else
    fail "llama.cpp 並非精確位於 tag b8407：${llama_tag:-未知}"
fi

section 'Benchmark Python 環境'
aiperf_bin="$workspace/aiperf/.venv/bin/aiperf"
benchy_bin="$workspace/llama-benchy/.venv/bin/llama-benchy"
benchlab_bin="$workspace/local-llm-serving-lab/.venv/bin/benchlab"

if [[ -x "$aiperf_bin" ]]; then
    check_version_contains 'AIPerf CLI' '0.12.0' "$aiperf_bin" --version
else
    fail "找不到 AIPerf 執行檔：$aiperf_bin"
fi

if [[ -x "$benchy_bin" ]]; then
    check_version_contains 'llama-benchy CLI' '0.4.1.dev1' "$benchy_bin" --version
else
    fail "找不到 llama-benchy 執行檔：$benchy_bin"
fi

if [[ -x "$benchlab_bin" ]]; then
    check_version_contains 'benchlab CLI' '0.1.0' "$benchlab_bin" --version
else
    fail "找不到 benchlab 執行檔：$benchlab_bin"
fi

for lockfile in "$workspace/aiperf/uv.lock" "$workspace/llama-benchy/uv.lock" \
    "$workspace/local-llm-serving-lab/uv.lock"; do
    if [[ -f "$lockfile" ]]; then
        pass "依賴鎖定檔存在：$lockfile"
    else
        fail "缺少依賴鎖定檔：$lockfile"
    fi
done

section 'llama.cpp CUDA 服務端'
llama_server="$workspace/llama.cpp/build/bin/llama-server"
if [[ -x "$llama_server" ]]; then
    check_version_contains 'llama-server build' '312cf03' "$llama_server" --version
    check_version_contains 'llama-server CUDA device' 'NVIDIA GeForce RTX 4090' \
        "$llama_server" --list-devices
else
    fail "找不到 llama-server：$llama_server"
fi

section 'vLLM Python 與 CUDA 環境'
vllm_root="$workspace/vllm-gpt-oss"
vllm_python="$vllm_root/.venv/bin/python"
vllm_bin="$vllm_root/.venv/bin/vllm"

if [[ -x "$vllm_bin" ]]; then
    check_version_contains 'vLLM CLI' '0.26.0' "$vllm_bin" --version
else
    fail "找不到 vLLM 執行檔：$vllm_bin"
fi

if [[ -x "$vllm_python" ]]; then
    if vllm_info=$("$vllm_python" -c \
        'import torch, vllm; print(vllm.__version__); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())' \
        2>&1); then
        if [[ "$vllm_info" == *$'0.26.0\n'* && "$vllm_info" == *'2.11.0+cu130'* \
            && "$vllm_info" == *$'13.0\nTrue'* ]]; then
            pass 'vLLM 0.26.0 / PyTorch 2.11.0+cu130 / CUDA 可用'
        else
            fail "vLLM/PyTorch 版本組合不符合預期：$vllm_info"
        fi
    else
        fail "無法匯入 vLLM/PyTorch：$vllm_info"
    fi

    if extension_info=$("$vllm_python" -c \
        'import vllm._C_stable_libtorch; print("vLLM stable CUDA extension 正常")' 2>&1); then
        pass "$extension_info"
    else
        fail "vLLM stable CUDA extension 載入失敗：$extension_info"
    fi

    if allocation_info=$("$vllm_python" -c \
        'import torch; x=torch.ones(1, device="cuda"); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), x.item())' \
        2>&1); then
        if [[ "$allocation_info" == *'NVIDIA GeForce RTX 4090'* \
            && "$allocation_info" == *'(8, 9)'* && "$allocation_info" == *'1.0'* ]]; then
            pass "vLLM 環境 GPU allocation：$allocation_info"
        else
            fail "GPU allocation 結果不符合預期：$allocation_info"
        fi
    else
        fail "GPU allocation 失敗：$allocation_info"
    fi
else
    fail "找不到 vLLM Python：$vllm_python"
fi

if [[ -f "$vllm_root/requirements.lock.txt" ]]; then
    pass "vLLM 依賴快照：$vllm_root/requirements.lock.txt"
else
    warn '缺少 vLLM requirements.lock.txt；請執行 uv pip freeze | sort > requirements.lock.txt'
fi

section '服務與模型路徑'
expected_llama_server="$workspace/llama.cpp/build/bin/llama-server"
expected_model_root="$workspace/models"
if [[ "${LLAMA_SERVER_BIN:-}" == "$expected_llama_server" ]]; then
    pass "LLAMA_SERVER_BIN=$LLAMA_SERVER_BIN"
else
    warn "LLAMA_SERVER_BIN 未設定或路徑不符；預期為 $expected_llama_server"
fi
if [[ "${LLAMA_CPP_SOURCE:-}" == "$workspace/llama.cpp" ]]; then
    pass "LLAMA_CPP_SOURCE=$LLAMA_CPP_SOURCE"
else
    warn "LLAMA_CPP_SOURCE 未設定或路徑不符；預期為 $workspace/llama.cpp"
fi
if [[ "${BENCHLAB_MODEL_ROOT:-}" == "$expected_model_root" ]]; then
    pass "BENCHLAB_MODEL_ROOT=$BENCHLAB_MODEL_ROOT"
else
    warn "BENCHLAB_MODEL_ROOT 未設定或路徑不符；預期為 $expected_model_root"
fi

section '驗證摘要'
printf 'PASS=%d WARN=%d FAIL=%d\n' "$passes" "$warnings" "$failures"
printf '通過=%d 警告=%d 失敗=%d\n' "$passes" "$warnings" "$failures"
printf '本腳本未下載或載入模型，也未執行 benchmark。\n'

if (( failures > 0 )); then
    exit 1
fi
