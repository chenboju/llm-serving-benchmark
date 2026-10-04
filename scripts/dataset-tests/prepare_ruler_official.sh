#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/LLM_infra}"

RULER_ROOT="${RULER_ROOT:-$WORKSPACE_ROOT/RULER}"
RULER_REVISION="${RULER_REVISION:-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a}"
RULER_WRAPPER_BRANCH="${RULER_WRAPPER_BRANCH:-rulerv1-ns}"
RULER_WRAPPER_REVISION="${RULER_WRAPPER_REVISION:-e8bbff677ca2c239640dc90f93310dcf32408c93}"
NEMO_SKILLS_BRANCH="${NEMO_SKILLS_BRANCH:-chsieh/ruler-remove-prefix}"
NEMO_SKILLS_REVISION="${NEMO_SKILLS_REVISION:-f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3}"
TOKENIZER="${TOKENIZER:-$HOME/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/datasets/raw/ruler-official}"
TMP_DATA_DIR="${TMP_DATA_DIR:-$PROJECT_ROOT/datasets/raw/ruler-work}"
LENGTHS=(${RULER_LENGTHS:-4096 8192 16384 32768 61440})
ACCURACY_SAMPLES_PER_TASK="${RULER_ACCURACY_SAMPLES_PER_TASK:-100}"
PERFORMANCE_60K_SAMPLES_PER_TASK="${RULER_60K_SAMPLES_PER_TASK:-20}"
TASKS=(
    niah_single_1 niah_single_2 niah_single_3
    niah_multikey_1 niah_multikey_2 niah_multikey_3
    niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)

if [[ ! -d "$RULER_ROOT/.git" ]]; then
    echo "Error: clone NVIDIA/RULER at $RULER_ROOT first." >&2
    exit 1
fi

for command_name in git wget; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: command '$command_name' is missing." >&2
        exit 1
    fi
done

actual_ruler_revision="$(git -C "$RULER_ROOT" rev-parse HEAD)"
if [[ "$actual_ruler_revision" != "$RULER_REVISION" ]]; then
    echo "Error: RULER generator revision is $actual_ruler_revision; expected $RULER_REVISION" >&2
    exit 1
fi

if [[ ! -f "$TOKENIZER/tokenizer.json" && ! -f "$TOKENIZER/tokenizer_config.json" ]]; then
    echo "Error: tokenizer snapshot not found: $TOKENIZER" >&2
    exit 1
fi

missing_modules=()
for module in bs4 html2text nltk numpy scipy tqdm transformers wonderwords yaml; do
    if ! python -c "import $module" >/dev/null 2>&1; then
        missing_modules+=("$module")
    fi
done
if (( ${#missing_modules[@]} > 0 )); then
    echo "Error: missing Python modules: ${missing_modules[*]}" >&2
    echo "Install $SCRIPT_DIR/requirements.txt in the active environment:" >&2
    echo "  python -m pip install -r $SCRIPT_DIR/requirements.txt" >&2
    exit 1
fi

json_dir="$RULER_ROOT/scripts/data/synthetic/json"
if [[ ! -f "$json_dir/PaulGrahamEssays.json" ]]; then
    (cd "$json_dir" && python download_paulgraham_essay.py)
fi
if [[ ! -f "$json_dir/hotpotqa.json" || ! -f "$json_dir/squad.json" ]]; then
    (cd "$json_dir" && bash download_qa_dataset.sh)
fi

mkdir -p "$DATA_DIR" "$TMP_DATA_DIR"

for length in "${LENGTHS[@]}"; do
    adjusted_length=$((length - 50))
    samples_per_task="$ACCURACY_SAMPLES_PER_TASK"
    if [[ "$length" -eq 61440 ]]; then
        samples_per_task="$PERFORMANCE_60K_SAMPLES_PER_TASK"
    fi
    setup="gpt-oss-20b-${length}"
    generated="$TMP_DATA_DIR/$setup/generated"
    destination="$DATA_DIR/$length"
    mkdir -p "$generated" "$destination"

    for task in "${TASKS[@]}"; do
        task_output="$generated/$task/test.jsonl"
        if [[ -f "$task_output" ]]; then
            existing_samples="$(awk 'NF { count += 1 } END { print count + 0 }' "$task_output")"
            if [[ "$existing_samples" -eq "$samples_per_task" ]]; then
                echo "Using existing RULER data: task=$task length=$length samples=$existing_samples"
                continue
            fi
            echo "Regenerating RULER data: task=$task length=$length expected=$samples_per_task found=$existing_samples"
        fi
        echo "Preparing official RULERv1: setup=$setup task=$task length=$length"
        (
            cd "$RULER_ROOT/scripts/data"
            python prepare.py \
                --save_dir "$generated" \
                --benchmark synthetic \
                --subset test \
                --task "$task" \
                --model_template_type base \
                --prepare_for_ns \
                --num_samples "$samples_per_task" \
                --max_seq_length "$adjusted_length" \
                --tokenizer_path "$TOKENIZER" \
                --tokenizer_type hf
        )
    done

    python "$SCRIPT_DIR/convert_ruler_for_nemo.py" \
        --source-dir "$generated" \
        --output-dir "$destination" \
        --length "$length" \
        --expected-samples "$samples_per_task"
done

{
    printf 'ruler_generator_revision=%s\n' "$actual_ruler_revision"
    printf 'ruler_wrapper_branch=%s\n' "$RULER_WRAPPER_BRANCH"
    printf 'ruler_wrapper_revision=%s\n' "$RULER_WRAPPER_REVISION"
    printf 'nemo_skills_branch=%s\n' "$NEMO_SKILLS_BRANCH"
    printf 'nemo_skills_revision=%s\n' "$NEMO_SKILLS_REVISION"
    printf 'tokenizer=%s\n' "$TOKENIZER"
    printf 'lengths=%s\n' "${LENGTHS[*]}"
    printf 'accuracy_samples_per_task=%s\n' "$ACCURACY_SAMPLES_PER_TASK"
    printf 'performance_60k_samples_per_task=%s\n' "$PERFORMANCE_60K_SAMPLES_PER_TASK"
    printf 'prepared_at=%s\n' "$(date --iso-8601=seconds)"
} >"$DATA_DIR/official-versions.txt"

python "$SCRIPT_DIR/prepare_datasets.py" ruler \
    --source-dir "$DATA_DIR" \
    --output-dir "$PROJECT_ROOT/datasets/prepared/ruler" \
    --tokenizer "$TOKENIZER"

echo "RULER data: $DATA_DIR"
echo "AIPerf data: $PROJECT_ROOT/datasets/prepared/ruler"
