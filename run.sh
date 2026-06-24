#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
seed_list="2026"  
# seed_list="2023 42"  
# seed_list="2026 2023 42"  

seg_list="1 3 6 12"
ablation_seg_list="1 3 6 12"

pred_lens=(
  "12"
  "24"
  "48"
  "96"
  "192"
  "336"
  "720"
)

models=(
  "DRFormer"
#   "iTransformer"
#   "S-Mamba"
#   "Autoformer"
#   "PatchTST"
)

datasets=(
  "ECL"
  "Traffic"
  "PEMS/03"
#   "PEMS/04"
#   "PEMS/07"
#   "PEMS/08"
  "SolarEnergy"
#   "ETT/ETTh1"
  "ETT/ETTh2"
#   "ETT/ETTm1"
#   "ETT/ETTm2"
  "Exchange"
#   "Weather"
)

RUN_MULTIVARIATE_FORECASTING=${RUN_MULTIVARIATE_FORECASTING:-false}
RUN_ABLATION=${RUN_ABLATION:-false}
RUN_SEG_NUM=${RUN_SEG_NUM:-true}

DRY_RUN=${DRY_RUN:-false}
RESUME_AFTER_SCRIPT=${RESUME_AFTER_SCRIPT:-}

seg_ablations=(
  "TIP"
  "TIP+DIP"
)

resume_ready=true
if [ -n "$RESUME_AFTER_SCRIPT" ]; then
    resume_ready=false
    echo "Resuming after: $RESUME_AFTER_SCRIPT"
fi

get_multivariate_script() {
    local model="$1"
    local dataset="$2"
    local group="${dataset%%/*}"
    local suffix="${dataset#*/}"

    if [ "$dataset" = "$group" ]; then
        echo "./scripts/multivariate_forecasting/${group}/${model}.sh"
    else
        echo "./scripts/multivariate_forecasting/${group}/${model}_${suffix}.sh"
    fi
}

get_seg_ablation_script() {
    local seg="$1"
    local ablation="$2"
    local dataset="$3"
    local group="${dataset%%/*}"
    local suffix="${dataset#*/}"
    local model

    case "$ablation" in
        TIP)
            if [ "$seg" = "1" ]; then
                model="iTransformer"
            else
                model="DSW_iTransformer"
            fi
            ;;
        TIP+DIP)
            model="DRFormer"
            ;;
        *)
            echo "Unknown segment ablation: $ablation" >&2
            exit 1
            ;;
    esac

    if [ "$dataset" = "$group" ]; then
        echo "./scripts/ablation_study/${seg}-DSW+${ablation}/${group}/${model}.sh"
    else
        echo "./scripts/ablation_study/${seg}-DSW+${ablation}/${group}/${model}_${suffix}.sh"
    fi
}

get_seg_num_script() {
    local seg="$1"
    local dataset="$2"
    local group="${dataset%%/*}"
    local suffix="${dataset#*/}"
    local model="DRFormer"

    if [ "$dataset" = "$group" ]; then
        echo "./scripts/seg_num/${seg}/${group}/${model}.sh"
    else
        echo "./scripts/seg_num/${seg}/${group}/${model}_${suffix}.sh"
    fi
}

should_skip_until_resume() {
    local script_path="$1"

    if [ -z "$RESUME_AFTER_SCRIPT" ]; then
        return 1
    fi

    if [ "$resume_ready" = false ]; then
        if [ "$script_path" = "$RESUME_AFTER_SCRIPT" ]; then
            resume_ready=true
            echo "Resume point reached after: $script_path"
        fi
        return 0
    fi

    return 1
}

run_script_with_pred_lens() {
    local script_path="$1"
    local tmp_script
    local pred_len
    local pred_len_values=" "

    for pred_len in "${pred_lens[@]}"; do
        pred_len_values="${pred_len_values}${pred_len} "
    done

    if [[ "$pred_len_values" == *" all "* ]]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Dry run OK: $script_path (pred_len: all)"
            return 0
        fi
        bash "$script_path"
        return $?
    fi

    tmp_script=$(mktemp) || exit 1

    awk -v pred_len_values="$pred_len_values" '
        function line_continues(line) {
            return line ~ /\\[[:space:]]*$/
        }

        function block_pred_len(block,   n, lines, i, line) {
            n = split(block, lines, "\n")
            for (i = 1; i <= n; i++) {
                line = lines[i]
                sub(/#.*/, "", line)
                if (line ~ /--pred_len[[:space:]]+[0-9]+/) {
                    sub(/^.*--pred_len[[:space:]]+/, "", line)
                    sub(/[[:space:]\\].*$/, "", line)
                    return line
                }
            }
            return ""
        }

        function flush_block(   pred_len) {
            if (!in_block) {
                return
            }

            pred_len = block_pred_len(block)
            if (index(pred_len_values, " " pred_len " ") > 0) {
                print block
                printed_blocks++
            }

            block = ""
            in_block = 0
        }

        /^[[:space:]]*python[[:space:]]+-u[[:space:]]+run\.py/ {
            flush_block()
            in_block = 1
            block = $0 ORS
            if (!line_continues($0)) {
                flush_block()
            }
            next
        }

        {
            if (in_block) {
                block = block $0 ORS
                if (!line_continues($0)) {
                    flush_block()
                }
            } else {
                print
            }
        }

        END {
            flush_block()
            if (printed_blocks == 0) {
                exit 2
            }
        }
    ' "$script_path" > "$tmp_script"

    local awk_status=$?
    if [ "$awk_status" -eq 2 ]; then
        echo "Skipping: $script_path (no matching pred_len in: ${pred_lens[*]})"
        rm -f "$tmp_script"
        return 0
    elif [ "$awk_status" -ne 0 ]; then
        echo "Failed to filter script: $script_path" >&2
        rm -f "$tmp_script"
        exit "$awk_status"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "Dry run OK: $script_path (pred_len: ${pred_lens[*]})"
        rm -f "$tmp_script"
        return 0
    fi

    bash "$tmp_script"
    local run_status=$?
    rm -f "$tmp_script"

    if [ "$run_status" -ne 0 ]; then
        echo "Failed: $script_path (pred_len: ${pred_lens[*]})" >&2
        exit "$run_status"
    fi
}

for fix_seed in $seed_list; do
    export FIX_SEED=$fix_seed

    echo "======================================"
    echo "FIX_SEED = $FIX_SEED"
    echo "pred_lens = ${pred_lens[*]}"
    echo "RUN_MULTIVARIATE_FORECASTING = $RUN_MULTIVARIATE_FORECASTING"
    echo "RUN_ABLATION = $RUN_ABLATION"
    echo "RUN_SEG_NUM = $RUN_SEG_NUM"
    echo "seg_list = $seg_list"
    echo "ablation_seg_list = $ablation_seg_list"
    echo "DRY_RUN = $DRY_RUN"
    echo "======================================"

    # 1. Multivariate Forecasting
    if [ "$RUN_MULTIVARIATE_FORECASTING" = true ]; then
        echo ">>>> Running Multivariate Forecasting"

        for model in "${models[@]}"; do
            echo "---- model = $model ----"

            for ds in "${datasets[@]}"; do
                script_path=$(get_multivariate_script "$model" "$ds")
                if should_skip_until_resume "$script_path"; then
                    continue
                fi
                if [ ! -f "$script_path" ]; then
                    echo "Missing script: $script_path" >&2
                    exit 1
                fi
                echo "Running: $script_path"
                run_script_with_pred_lens "$script_path"
            done
        done
    fi

    # 2. Segment Ablation
    if [ "$RUN_ABLATION" = true ]; then
        echo ">>>> Running Segment Ablation"

        for seg in $ablation_seg_list; do
            echo "---- seg_num = $seg ----"

            for ablation in "${seg_ablations[@]}"; do
                echo "---- ablation = $ablation ----"

                for ds in "${datasets[@]}"; do
                    script_path=$(get_seg_ablation_script "$seg" "$ablation" "$ds")
                    if should_skip_until_resume "$script_path"; then
                        continue
                    fi
                    if [ ! -f "$script_path" ]; then
                        echo "Missing script: $script_path" >&2
                        exit 1
                    fi
                    echo "Running: $script_path"
                    run_script_with_pred_lens "$script_path"
                done
            done
        done
    fi

    # 3. Segment Number Experiments
    if [ "$RUN_SEG_NUM" = true ]; then
        echo ">>>> Running Segment Number Experiments"

        for seg in $seg_list; do
            echo "---- seg_num = $seg ----"

            for ds in "${datasets[@]}"; do
                script_path=$(get_seg_num_script "$seg" "$ds")
                if should_skip_until_resume "$script_path"; then
                    continue
                fi
                if [ ! -f "$script_path" ]; then
                    echo "Skipping unsupported seg_num script: $script_path"
                    continue
                fi
                echo "Running: $script_path"
                run_script_with_pred_lens "$script_path"
            done
        done
    fi
done

echo "All selected experiments finished."
