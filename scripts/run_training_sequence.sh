#!/usr/bin/env bash
# =============================================================================
# MedSAM-PCa 顺序训练脚本（单卡串行）
#
# ⚠️  Rule 1（PROJECT_CONSTRAINTS.md）：正式训练必须由【用户本人】启动。
#     本脚本只提供便利的串行封装，AI 不会执行它。
#
# 用法
#   bash scripts/run_training_sequence.sh                # 默认串行训练 E2 -> E3
#   bash scripts/run_training_sequence.sh e2_unetr e3_multilevel_fpn
#   bash scripts/run_training_sequence.sh --dry-run e2_unetr e3_multilevel_fpn
#   bash scripts/run_training_sequence.sh --force e2_unetr e3_multilevel_fpn
#
# 行为
#   * 严格串行：前一个模型训练成功结束后，才启动下一个
#   * 前一个失败 -> 立即停止，不再启动后续模型（避免浪费 GPU 时间）
#   * 已完成（metrics.csv 行数 >= 配置 epochs）的模型默认跳过，避免误覆盖
#   * 启动前检查 GPU 空闲显存，不足则拒绝开始
#   * 每个模型的终端输出同时落盘到 outputs/runs/<model>/console.log
#   * 全部结束后打印 best positive-slice Dice 汇总
#
# 推荐配合 tmux 使用：
#   tmux new -s medsam_seq
#   bash scripts/run_training_sequence.sh
#   Ctrl+B 然后 D 脱离
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

TRAIN_PYTHON="${TRAIN_PYTHON:-/root/anaconda3/envs/lm/bin/python}"
GPU_ID="${GPU_ID:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
DEFAULT_MODELS=(e2_unetr e3_multilevel_fpn)

FORCE=0
DRY_RUN=0
MODELS=()

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)          FORCE=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    -*)               echo "未知选项: $1" >&2; usage; exit 2 ;;
    *)                MODELS+=("$1"); shift ;;
  esac
done
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("${DEFAULT_MODELS[@]}")

# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
log()  { printf '\n\033[1;36m[%s] %s\033[0m\n' "$(date '+%F %T')" "$*"; }
warn() { printf '\n\033[1;33m[WARN] %s\033[0m\n' "$*" >&2; }
fail() { printf '\n\033[1;31m[FAIL] %s\033[0m\n' "$*" >&2; }

cd "$PROJECT_ROOT" || { fail "无法进入项目根目录 $PROJECT_ROOT"; exit 1; }

# 配置中的 epochs
config_epochs() {
  "$TRAIN_PYTHON" - "$1" <<'PY' 2>/dev/null
import sys, yaml
print(yaml.safe_load(open(f"configs/{sys.argv[1]}.yaml", encoding="utf-8"))["train"]["epochs"])
PY
}

# 是否已完成：metrics.csv 行数（不含表头）>= 配置 epochs
is_completed() {
  local model="$1"
  local csv="outputs/runs/$model/metrics.csv"
  local want have
  [[ -f "$csv" ]] || return 1
  want="$(config_epochs "$model")"
  [[ -n "$want" ]] || return 1
  have=$(( $(wc -l < "$csv") - 1 ))
  [[ "$have" -ge "$want" ]]
}

# GPU 空闲显存（MiB）
gpu_free_mib() {
  "$TRAIN_PYTHON" - <<'PY' 2>/dev/null || echo 0
import torch
if torch.cuda.is_available():
    free, _ = torch.cuda.mem_get_info(0)
    print(int(free // 1024 ** 2))
else:
    print(0)
PY
}

# 汇总（读 metrics.csv）
summarize() {
  local model="$1"
  local csv="outputs/runs/$model/metrics.csv"
  [[ -f "$csv" ]] || { echo "  $model: 无 metrics.csv"; return; }
  "$TRAIN_PYTHON" - "$csv" "$model" <<'PY' 2>/dev/null
import sys, pandas as pd
csv, model = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv)
col = "val_positive_dice"
if col not in df.columns or df[col].dropna().empty:
    print(f"  {model}: 无 validation 记录")
else:
    v = df.dropna(subset=[col])
    i = v[col].idxmax()
    r = v.loc[i]
    last = v.iloc[-1]
    print(f"  {model:<22} best pos-Dice={r[col]:.4f} @epoch {int(r['epoch']):<3} "
          f"| final={last[col]:.4f} | epochs={len(df)}")
PY
}

# --------------------------------------------------------------------------- #
# 前置检查
# --------------------------------------------------------------------------- #
log "项目根目录: $PROJECT_ROOT"
log "Python: $TRAIN_PYTHON"
log "待训练序列: ${MODELS[*]}"
[[ $FORCE -eq 1 ]] && warn "--force：将忽略「已完成」检查，可能覆盖既有结果"

for m in "${MODELS[@]}"; do
  [[ -f "configs/$m.yaml" ]] || { fail "配置不存在: configs/$m.yaml"; exit 1; }
done

if pgrep -f "train\.py --config" >/dev/null 2>&1; then
  fail "检测到已有 train.py --config 进程正在运行："
  pgrep -af "train\.py --config" | sed 's/^/    /'
  fail "单卡服务器一次只应运行一个训练，请先等待其结束。"
  exit 1
fi

FREE_MIB="$(gpu_free_mib)"
log "GPU 空闲显存: ${FREE_MIB} MiB (需要 >= ${MIN_FREE_MIB} MiB)"
if [[ "${FREE_MIB:-0}" -lt "$MIN_FREE_MIB" ]]; then
  fail "GPU 空闲显存不足（可能是别的任务在占用），拒绝开始。"
  exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
  log "--dry-run：仅列出将要执行的命令"
  for m in "${MODELS[@]}"; do
    if [[ $FORCE -eq 0 ]] && is_completed "$m"; then
      echo "  [SKIP] $m（已完成）"
    else
      echo "  [RUN ] CUDA_VISIBLE_DEVICES=$GPU_ID $TRAIN_PYTHON train.py --config configs/$m.yaml"
    fi
  done
  exit 0
fi

# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #
declare -a RAN=() SKIPPED=()
OVERALL_START=$(date +%s)

for m in "${MODELS[@]}"; do
  if [[ $FORCE -eq 0 ]] && is_completed "$m"; then
    log "[$m] 已完成（metrics.csv 达标），跳过。如需重跑请加 --force"
    SKIPPED+=("$m")
    continue
  fi

  console="outputs/runs/$m/console.log"
  mkdir -p "outputs/runs/$m"

  log "[$m] 开始训练  $(date '+%F %T')"
  log "[$m] 命令: CUDA_VISIBLE_DEVICES=$GPU_ID $TRAIN_PYTHON train.py --config configs/$m.yaml"
  log "[$m] 终端输出同时写入: $console"

  start=$(date +%s)
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$TRAIN_PYTHON" train.py --config "configs/$m.yaml" 2>&1 \
    | tee -a "$console"
  rc=${PIPESTATUS[0]}
  dur=$(( $(date +%s) - start ))

  if [[ $rc -ne 0 ]]; then
    fail "[$m] 训练失败 (exit=$rc)，耗时 ${dur}s。停止后续训练。"
    fail "请检查 outputs/runs/$m/train.log 与 console.log"
    summarize "$m"
    exit "$rc"
  fi

  log "[$m] 完成，耗时 $(( dur / 60 )) 分 $(( dur % 60 )) 秒"
  summarize "$m"
  RAN+=("$m")
done

TOTAL=$(( $(date +%s) - OVERALL_START ))
log "全部结束，总耗时 $(( TOTAL / 3600 )) 小时 $(( (TOTAL % 3600) / 60 )) 分"

echo
echo "================ 结果汇总 ================"
for m in "${MODELS[@]}"; do summarize "$m"; done
echo "========================================="
[[ ${#SKIPPED[@]} -gt 0 ]] && echo "跳过（已完成）: ${SKIPPED[*]}"
[[ ${#RAN[@]}     -gt 0 ]] && echo "本次运行      : ${RAN[*]}"
echo
echo "下一步（由你本人执行）—— 评估 best checkpoint："
for m in "${MODELS[@]}"; do
  echo "  CUDA_VISIBLE_DEVICES=$GPU_ID $TRAIN_PYTHON evaluate.py --config configs/$m.yaml --checkpoint outputs/runs/$m/checkpoint_best.pth"
done
