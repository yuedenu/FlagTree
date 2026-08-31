#!/bin/bash
# ==============================================================================
# run_all_tests.sh — 批量验证固定 CommonIR 自定义编译 pipeline
#
# 用法:
#   ./run_all_tests.sh
#   ./run_all_tests.sh native_matmul.py native_fa.py  # 只运行指定文件
#
# 每个用例的 pipeline 已固定写入 Python kernel launch：
#   builtin.module(multi-buffer-pipeline,code-motion)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 结果统计
PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0

FAILED_LIST=()
SPECIFIED_FILES=()

# custom_pipeline 已固定在全部 9 个 Python 用例的 kernel launch 中。
DEFAULT_TEST_FILES=(
    "fa_4func_gpu.py"
    "matmul_add_residual_cv.py"
    "matmul_double_buffer.py"
    "matmul_double_buffer_serial.py"
    "native_fa.py"
    "native_matmul.py"
    "native_matmul_add_residual.py"
    "native_matmul_dsa.py"
    "native_matmul_dsa_slice.py"
)

# ---------- 参数解析 ----------
for arg in "$@"; do
    case "$arg" in
        *.py)
            SPECIFIED_FILES+=("$arg")
            ;;
        *)
            echo "未知参数: $arg"
            exit 1
            ;;
    esac
done

# ---------- 收集测试文件 ----------
collect_test_files() {
    if [ ${#SPECIFIED_FILES[@]} -gt 0 ]; then
        printf '%s\n' "${SPECIFIED_FILES[@]}"
        return
    fi

    printf '%s\n' "${DEFAULT_TEST_FILES[@]}"
}

# ---------- 运行单个测试 ----------
run_one() {
    local py_file="$1"
    local log_file="/tmp/test_${py_file%.py}.log"
    local test_args=()

    if [ "$py_file" = "matmul_double_buffer_serial.py" ]; then
        test_args=(--M 128 --N 256 --K 128 --num-cores 1)
    fi

    TOTAL=$((TOTAL + 1))

    if [ ! -f "$py_file" ]; then
        echo -e "  ${YELLOW}[SKIP]${NC} $py_file (文件不存在)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    printf "  [%2d] %-40s ... " "$TOTAL" "$py_file"

    # 运行测试，超时 300 秒
    if timeout 300 python "$py_file" "${test_args[@]}" > "$log_file" 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASSED=$((PASSED + 1))
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo -e "${YELLOW}TIMEOUT${NC}"
        else
            echo -e "${RED}FAIL${NC} (exit=$exit_code)"
        fi
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$py_file")
        # 打印最后 10 行日志方便排查
        echo -e "       ${CYAN}--- last 10 lines of log ---${NC}"
        tail -10 "$log_file" | sed 's/^/       /'
        echo ""
    fi
}

# ---------- 主流程 ----------
echo "============================================================"
echo " CommonIR fixed custom_pipeline 批量测试"
echo " 目录: $SCRIPT_DIR"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

mapfile -t TEST_FILES < <(collect_test_files)

if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "没有找到可运行的 .py 测试文件。"
    exit 0
fi

echo "共发现 ${#TEST_FILES[@]} 个测试文件，每个文件执行 1 次固定 pipeline:"
echo "  builtin.module(multi-buffer-pipeline,code-motion)"
echo ""

for f in "${TEST_FILES[@]}"; do
    run_one "$f"
done

# ---------- 汇总 ----------
echo ""
echo "============================================================"
echo -e " 结果汇总: 总计=${TOTAL}  ${GREEN}通过=${PASSED}${NC}  ${RED}失败=${FAILED}${NC}  ${YELLOW}跳过=${SKIPPED}${NC}"
echo "============================================================"

if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo ""
    echo -e " ${RED}失败用例:${NC}"
    for f in "${FAILED_LIST[@]}"; do
        echo "   - $f"
    done
    echo ""
    exit 1
fi

exit 0
