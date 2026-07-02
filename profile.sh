#!/usr/bin/env bash
# filepath: profile.sh
#
# NVIDIA profiling helper for the GPU integrator. Runs the integrator under
# Nsight Systems (nsys, timeline) or Nsight Compute (ncu, per-kernel metrics)
# and writes a timestamped report into profiling/. Defaults to the integration
# test data so it runs with no arguments.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values, each overridable by env var or CLI flag
DATA="${DATA:-/scratch/ffs_integrate_test_data}"
REFLECTION="${REFLECTION:-$DATA/predicted.refl}"
EXPERIMENT="${EXPERIMENT:-$DATA/indexed.expt}"
INTEGRATOR="${INTEGRATOR:-build/bin/integrator}"
OUTPUT_DIR="${OUTPUT_DIR:-profiling}"

# Fallback location for the profilers when they are not on PATH
CUDA_BIN="/dls_sw/apps/cuda/12.6.3/bin"

# Default integrator arguments, used when nothing is passed after --
DEFAULT_INTEGRATOR_ARGS=(-a dials --sigma_b 0.03 --sigma_m 0.1 --background constant)

# ncu defaults
NCU_LAUNCH_COUNT=20
NCU_SET=detailed
NCU_KERNEL=""

show_help() {
    cat << EOF
Usage: $0 <subcommand> [OPTIONS] [-- <extra integrator args>]

NVIDIA profiling helper for the GPU integrator. Produces Nsight report files
under '${OUTPUT_DIR}/' that can be opened in nsys-ui / ncu-ui.

SUBCOMMANDS:
    nsys                   Nsight Systems capture (timeline / overview).
    ncu                    Nsight Compute capture (per-kernel metrics).
    help                   Show this help message.

COMMON OPTIONS:
    -r, --reflection FILE  Input reflection table (default: \$DATA/predicted.refl).
    -e, --experiment FILE  Input experiment list (default: \$DATA/indexed.expt).
    -o, --output-dir DIR   Directory for report files (default: ${OUTPUT_DIR}).
    -h, --help             Show this help message.

NCU OPTIONS:
    --set NAME             Metric section set: basic, detailed or full (default: ${NCU_SET}).
                           'detailed'/'full' populate every ncu-ui chart but are slower.
    --detailed             Shortcut for --set detailed.
    --full                 Shortcut for --set full.
    --launch-count N       Cap the number of profiled kernel launches (default: ${NCU_LAUNCH_COUNT}).
                           Use 0 to profile every launch.
    --kernel REGEX         Only profile kernels whose (demangled) name matches REGEX.

PASS-THROUGH:
    Anything after '--' is forwarded verbatim to the integrator. When provided,
    it replaces the default integrator arguments
    (${DEFAULT_INTEGRATOR_ARGS[*]}).

ENVIRONMENT:
    DATA, REFLECTION, EXPERIMENT, INTEGRATOR, OUTPUT_DIR may override the defaults.

EXAMPLES:
    $0 nsys                                   # Timeline on the default test data.
    $0 ncu                                    # All ncu-ui charts, capped at ${NCU_LAUNCH_COUNT} launches.
    $0 ncu --set basic                        # Fewer sections, faster capture.
    $0 ncu --full --launch-count 0            # Full metric set on every launch (slow).
    $0 ncu --kernel kabsch                    # Focus on the Kabsch kernel(s).
    $0 nsys -- -a ellipsoid --background glm  # Override integrator args.
EOF
}

# Resolve a profiler binary, preferring one on PATH over the fallback install
resolve_tool() {
    local tool="$1"
    if command -v "$tool" >/dev/null 2>&1; then
        command -v "$tool"
    elif [[ -x "$CUDA_BIN/$tool" ]]; then
        echo "$CUDA_BIN/$tool"
    else
        return 1
    fi
}

# Check the integrator binary and input files exist before profiling
check_inputs() {
    if [[ ! -x "$INTEGRATOR" ]]; then
        print_error "Integrator binary not found or not executable: $INTEGRATOR"
        print_error "Build it first with ./build.sh, or set INTEGRATOR to its path."
        exit 1
    fi
    if [[ ! -f "$REFLECTION" ]]; then
        print_error "Reflection file not found: $REFLECTION"
        exit 1
    fi
    if [[ ! -f "$EXPERIMENT" ]]; then
        print_error "Experiment file not found: $EXPERIMENT"
        exit 1
    fi
}

# Parse options. Anything after -- lands in EXTRA_ARGS for the integrator.
EXTRA_ARGS=()
parse_args() {
    local subcommand="$1"
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -r|--reflection)
                REFLECTION="$2"; shift 2 ;;
            -e|--experiment)
                EXPERIMENT="$2"; shift 2 ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"; shift 2 ;;
            --launch-count)
                NCU_LAUNCH_COUNT="$2"; shift 2 ;;
            --kernel)
                NCU_KERNEL="$2"; shift 2 ;;
            --set)
                NCU_SET="$2"; shift 2 ;;
            --detailed)
                NCU_SET=detailed; shift ;;
            --full)
                NCU_SET=full; shift ;;
            -h|--help)
                show_help; exit 0 ;;
            --)
                shift
                EXTRA_ARGS=("$@")
                break ;;
            *)
                print_error "Unknown option for '$subcommand': $1"
                show_help
                exit 1 ;;
        esac
    done
}

# Build the integrator command line: binary, inputs, then pass-through or default args
build_integrator_cmd() {
    INTEGRATOR_CMD=("$INTEGRATOR" -r "$REFLECTION" -e "$EXPERIMENT")
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        INTEGRATOR_CMD+=("${EXTRA_ARGS[@]}")
    else
        INTEGRATOR_CMD+=("${DEFAULT_INTEGRATOR_ARGS[@]}")
    fi
}

run_nsys() {
    parse_args nsys "$@"
    check_inputs

    local nsys
    if ! nsys=$(resolve_tool nsys); then
        print_error "nsys not found on PATH or in $CUDA_BIN"
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"
    local report="$OUTPUT_DIR/integrator_nsys_$(date +%Y%m%d_%H%M%S)"

    build_integrator_cmd
    print_status "Using nsys: $nsys"
    print_status "Profiling: ${INTEGRATOR_CMD[*]}"

    "$nsys" profile \
        -t cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --stats=true \
        -f true \
        -o "$report" \
        "${INTEGRATOR_CMD[@]}"

    print_success "Nsight Systems report written to ${report}.nsys-rep"
    print_status "Open with: nsys-ui ${report}.nsys-rep"
}

run_ncu() {
    parse_args ncu "$@"
    check_inputs

    local ncu
    if ! ncu=$(resolve_tool ncu); then
        print_error "ncu not found on PATH or in $CUDA_BIN"
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"
    local report="$OUTPUT_DIR/integrator_ncu_$(date +%Y%m%d_%H%M%S)"

    # The section set controls how much detail is collected per kernel. The
    # launch cap controls how many kernel launches get profiled.
    local ncu_opts=(--kernel-name-base demangled --set "$NCU_SET" -f -o "$report")
    print_status "Collecting '$NCU_SET' metric set."
    if [[ "$NCU_LAUNCH_COUNT" -gt 0 ]]; then
        ncu_opts+=(--launch-count "$NCU_LAUNCH_COUNT")
        print_status "Capped at $NCU_LAUNCH_COUNT kernel launches."
    else
        print_status "Profiling every kernel launch (this may take a long time)."
    fi
    if [[ -n "$NCU_KERNEL" ]]; then
        ncu_opts+=(-k "$NCU_KERNEL")
        print_status "Restricting to kernels matching: $NCU_KERNEL"
    fi

    build_integrator_cmd
    print_status "Using ncu: $ncu"
    print_status "Profiling: ${INTEGRATOR_CMD[*]}"

    "$ncu" "${ncu_opts[@]}" "${INTEGRATOR_CMD[@]}"

    print_success "Nsight Compute report written to ${report}.ncu-rep"
    print_status "Open with: ncu-ui ${report}.ncu-rep"
}

main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 1
    fi

    local subcommand="$1"
    shift
    case "$subcommand" in
        nsys)
            run_nsys "$@" ;;
        ncu)
            run_ncu "$@" ;;
        help|-h|--help)
            show_help ;;
        *)
            print_error "Unknown subcommand: $subcommand"
            show_help
            exit 1 ;;
    esac
}

main "$@"
