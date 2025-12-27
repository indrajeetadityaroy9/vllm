#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Optimized build script for vLLM with fault injection on H100
# Targets SM90 only, strips unnecessary backends for faster builds
#
# Usage: ./scripts/build_fault_inject.sh [--clean]
#
# Options:
#   --clean    Force clean build (removes build/ and *.so files)

set -e

# Ensure pip-installed cmake (4.x) takes precedence over system cmake (3.22)
export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$VLLM_ROOT"

# Parse arguments
CLEAN_BUILD=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
    esac
done

echo "=========================================="
echo "vLLM Fault Injection Build Script"
echo "=========================================="
echo "Target: NVIDIA H100 (SM90)"
echo "Root: $VLLM_ROOT"
echo "CMake: $(which cmake) ($(cmake --version | head -1))"
echo ""

# Verify cmake version >= 3.26
CMAKE_VERSION=$(cmake --version | head -1 | grep -oP '\d+\.\d+')
CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d. -f1)
CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d. -f2)
if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 26 ]); then
    echo "ERROR: CMake 3.26+ required, found $CMAKE_VERSION"
    echo "Install with: pip install cmake --upgrade"
    exit 1
fi

# Clean previous build if requested or if build directory doesn't exist
if [ "$CLEAN_BUILD" = true ]; then
    echo "[1/6] Cleaning previous build artifacts..."
    rm -rf build/ vllm/*.so vllm/**/*.so 2>/dev/null || true
    pip uninstall vllm -y 2>/dev/null || true
else
    echo "[1/6] Incremental build (use --clean for full rebuild)"
fi

# H100-optimized build configuration
echo "[2/6] Setting build environment..."
export TORCH_CUDA_ARCH_LIST="9.0"      # SM90 only (H100)
export MAX_JOBS=48                      # Use most of 52 threads
export NVCC_THREADS=2                   # Prevent nvcc memory exhaustion
export CMAKE_BUILD_TYPE=Release

# Strip unnecessary backends (SAFE: uses official flags)
export VLLM_INSTALL_PUNICA_KERNELS=0    # Skip LoRA/Punica kernels

# Enable fault injection
export CMAKE_ARGS="-DVLLM_FAULT_INJECT=ON"

echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  NVCC_THREADS=$NVCC_THREADS"
echo "  CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
echo "  VLLM_INSTALL_PUNICA_KERNELS=$VLLM_INSTALL_PUNICA_KERNELS"
echo "  CMAKE_ARGS=$CMAKE_ARGS"
echo ""

# Build
echo "[3/6] Building vLLM with fault injection..."
BUILD_START=$(date +%s)

pip install -e . --no-build-isolation -v 2>&1 | tee build.log

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

echo ""
echo "[4/6] Build completed in ${BUILD_TIME} seconds"

# Verify fault injection is available
echo "[5/6] Verifying fault injection..."
if python -c "import torch; assert hasattr(torch.ops._C, 'set_fault_injection_config'), 'Fault injection not available'"; then
    echo "  Fault injection: AVAILABLE"
else
    echo "  ERROR: Fault injection not available!"
    echo "  Check build.log for errors"
    exit 1
fi

# Show binary sizes
echo "[6/6] Binary sizes:"
find vllm -name "*.so" -exec ls -lh {} \; 2>/dev/null || echo "  No .so files found in vllm/"

echo ""
echo "=========================================="
echo "Build successful!"
echo "Build time: ${BUILD_TIME} seconds"
echo "Log file: build.log"
echo "=========================================="
