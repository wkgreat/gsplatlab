export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda

# 统一 .pyc 输出目录：源码树里不再生成 __pycache__
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$SCRIPT_DIR/.cache/pycache}"

cmake -S . -B build/cugsplat -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cugsplat
conda run -n gsplatlab python -m pip install -e ./pygsplat --no-build-isolation
conda run -n gsplatlab python ./pygsplat/gsplat_rasterize_test.py