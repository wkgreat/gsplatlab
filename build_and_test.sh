export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
cmake -S . -B build/cugsplat -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cugsplat
conda run -n gsplatlab python -m pip install -e ./pygsplat --no-build-isolation
conda run -n gsplatlab python ./pygsplat/gsplat_rasterize_test.py