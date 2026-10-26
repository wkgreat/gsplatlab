cmake -S . -B build\cugsplat -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build\cugsplat 
conda activate gsplatlab
python -m pip install -e .\pygsplat --no-build-isolation
python .\pygsplat\gsplat_rasterize_test.py