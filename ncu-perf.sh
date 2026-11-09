ProfileFile="profile/ncu_$(date +%Y%m%d_%H%M%S)"
LaunchSkip=0
#KernelName=gsplatTileAlphaCacheKernel
KernelName=regex:gsplatTile

/usr/local/cuda/bin/ncu \
    --replay-mode application \
    --kernel-name $KernelName \
    --launch-skip $LaunchSkip \
    --launch-count 3 \
    --set basic \
    --section ComputeWorkloadAnalysis \
    --metrics inst_executed \
    -o "$ProfileFile" \
    python ./pygsplat/test/test_cugsplat.py

/usr/local/cuda/bin/ncu-ui "${ProfileFile}.ncu-repz"