
# 编译安装
## windows
```powershell
. "E:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:DISTUTILS_USE_SDK = "1"
$env:CUDA_HOME = "D:\Program Files\NVIDIA\CUDA\v13.2"
$env:CUDA_PATH = "D:\Program Files\NVIDIA\CUDA\v13.2"
$env:Path = "D:\Program Files\NVIDIA\CUDA\v13.2\bin;$env:Path"
python -m pip install -v -e . --no-build-isolation
```

## Linux
```shell
export CUDA_HOME=/usr/local/cuda                                                                                                                                             
export CUDA_PATH=/usr/local/cuda                                                                                                                     
export CUDACXX=/usr/local/cuda/bin/nvcc                                                                                                                         
export PATH=/usr/local/cuda/bin:$PATH
python -m pip install -v -e . --no-build-isolation
```