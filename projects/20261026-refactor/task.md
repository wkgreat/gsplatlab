# 描述
代码重构
原始代码在 demos\gsplatrender，现在代码已经抽取代码至根目录下的cugsplat和pygsplat两个文件夹下
- cugsplat 3dgs的c++及cuda实现  
- pygplst cugsplat的python绑定  
代码已经拷贝过来，但是cmake构建及代码依赖还没配置好

# 任务：
- 补全cmake
  - cugsplat编译成动态链接库
- 修改setup.py, python引用cugsplat的动态连接库，python模块重名成pygsplat
- 修改pygsplat\gsplat_rasterize_test.py import 新的 pygsplat

# 目标：
- 使用conda的gsplatlab环境
- cugsplat编译成功
- pygsplat\gsplat_rasterize_test.py 运行成功，其输出与基线 pygsplat\baseline_20261017.png 结果一致

