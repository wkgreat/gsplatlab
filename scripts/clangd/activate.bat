@echo off
rem pixi 激活时刷新 clangd 配置（Windows）
"%CONDA_PREFIX%\python.exe" "%PIXI_PROJECT_ROOT%\scripts\clangd\gen_clangd.py"
