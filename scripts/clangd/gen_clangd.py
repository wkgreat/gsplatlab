"""按当前机器 + 当前环境生成 clangd / VSCode 配置。

生成（覆盖写，内容相同则跳过）：
  - demos/gsplatrender/.clangd
  - .vscode/settings.json

机器相关路径全部在运行时推导：
  - python / numpy 头文件  -> 当前解释器（pixi env 内的 python）
  - CUDA                   -> CUDA_HOME / CUDA_PATH，找不到再按平台默认值兜底

通常在 pixi 激活时自动运行；也可手动执行 `pixi run clangd-config`。
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = Path(__file__).resolve().parent

CLANGD_OUT = REPO_ROOT / "demos" / "gsplatrender" / ".clangd"
VSCODE_OUT = REPO_ROOT / ".vscode" / "settings.json"


def log(msg: str) -> None:
    print(f"[clangd-config] {msg}", flush=True)


def fwd(path: str | Path) -> str:
    """统一成正斜杠，Windows 路径在 clangd/YAML 里更稳。"""
    return str(path).replace("\\", "/")


def python_include_dirs() -> list[Path]:
    prefix = Path(sys.prefix)
    candidates = [
        sysconfig.get_path("include"),
        prefix / "include",
        prefix / "Include",  # Windows conda
        prefix / "Library" / "include",  # Windows conda
    ]
    seen: list[Path] = []
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.is_dir() and p not in seen:
            seen.append(p)
    return seen


def numpy_include_dir() -> Path | None:
    # 从当前环境（pixi env）的 site-packages 直接定位，避免 import numpy 时
    # 被 PYTHONPATH / user site-packages 里同名包抢先。
    purelib = sysconfig.get_path("purelib")
    if purelib:
        for cand in (Path(purelib) / "numpy" / "_core" / "include",):
            if cand.is_dir():
                return cand
    log("警告：当前环境里找不到 numpy 头文件，.clangd 将不含 numpy 路径")
    return None


def cuda_root() -> Path | None:
    for var in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(var)
        if val and (Path(val) / "include").is_dir():
            return Path(val)

    if os.name != "nt":
        default = Path("/usr/local/cuda")
        if (default / "include").is_dir():
            return default
        return None

    # Windows：扫描 NVIDIA 默认安装目录，取版本号最大的一个
    roots = [
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
        Path("C:/Program Files/NVIDIA/CUDA"),
    ]
    found = []
    for root in roots:
        if root.is_dir():
            found.extend(sorted(p for p in root.glob("v*") if (p / "include").is_dir()))
    if found:
        return found[-1]
    return None


def yaml_flag(flag: str) -> str:
    return f'    - "{flag}"'


def render_clangd(cuda: Path | None, cuda_arch: str) -> str:
    header_flags = ["-std=c++17"]
    for inc in python_include_dirs():
        header_flags.append(f"-I{fwd(inc)}")
    numpy_inc = numpy_include_dir()
    if numpy_inc is not None:
        header_flags.append(f"-I{fwd(numpy_inc)}")

    cuda_includes: list[str] = []
    if cuda is not None:
        cuda_includes.append(f"-I{fwd(cuda / 'include')}")
        cccl = cuda / "include" / "cccl"  # CUDA 13 起 CUB/Thrust/libcudacxx 挪到这里
        if cccl.is_dir():
            cuda_includes.append(f"-I{fwd(cccl)}")
    else:
        log("警告：未找到 CUDA（CUDA_HOME/CUDA_PATH 未设置），.clangd 将不含 CUDA 头文件路径")

    template = (TEMPLATE_DIR / "clangd.template").read_text(encoding="utf-8")
    return (
        template.replace(
            "{{HEADER_FLAGS}}",
            "\n".join(yaml_flag(f) for f in header_flags + cuda_includes),
        )
        .replace("{{CUDA_ROOT}}", fwd(cuda) if cuda else "/usr/local/cuda")
        .replace("{{CUDA_ARCH}}", cuda_arch)
    )


def clangd_path_setting() -> str:
    """VSCode 的 clangd.path：优先用激活后 PATH 上的 clangd，转成工作区相对路径。"""
    exe = shutil.which("clangd")
    if exe:
        try:
            rel = Path(exe).resolve().relative_to(REPO_ROOT)
            return "${workspaceFolder}/" + fwd(rel)
        except ValueError:
            return fwd(exe)
    # 兜底：pixi 默认环境的常见位置
    if os.name == "nt":
        return "${workspaceFolder}/.pixi/envs/default/Library/bin/clangd.exe"
    return "${workspaceFolder}/.pixi/envs/default/bin/clangd"


def render_vscode() -> str:
    template = (TEMPLATE_DIR / "vscode_settings.template.json").read_text(
        encoding="utf-8"
    )
    return template.replace("{{CLANGD_PATH}}", clangd_path_setting())


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return True


def main() -> int:
    cuda = cuda_root()
    cuda_arch = os.environ.get("GSPLAT_CUDA_ARCH", "sm_120")

    changed = []
    if write_if_changed(CLANGD_OUT, render_clangd(cuda, cuda_arch)):
        changed.append(CLANGD_OUT)
    if write_if_changed(VSCODE_OUT, render_vscode()):
        changed.append(VSCODE_OUT)

    if changed:
        for path in changed:
            log(f"已更新 {path.relative_to(REPO_ROOT)}")
    else:
        log("无变化")
    return 0


if __name__ == "__main__":
    # 控制台默认可能是 GBK，先切成 utf-8 再输出中文日志
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(main())
