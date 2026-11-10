#!/usr/bin/env python3
"""按 benchmark/README.md 的约定执行基准测试，并把结果写入 markdown 报告。

用法:
    python benchmark.py --repleat 10 --report benchmark-report.md \
        python3 pygsplat/test/test_cugsplat.py --data data/Ceramic_Studio.ply --scene data/Ceramic_Studio.scene.json

报告不存在时创建表头，已存在时追加一行。
"""

import argparse
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HEADER = [
    "datetime",
    "os",
    "gpu",
    "cuda",
    "script",
    "data",
    "scene",
    "repeat",
    "total time(ms)",
    "mean time (ms)",
    "median time (ms)",
    "min time(ms)",
    "max time(ms)",
    "90 line",
    "99 line",
]


def run_capture(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def gpu_info():
    out = run_capture(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    lines = out.strip().splitlines()
    if not lines:
        return "unknown"
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 2:
        return parts[0]
    try:
        mem_gb = round(int(float(parts[1])) / 1024)
    except ValueError:
        return parts[0]
    return f"{parts[0]} {mem_gb}GB"


def cuda_version():
    match = re.search(r"release\s+([\d.]+)", run_capture(["nvcc", "--version"]))
    if match:
        return match.group(1)
    match = re.search(r"CUDA Version:\s*([\d.]+)", run_capture(["nvidia-smi"]))
    return match.group(1) if match else "unknown"


def os_name():
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME", platform.platform())
    except (AttributeError, OSError):
        return platform.platform()


def percentile(sorted_values, p):
    """nearest-rank 百分位。"""
    rank = max(1, math.ceil(p / 100.0 * len(sorted_values)))
    return sorted_values[rank - 1]


def arg_value(command, flag):
    if flag in command:
        index = command.index(flag)
        if index + 1 < len(command):
            return os.path.basename(command[index + 1])
    return ""


def write_report(report, row):
    report = Path(report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    exists = report.exists() and report.stat().st_size > 0

    needs_newline = False
    if exists:
        with report.open("rb") as fp:
            fp.seek(-1, os.SEEK_END)
            needs_newline = fp.read(1) != b"\n"

    with report.open("a", encoding="utf-8") as fp:
        if not exists:
            fp.write("| " + " | ".join(HEADER) + " |\n")
            fp.write("| " + " | ".join(["---"] * len(HEADER)) + " |\n")
        elif needs_newline:
            fp.write("\n")
        fp.write("| " + " | ".join(row) + " |\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="执行基准测试并追加结果到 markdown 报告")
    parser.add_argument("--repeat", "--repleat", dest="repeat", type=int, default=10, help="重复执行次数，默认 10")
    parser.add_argument("--interval", type=float, default=2.0, help="每轮之间的间隔秒数，默认 2")
    parser.add_argument("--report", default=str(Path(__file__).resolve().parent / "benchmark-report.md"),
                        help="报告路径，默认 benchmark/benchmark-report.md")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="要执行的命令，例如 python3 xxx.py --data ...")
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("缺少要执行的命令")
    if args.repeat <= 0:
        parser.error("--repeat 必须大于 0")

    times = []
    for index in range(args.repeat):
        if index > 0 and args.interval > 0:
            time.sleep(args.interval)
        start = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            print(f"[benchmark] 第 {index + 1} 次执行失败，退出码 {proc.returncode}", file=sys.stderr)
            return 1
        times.append(elapsed_ms)
        print(f"[benchmark] {index + 1}/{args.repeat}: {elapsed_ms:.2f} ms")

    sorted_times = sorted(times)
    script = next((os.path.basename(tok) for tok in command if tok.endswith(".py")), os.path.basename(command[0]))
    row = [
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        os_name(),
        gpu_info(),
        cuda_version(),
        script,
        arg_value(command, "--data"),
        arg_value(command, "--scene"),
        str(args.repeat),
        f"{sum(times):.2f}",
        f"{sum(times) / len(times):.2f}",
        f"{statistics.median(times):.2f}",
        f"{sorted_times[0]:.2f}",
        f"{sorted_times[-1]:.2f}",
        f"{percentile(sorted_times, 90):.2f}",
        f"{percentile(sorted_times, 99):.2f}",
    ]

    report = write_report(args.report, row)

    width = max(len(name) for name in HEADER)
    print("[benchmark] 测试结果:")
    for name, value in zip(HEADER, row):
        print(f"  {name.ljust(width)} : {value if value else '-'}")
    print(f"[benchmark] 结果已写入 {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
