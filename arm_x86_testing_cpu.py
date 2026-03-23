#!/usr/bin/env python3
"""PyTorch CPU dispatch latency benchmark for cross-platform comparison.

Usage:
    # 固定线程数运行
    OMP_NUM_THREADS=64 python test_cpu.py

    # 绑核运行（减少 NUMA 干扰）
    numactl --cpunodebind=0 --membind=0 python test_cpu.py

    # 对比两个平台的结果
    python test_cpu.py --compare pytorch_bench_aarch64_*.json pytorch_bench_x86_64_*.json
"""

import argparse
import json
import platform
import time

import torch
import torch.utils.benchmark as benchmark


# ============================================================
# 1. 纯 dispatch 开销（极小 tensor，计算量可忽略）
# ============================================================

def bench_dispatch_overhead():
    """用极小 tensor 隔离 dispatcher 开销，而非计算本身。"""
    results = {}

    shapes = {
        "scalar": (),
        "1x1": (1, 1),
        "2x2": (2, 2),
        "4x4": (4, 4),
    }

    ops = {
        "add": lambda a, b: torch.add(a, b),
        "mul": lambda a, b: torch.mul(a, b),
        "relu": lambda a, _: torch.relu(a),
        "clone": lambda a, _: a.clone(),
        "empty_like": lambda a, _: torch.empty_like(a),
    }

    for shape_name, shape in shapes.items():
        a = torch.randn(shape)
        b = torch.randn(shape)
        for op_name, op_fn in ops.items():
            t = benchmark.Timer(
                stmt="op_fn(a, b)",
                globals={"op_fn": op_fn, "a": a, "b": b},
                label="dispatch_overhead",
                sub_label=f"{op_name}_{shape_name}",
                description="latency",
                num_threads=1,
            )
            m = t.blocked_autorange(min_run_time=1.0)
            results[f"{op_name}_{shape_name}"] = m.median * 1e6  # μs

    return results


# ============================================================
# 2. 线程调度开销（对比单线程 vs 多线程在小任务上的差异）
# ============================================================

def bench_thread_scheduling():
    """小矩阵乘法在不同线程数下的表现，差异反映线程调度开销。"""
    results = {}
    sizes = [8, 32, 64, 256]
    thread_counts = [1, 4, 16, 64]
    max_threads = torch.get_num_threads()

    for n in sizes:
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        for num_threads in thread_counts:
            if num_threads > max_threads * 2:
                continue
            t = benchmark.Timer(
                stmt="torch.mm(a, b)",
                globals={"a": a, "b": b, "torch": torch},
                label="thread_scheduling",
                sub_label=f"mm_{n}x{n}_t{num_threads}",
                num_threads=num_threads,
            )
            m = t.blocked_autorange(min_run_time=1.0)
            results[f"mm_{n}x{n}_t{num_threads}"] = m.median * 1e6

    return results


# ============================================================
# 3. 典型算子端到端性能（含 dispatch + compute）
# ============================================================

def bench_operators():
    """不同规模下典型算子的端到端耗时。"""
    results = {}
    sizes = [64, 256, 1024, 4096]

    for n in sizes:
        a = torch.randn(n, n)
        b = torch.randn(n, n)

        cases = [
            ("matmul", "torch.mm(a, b)", {"a": a, "b": b, "torch": torch}),
            ("add", "torch.add(a, b)", {"a": a, "b": b, "torch": torch}),
            ("relu", "torch.relu(a)", {"a": a, "torch": torch}),
            ("softmax", "torch.softmax(a, dim=1)", {"a": a, "torch": torch}),
            (
                "layernorm",
                "torch.nn.functional.layer_norm(a, [n])",
                {"a": a, "n": n, "torch": torch},
            ),
            (
                "conv1d",
                "torch.nn.functional.conv1d(x, w)",
                {
                    "x": torch.randn(1, 1, n),
                    "w": torch.randn(1, 1, 3),
                    "torch": torch,
                },
            ),
        ]

        for name, stmt, globs in cases:
            t = benchmark.Timer(
                stmt=stmt,
                globals=globs,
                label="operators",
                sub_label=f"{name}_{n}",
            )
            m = t.blocked_autorange(min_run_time=1.0)
            results[f"{name}_{n}"] = m.median * 1e6

    return results


# ============================================================
# 4. Dispatch 路径对比：Python eager vs TorchScript
# ============================================================

def bench_torchscript_vs_eager():
    """TorchScript 跳过 Python dispatcher，差值 ≈ Python 层 dispatch 开销。"""
    results = {}

    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.add(a, b))

    scripted = torch.jit.script(fn)

    for n in [1, 64, 256]:
        a = torch.randn(n, n)
        b = torch.randn(n, n)

        for mode, func in [("eager", fn), ("jit", scripted)]:
            t = benchmark.Timer(
                stmt="func(a, b)",
                globals={"func": func, "a": a, "b": b},
                label="eager_vs_jit",
                sub_label=f"{mode}_{n}x{n}",
            )
            m = t.blocked_autorange(min_run_time=1.0)
            results[f"{mode}_{n}x{n}"] = m.median * 1e6

    return results


# ============================================================
# 环境信息收集
# ============================================================

def collect_env():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_backends": {
            "mkl": (
                torch.backends.mkl.is_available()
                if hasattr(torch.backends, "mkl")
                else False
            ),
            "openmp": (
                torch.backends.openmp.is_available()
                if hasattr(torch.backends, "openmp")
                else False
            ),
            "mkldnn": (
                torch.backends.mkldnn.is_available()
                if hasattr(torch.backends, "mkldnn")
                else False
            ),
        },
    }


# ============================================================
# 结果对比
# ============================================================

def compare(file_a, file_b):
    """对比两个平台的 benchmark JSON 结果。"""
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    print(
        f"{'':40s} {'Platform A':>12s} {'Platform B':>12s} {'Ratio B/A':>10s}"
    )
    print(
        f"{'':40s} {a['env']['machine']:>12s} {b['env']['machine']:>12s}"
    )
    print("=" * 76)

    for section in a["results"]:
        if section not in b["results"]:
            continue
        print(f"\n  {section}")
        print(f"  {'─' * 70}")
        keys = sorted(
            set(a["results"][section]) & set(b["results"][section])
        )
        for k in keys:
            va = a["results"][section][k]
            vb = b["results"][section][k]
            ratio = vb / va if va > 0 else float("inf")
            marker = "◀ B faster" if ratio < 0.9 else ("▶ A faster" if ratio > 1.1 else "≈")
            print(
                f"  {k:40s} {va:10.2f} μs {vb:10.2f} μs {ratio:8.2f}x {marker}"
            )


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch CPU dispatch latency benchmark"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help="Compare two benchmark JSON files instead of running benchmarks",
    )
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    env = collect_env()
    print("=" * 60)
    print(f"Platform:  {env['platform']}")
    print(f"Processor: {env['processor']} ({env['machine']})")
    print(f"PyTorch:   {env['pytorch']}")
    print(f"Threads:   {env['torch_num_threads']} (interop: {env['torch_num_interop_threads']})")
    print(f"Backends:  {env['torch_backends']}")
    print("=" * 60)

    print("\n[1/4] Dispatch overhead (small tensors)...")
    dispatch = bench_dispatch_overhead()

    print("[2/4] Thread scheduling...")
    threading = bench_thread_scheduling()

    print("[3/4] Operator benchmarks...")
    operators = bench_operators()

    print("[4/4] Eager vs TorchScript...")
    jit_cmp = bench_torchscript_vs_eager()

    all_results = {
        "dispatch_overhead_us": dispatch,
        "thread_scheduling_us": threading,
        "operators_us": operators,
        "eager_vs_jit_us": jit_cmp,
    }

    for section, data in all_results.items():
        print(f"\n{'─' * 50}")
        print(f"  {section}")
        print(f"{'─' * 50}")
        for k, v in sorted(data.items()):
            print(f"  {k:40s} {v:10.2f} μs")

    output = {"env": env, "results": all_results}
    fname = f"pytorch_bench_{env['machine']}_{int(time.time())}.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")


if __name__ == "__main__":
    main()
