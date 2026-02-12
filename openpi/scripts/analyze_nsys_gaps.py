#!/usr/bin/env python3
"""
Nsight Systems Gap Analysis Script.

Purpose: 自动化诊断 Denoise 模块的性能瓶颈
1. Kernel Gaps (Launch Latency) - CPU 是否在阻塞 GPU？
2. Memory Throughput - 是否 HBM Bound？
3. Stream Synchronization - 是否有隐式同步？

Usage:
    # 首先导出 nsys 报告为 sqlite
    nsys export --type=sqlite denoise_profile.nsys-rep

    # 然后运行分析
    python scripts/analyze_nsys_gaps.py denoise_profile.sqlite

Diagnostic Criteria:
    - Gap > 50us: SEVERE (CPU Launch Bound)
    - Gap 20-50us: WARNING
    - Gap < 20us: NORMAL (CUDA Graph working)

    - DRAM Bandwidth > 80%: Memory Bound (need L2 Cache Residency)
    - SM Utilization < 30% with high DRAM: Definitely Memory Bound

    - cudaStreamSynchronize present: CRITICAL (implicit sync, must remove)

Author: Turbo-Pi Team
Date: 2026-02-12
"""

import sys
import sqlite3
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class KernelEvent:
    """Represents a CUDA kernel execution."""
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    grid_x: int = 0
    grid_y: int = 0
    grid_z: int = 0
    block_x: int = 0
    block_y: int = 0
    block_z: int = 0
    registers_per_thread: int = 0
    shared_memory: int = 0
    stream_id: int = 0


@dataclass
class NVTXRange:
    """Represents an NVTX range marker."""
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int


@dataclass
class SyncEvent:
    """Represents a CUDA synchronization event."""
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int


@dataclass
class GapAnalysis:
    """Gap analysis result between kernels."""
    from_kernel: str
    to_kernel: str
    gap_us: float
    severity: str  # NORMAL, WARNING, SEVERE


# ============================================================================
# Database Query Functions
# ============================================================================

def query_kernels(conn: sqlite3.Connection) -> List[KernelEvent]:
    """Query all CUDA kernel events."""
    cursor = conn.cursor()

    # Try different table names (nsys schema varies by version)
    tables_to_try = [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUDA_KERNEL_EXEC",
        "GPU_ACTIVITIES",
    ]

    for table in tables_to_try:
        try:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                cursor.execute(f"""
                    SELECT
                        demangledName, start, end, (end - start) as duration,
                        gridX, gridY, gridZ, blockX, blockY, blockZ,
                        registersPerThread, staticSharedMemory, streamId
                    FROM {table}
                    ORDER BY start
                """)
                rows = cursor.fetchall()
                return [
                    KernelEvent(
                        name=row[0] or "unknown",
                        start_ns=row[1],
                        end_ns=row[2],
                        duration_ns=row[3],
                        grid_x=row[4] or 0,
                        grid_y=row[5] or 0,
                        grid_z=row[6] or 0,
                        block_x=row[7] or 0,
                        block_y=row[8] or 0,
                        block_z=row[9] or 0,
                        registers_per_thread=row[10] or 0,
                        shared_memory=row[11] or 0,
                        stream_id=row[12] or 0,
                    )
                    for row in rows
                ]
        except sqlite3.OperationalError:
            continue

    # Fallback: try to find any table with kernel data
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Available tables: {tables}")

    return []


def query_nvtx_ranges(conn: sqlite3.Connection) -> List[NVTXRange]:
    """Query all NVTX range markers."""
    cursor = conn.cursor()

    tables_to_try = [
        "NVTX_EVENTS",
        "NVTX_RANGES",
    ]

    for table in tables_to_try:
        try:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                cursor.execute(f"""
                    SELECT text, start, end, (end - start) as duration
                    FROM {table}
                    WHERE end > start
                    ORDER BY start
                """)
                rows = cursor.fetchall()
                return [
                    NVTXRange(
                        name=row[0] or "unknown",
                        start_ns=row[1],
                        end_ns=row[2],
                        duration_ns=row[3],
                    )
                    for row in rows
                ]
        except sqlite3.OperationalError:
            continue

    return []


def query_sync_events(conn: sqlite3.Connection) -> List[SyncEvent]:
    """Query CUDA synchronization events."""
    cursor = conn.cursor()

    tables_to_try = [
        "CUDA_API",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
    ]

    sync_keywords = ["Synchronize", "sync", "Wait"]

    for table in tables_to_try:
        try:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                # Build OR conditions for sync keywords
                conditions = " OR ".join([f"name LIKE '%{kw}%'" for kw in sync_keywords])
                cursor.execute(f"""
                    SELECT name, start, end, (end - start) as duration
                    FROM {table}
                    WHERE {conditions}
                    ORDER BY start
                """)
                rows = cursor.fetchall()
                return [
                    SyncEvent(
                        name=row[0] or "unknown",
                        start_ns=row[1],
                        end_ns=row[2],
                        duration_ns=row[3],
                    )
                    for row in rows
                ]
        except sqlite3.OperationalError:
            continue

    return []


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_kernel_gaps(kernels: List[KernelEvent]) -> List[GapAnalysis]:
    """Analyze gaps between consecutive kernels."""
    gaps = []

    for i in range(1, len(kernels)):
        prev = kernels[i - 1]
        curr = kernels[i]

        gap_ns = curr.start_ns - prev.end_ns
        gap_us = gap_ns / 1000.0

        # Determine severity
        if gap_us > 50:
            severity = "SEVERE"
        elif gap_us > 20:
            severity = "WARNING"
        else:
            severity = "NORMAL"

        gaps.append(GapAnalysis(
            from_kernel=prev.name[:50],
            to_kernel=curr.name[:50],
            gap_us=gap_us,
            severity=severity,
        ))

    return gaps


def analyze_step_gaps(nvtx_ranges: List[NVTXRange], kernels: List[KernelEvent]) -> Dict:
    """Analyze gaps between denoising steps."""
    # Find step markers
    step_ranges = [r for r in nvtx_ranges if r.name.startswith("Denoise_Step_")]
    step_ranges.sort(key=lambda x: x.start_ns)

    step_gaps = []
    for i in range(1, len(step_ranges)):
        prev = step_ranges[i - 1]
        curr = step_ranges[i]

        # Find last kernel in prev step
        prev_kernels = [k for k in kernels if prev.start_ns <= k.start_ns <= prev.end_ns]
        curr_kernels = [k for k in kernels if curr.start_ns <= k.start_ns <= curr.end_ns]

        if prev_kernels and curr_kernels:
            last_kernel = max(prev_kernels, key=lambda k: k.end_ns)
            first_kernel = min(curr_kernels, key=lambda k: k.start_ns)
            gap_us = (first_kernel.start_ns - last_kernel.end_ns) / 1000.0

            step_gaps.append({
                "from_step": prev.name,
                "to_step": curr.name,
                "gap_us": gap_us,
                "severity": "SEVERE" if gap_us > 50 else ("WARNING" if gap_us > 20 else "NORMAL"),
            })

    return {
        "step_count": len(step_ranges),
        "step_gaps": step_gaps,
        "mean_gap_us": sum(g["gap_us"] for g in step_gaps) / len(step_gaps) if step_gaps else 0,
        "max_gap_us": max(g["gap_us"] for g in step_gaps) if step_gaps else 0,
    }


def analyze_kernel_types(kernels: List[KernelEvent]) -> Dict:
    """Analyze kernel type distribution."""
    kernel_stats = {}

    for k in kernels:
        # Simplify kernel name
        name = k.name
        if "gemm" in name.lower() or "matmul" in name.lower():
            category = "GEMM/MatMul"
        elif "attention" in name.lower() or "softmax" in name.lower():
            category = "Attention"
        elif "elementwise" in name.lower() or "add" in name.lower() or "mul" in name.lower():
            category = "Elementwise"
        elif "reduce" in name.lower() or "sum" in name.lower():
            category = "Reduction"
        elif "copy" in name.lower() or "memcpy" in name.lower():
            category = "Memory Copy"
        else:
            category = "Other"

        if category not in kernel_stats:
            kernel_stats[category] = {"count": 0, "total_us": 0.0, "kernels": []}

        kernel_stats[category]["count"] += 1
        kernel_stats[category]["total_us"] += k.duration_ns / 1000.0
        if len(kernel_stats[category]["kernels"]) < 5:  # Keep sample
            kernel_stats[category]["kernels"].append(name[:60])

    return kernel_stats


def analyze_mlp_kernels(kernels: List[KernelEvent], nvtx_ranges: List[NVTXRange]) -> Dict:
    """Analyze MLP-specific kernel performance."""
    # Find MLP NVTX ranges
    mlp_ranges = [r for r in nvtx_ranges if "/MLP" in r.name]

    mlp_analysis = {
        "mlp_range_count": len(mlp_ranges),
        "mlp_kernels": [],
        "mlp_total_us": 0.0,
        "memory_bound_indicators": [],
    }

    for mlp_range in mlp_ranges:
        mlp_kernels = [
            k for k in kernels
            if mlp_range.start_ns <= k.start_ns <= mlp_range.end_ns
        ]
        for k in mlp_kernels:
            mlp_analysis["mlp_kernels"].append({
                "name": k.name[:60],
                "duration_us": k.duration_ns / 1000.0,
                "grid": f"({k.grid_x}, {k.grid_y}, {k.grid_z})",
                "block": f"({k.block_x}, {k.block_y}, {k.block_z})",
            })
            mlp_analysis["mlp_total_us"] += k.duration_ns / 1000.0

    return mlp_analysis


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    kernels: List[KernelEvent],
    nvtx_ranges: List[NVTXRange],
    sync_events: List[SyncEvent],
    output_path: Optional[str] = None,
) -> str:
    """Generate comprehensive analysis report."""
    lines = []

    lines.append("=" * 80)
    lines.append("     DENOISE MODULE PERFORMANCE DIAGNOSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # ========================================
    # 1. Summary Statistics
    # ========================================
    lines.append("## 1. SUMMARY STATISTICS")
    lines.append("-" * 40)
    lines.append(f"  Total Kernels: {len(kernels)}")
    lines.append(f"  NVTX Ranges: {len(nvtx_ranges)}")
    lines.append(f"  Sync Events: {len(sync_events)}")

    if kernels:
        total_kernel_time = sum(k.duration_ns for k in kernels) / 1e6  # ms
        total_wall_time = (kernels[-1].end_ns - kernels[0].start_ns) / 1e6
        gpu_utilization = (total_kernel_time / total_wall_time * 100) if total_wall_time > 0 else 0
        lines.append(f"  Total Kernel Time: {total_kernel_time:.2f} ms")
        lines.append(f"  Wall Clock Time: {total_wall_time:.2f} ms")
        lines.append(f"  GPU Utilization: {gpu_utilization:.1f}%")

        if gpu_utilization < 50:
            lines.append("  ⚠️  WARNING: Low GPU utilization indicates CPU bottleneck!")
    lines.append("")

    # ========================================
    # 2. Kernel Gap Analysis (CRITICAL)
    # ========================================
    lines.append("## 2. KERNEL GAP ANALYSIS (Launch Latency)")
    lines.append("-" * 40)

    gaps = analyze_kernel_gaps(kernels)
    severe_gaps = [g for g in gaps if g.severity == "SEVERE"]
    warning_gaps = [g for g in gaps if g.severity == "WARNING"]

    lines.append(f"  Total Gaps Analyzed: {len(gaps)}")
    lines.append(f"  SEVERE Gaps (>50us): {len(severe_gaps)}")
    lines.append(f"  WARNING Gaps (20-50us): {len(warning_gaps)}")
    lines.append(f"  NORMAL Gaps (<20us): {len(gaps) - len(severe_gaps) - len(warning_gaps)}")

    if gaps:
        avg_gap = sum(g.gap_us for g in gaps) / len(gaps)
        max_gap = max(g.gap_us for g in gaps)
        lines.append(f"  Average Gap: {avg_gap:.2f} us")
        lines.append(f"  Max Gap: {max_gap:.2f} us")

        if avg_gap > 50:
            lines.append("")
            lines.append("  🚨 DIAGNOSIS: CPU LAUNCH BOUND")
            lines.append("     - Python for-loop overhead is significant")
            lines.append("     - Recommend: CUDA Graph capture or kernel fusion")
        elif avg_gap > 20:
            lines.append("")
            lines.append("  ⚠️  DIAGNOSIS: MODERATE LAUNCH OVERHEAD")
            lines.append("     - Some CPU intervention between kernels")
            lines.append("     - Consider: Reduce Python operations in hot path")
        else:
            lines.append("")
            lines.append("  ✅ DIAGNOSIS: KERNEL LAUNCH EFFICIENT")
            lines.append("     - CUDA Graph or continuous kernel submission working")
    lines.append("")

    # Top 10 worst gaps
    if severe_gaps:
        lines.append("  Top 10 Worst Gaps:")
        for i, g in enumerate(sorted(severe_gaps, key=lambda x: -x.gap_us)[:10]):
            lines.append(f"    {i+1}. {g.gap_us:.1f} us: {g.from_kernel[:30]} -> {g.to_kernel[:30]}")
        lines.append("")

    # ========================================
    # 3. Step-to-Step Gap Analysis
    # ========================================
    lines.append("## 3. STEP-TO-STEP GAP ANALYSIS")
    lines.append("-" * 40)

    step_analysis = analyze_step_gaps(nvtx_ranges, kernels)
    lines.append(f"  Steps Detected: {step_analysis['step_count']}")
    lines.append(f"  Mean Inter-Step Gap: {step_analysis['mean_gap_us']:.2f} us")
    lines.append(f"  Max Inter-Step Gap: {step_analysis['max_gap_us']:.2f} us")

    if step_analysis['mean_gap_us'] > 50:
        lines.append("")
        lines.append("  🚨 CRITICAL: Large gaps between denoising steps!")
        lines.append("     - This is the 'bubble' you suspected")
        lines.append("     - Each step is waiting for CPU to dispatch next iteration")
    elif step_analysis['mean_gap_us'] > 20:
        lines.append("")
        lines.append("  ⚠️  WARNING: Noticeable inter-step gaps")
    else:
        lines.append("")
        lines.append("  ✅ Steps are efficiently overlapped")
    lines.append("")

    # ========================================
    # 4. Stream Synchronization Analysis (CRITICAL)
    # ========================================
    lines.append("## 4. STREAM SYNCHRONIZATION ANALYSIS")
    lines.append("-" * 40)

    if sync_events:
        lines.append(f"  🚨 FOUND {len(sync_events)} SYNC EVENTS!")
        lines.append("  These are potential performance killers:")
        lines.append("")
        for i, s in enumerate(sync_events[:20]):
            lines.append(f"    {i+1}. {s.name}: {s.duration_ns/1000:.2f} us")

        lines.append("")
        lines.append("  DIAGNOSIS: IMPLICIT SYNCHRONIZATION DETECTED")
        lines.append("  Common causes:")
        lines.append("    - print(tensor) in hot path")
        lines.append("    - tensor.item() or tensor.cpu()")
        lines.append("    - Python conditionals on GPU tensor values")
        lines.append("    - torch.cuda.synchronize() calls")
        lines.append("")
        lines.append("  ACTION: Remove ALL sync points from denoise loop!")
    else:
        lines.append("  ✅ No explicit synchronization events found")
        lines.append("     (Note: some implicit syncs may not be captured)")
    lines.append("")

    # ========================================
    # 5. Kernel Type Analysis
    # ========================================
    lines.append("## 5. KERNEL TYPE DISTRIBUTION")
    lines.append("-" * 40)

    kernel_types = analyze_kernel_types(kernels)
    total_time = sum(s["total_us"] for s in kernel_types.values())

    for category in sorted(kernel_types.keys(), key=lambda x: -kernel_types[x]["total_us"]):
        stats = kernel_types[category]
        pct = (stats["total_us"] / total_time * 100) if total_time > 0 else 0
        lines.append(f"  {category}:")
        lines.append(f"    Count: {stats['count']}, Time: {stats['total_us']:.2f} us ({pct:.1f}%)")
    lines.append("")

    # ========================================
    # 6. MLP Analysis (Memory Bound Check)
    # ========================================
    lines.append("## 6. MLP KERNEL ANALYSIS (Memory Bandwidth)")
    lines.append("-" * 40)

    mlp_analysis = analyze_mlp_kernels(kernels, nvtx_ranges)
    lines.append(f"  MLP NVTX Ranges: {mlp_analysis['mlp_range_count']}")
    lines.append(f"  MLP Total Time: {mlp_analysis['mlp_total_us']:.2f} us")

    if mlp_analysis['mlp_kernels']:
        lines.append("  Sample MLP Kernels:")
        for i, k in enumerate(mlp_analysis['mlp_kernels'][:10]):
            lines.append(f"    {i+1}. {k['name']} ({k['duration_us']:.2f} us)")
            lines.append(f"       Grid: {k['grid']}, Block: {k['block']}")

    lines.append("")
    lines.append("  NOTE: To determine if memory-bound, check GPU Metrics in Nsight:")
    lines.append("    - If SM Utilization < 30% AND DRAM Bandwidth > 80%")
    lines.append("    - → Memory Bound: Need L2 Cache Residency or weight reuse")
    lines.append("")

    # ========================================
    # 7. Recommendations
    # ========================================
    lines.append("## 7. RECOMMENDATIONS")
    lines.append("=" * 40)

    # Based on analysis
    if severe_gaps:
        lines.append("  1. 🚨 PRIORITY: Reduce Kernel Launch Gaps")
        lines.append("     - Current gaps indicate CPU bottleneck")
        lines.append("     - Use CUDA Graph to capture entire denoise loop")
        lines.append("     - Or use persistent kernels (grid-level loop)")
        lines.append("")

    if sync_events:
        lines.append("  2. 🚨 PRIORITY: Remove Synchronization Points")
        lines.append("     - Found explicit sync events in trace")
        lines.append("     - Check for: print(), .item(), .cpu(), tensor conditionals")
        lines.append("")

    lines.append("  3. Check Memory Bandwidth in Nsight Systems GUI:")
    lines.append("     - Open .nsys-rep file in Nsight Systems")
    lines.append("     - Enable GPU Metrics row")
    lines.append("     - Look for DRAM Read/Write bandwidth")
    lines.append("     - If > 80% sustained during MLP: Memory Bound confirmed")
    lines.append("")

    lines.append("  4. For Memory-Bound Layers:")
    lines.append("     - Enable L2 Cache Residency (cudaStreamAttrValue)")
    lines.append("     - Consider weight compression (INT4/FP4)")
    lines.append("     - Or use persistent GEMM with tile-level reuse")
    lines.append("")

    lines.append("=" * 80)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze Nsight Systems export for denoise performance")
    parser.add_argument("sqlite_file", type=str, help="Path to .sqlite file (exported from nsys)")
    parser.add_argument("--output", "-o", type=str, help="Output report file path")
    parser.add_argument("--json", action="store_true", help="Also output JSON summary")
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_file)
    if not sqlite_path.exists():
        print(f"Error: File not found: {sqlite_path}")
        sys.exit(1)

    print(f"Analyzing: {sqlite_path}")
    print("-" * 40)

    conn = sqlite3.connect(str(sqlite_path))

    # Query data
    print("Querying kernel events...")
    kernels = query_kernels(conn)
    print(f"  Found {len(kernels)} kernels")

    print("Querying NVTX ranges...")
    nvtx_ranges = query_nvtx_ranges(conn)
    print(f"  Found {len(nvtx_ranges)} NVTX ranges")

    print("Querying sync events...")
    sync_events = query_sync_events(conn)
    print(f"  Found {len(sync_events)} sync events")

    conn.close()

    # Generate report
    output_path = args.output or str(sqlite_path.with_suffix(".analysis.txt"))
    report = generate_report(kernels, nvtx_ranges, sync_events, output_path)
    print("\n" + report)

    if args.json:
        json_path = str(sqlite_path.with_suffix(".analysis.json"))
        json_data = {
            "kernel_count": len(kernels),
            "nvtx_count": len(nvtx_ranges),
            "sync_count": len(sync_events),
            "gaps": [
                {"from": g.from_kernel, "to": g.to_kernel, "gap_us": g.gap_us, "severity": g.severity}
                for g in analyze_kernel_gaps(kernels)[:100]
            ],
            "step_analysis": analyze_step_gaps(nvtx_ranges, kernels),
            "kernel_types": analyze_kernel_types(kernels),
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON summary saved to: {json_path}")


if __name__ == "__main__":
    main()
