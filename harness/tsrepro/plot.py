"""Parse experiment results and draw the paper's figures and tables.

    python -m tsrepro.plot all            # every figure/table that has results
    python -m tsrepro.plot fig08 tab03    # selected ones

For each figure this writes ``figures/<name>.pdf``/``.png`` and
``figures/<name>.csv`` (reproduced values next to the paper's values), and
``figures/REPORT.md`` collects all comparison tables. Paper values come from
``paper_results/paper_values.csv`` and are drawn as black bars/markers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import plotstyle as ps
from .plotstyle import plt
from . import results as R
from .results import (
    PAPER,
    hand_back,
    REPO,
    THROUGHPUT,
    Run,
    by_workload,
    load_runs,
    mean_over_reps,
    paper_values,
    tags,
)

FIGURES = Path(REPO / "figures")
FIGURES_FAST = Path(REPO / "figures_fast")
STEADY_EPOCH = 2  # third epoch, as in the paper's analysis


def steady_epoch(run: Run, preferred: int = STEADY_EPOCH) -> int | None:
    """``preferred`` epoch, or the last logged one for shortened runs (--var epochs=N)."""
    if R.FAST:
        return None
    steps = set()
    for p in run.trainers:
        steps.update(int(s) for s in p.series(THROUGHPUT).step.unique())
    if not steps or preferred in steps:
        return preferred
    return max(s for s in steps if s <= preferred) if any(s <= preferred for s in steps) else max(steps)


def gpu_util_pct(run: Run, window) -> float:
    """Mean SM activity (DCGM) over the workload's GPUs; nvidia-smi utilisation if DCGM is missing."""
    vals = [run.monitor_mean("DCGMI - SM Active", window, g) * 100 for g in run.gpu_ids()]
    vals = [v for v in vals if not np.isnan(v)]
    if vals:
        return float(np.mean(vals))
    vals = [run.monitor_mean("SMI - GPU Util", window, g) for g in run.gpu_ids()]
    return mean_over_reps(vals)


def save(fig, name: str) -> None:
    if R.FAST:
        # Just above the figure (and any suptitle), so it never collides with axis labels.
        fig.text(0.5, 1.0, "fast mode: short runs, measured after warm-up (not the paper's full protocol)",
                 ha="center", va="bottom", fontsize=9, style="italic", color="dimgray")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)


def comparison(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if "paper" in df and "reproduced" in df:
        df["ratio"] = df["reproduced"] / df["paper"]
    df.to_csv(FIGURES / f"{name}.csv", index=False)
    return df


def _paper_lookup(figure: str):
    pv = paper_values(figure)

    def get(metric, label="", group="", x=""):
        m = pv[(pv.metric == metric) & (pv.label.astype(str) == str(label))]
        if group != "":
            m = m[m.group.astype(str) == str(group)]
        if x != "":
            m = m[m.x.astype(str) == str(x)]
        return float(m.value.iloc[0]) if len(m) else float("nan")

    return get


def _bars(ax, categories, series: dict, paper: dict | None = None, width=0.38, ylabel=""):
    """Grouped bars; ``paper`` draws the paper's value as a black tick on each bar."""
    x = np.arange(len(categories))
    n = len(series)
    for i, (label, values) in enumerate(series.items()):
        pos = x + (i - (n - 1) / 2) * width
        ax.bar(pos, values, width, label=label, color=ps.COLOURS[i], hatch=ps.HATCHES[i], edgecolor="black")
        if paper and label in paper:
            ax.plot(pos, paper[label], **ps.PAPER_STYLE)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylabel(ylabel)


def _paper_handle():
    from matplotlib.lines import Line2D

    return Line2D([], [], label="Paper", **ps.PAPER_STYLE)


# --------------------------------------------------------------------- Figure 1
def fig01() -> pd.DataFrame | None:
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    df = pd.read_csv(PAPER / "cloud_instances.csv", comment="#")
    cmap = LinearSegmentedColormap.from_list("", ["#B8D1FF", "#287E0B"])
    providers = ["AWS", "Azure", "GCP"]
    fig, axs = plt.subplots(1, 3, figsize=(8, 3))
    # As in the paper, the axes are the vCPU and GPU counts that AWS offers.
    aws = df[df.provider == "AWS"]
    vcpus = sorted(aws.vcpus.unique(), reverse=True)
    gpus = sorted(aws.gpus.unique())
    for ax, provider in zip(axs, providers):
        sub = df[df.provider == provider]
        heat = sub.groupby(["vcpus", "gpus"]).size().unstack(fill_value=0).reindex(index=vcpus, columns=gpus, fill_value=0)
        sns.heatmap(heat, ax=ax, cmap=cmap, vmin=1, vmax=9, cbar=False, annot=True, mask=heat == 0, linewidth=0.5)
        ax.set_title(provider)
        ax.grid(False)
        ax.set_xlabel("GPU Count" if provider == "Azure" else "")
        ax.set_ylabel("vCPU Count" if provider == "AWS" else "")
    save(fig, "fig01_cloud_cpu_gpu_ratio")
    return None


# --------------------------------------------------------------------- Figure 8
FIG08_MODELS = [
    ("legacy_seresnet18", "ResNet18", "1", "11"),
    ("regnetx_002", "RegNetX 2", "2", "21"),
    ("regnetx_004", "RegNetX 4", "3", "31"),
    ("tf_mobilenetv3_small_075", "MobileNet S", "5", "51"),
    ("mobilenetv3_large_100", "MobileNet L", "6", "61"),
]


def _timm_summary(runs: list[Run]) -> dict[str, float]:
    thr, cpu, gpu = [], [], []
    for r in runs:
        e = steady_epoch(r)
        w = r.window(e)
        thr.append(r.mean_throughput(e))
        cpu.append(r.cpu_util_pct(w))
        gpu.append(gpu_util_pct(r, w))
    return {"samples_per_s": mean_over_reps(thr), "cpu_util_pct": mean_over_reps(cpu), "gpu_util_pct": mean_over_reps(gpu)}


def fig08(exp: str = "fig08_a100_4way", title: str = "Figure 8: 4-way collocation on the A100 server (one model per GPU)") -> pd.DataFrame | None:
    runs = by_workload(load_runs(exp))
    if not runs:
        return None
    paper = _paper_lookup("fig08")
    rows = []
    for model, name, ns, sh in FIG08_MODELS:
        for mode, wid in (("Non-shared", ns), ("Shared", sh)):
            s = _timm_summary(runs.get(wid, []))
            for metric, v in s.items():
                rows.append(dict(model=name, mode=mode, metric=metric, reproduced=v, paper=paper(metric, mode, model)))
    df = pd.DataFrame(rows)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    names = [m[1] for m in FIG08_MODELS]
    for ax, metric, ylabel in zip(
        axs,
        ["samples_per_s", "cpu_util_pct", "gpu_util_pct"],
        ["Samples/s (per model)", "% CPU Utilization", "% GPU Utilization (SM activity)"],
    ):
        sub = df[df.metric == metric]
        series = {m: [sub[(sub.model == n) & (sub["mode"] == m)].reproduced.iloc[0] for n in names] for m in ("Non-shared", "Shared")}
        paper_s = {m: [sub[(sub.model == n) & (sub["mode"] == m)].paper.iloc[0] for n in names] for m in ("Non-shared", "Shared")}
        _bars(ax, names, series, paper_s, ylabel=ylabel)
        if metric != "samples_per_s":
            ax.set_ylim(0, 110)
    axs[0].legend(handles=axs[0].get_legend_handles_labels()[0] + [_paper_handle()], loc="upper left")
    fig.suptitle(title)
    save(fig, exp)
    return comparison(exp, df)


# --------------------------------------------------------------------- Figure 9
def fig09(exp: str = "fig09_a100_collocation", exp4: str = "fig08_a100_4way", title: str = "") -> pd.DataFrame | None:
    runs = by_workload(load_runs(exp))
    runs4 = by_workload(load_runs(exp4))
    if not runs:
        return None
    paper = _paper_lookup("fig09")
    ids = {
        ("Small", "Non-shared"): ["150", "151", "152", "5"],
        ("Small", "Shared"): ["155", "156", "157", "51"],
        ("Large", "Non-shared"): ["160", "161", "162", "6"],
        ("Large", "Shared"): ["165", "166", "167", "61"],
    }
    rows = []
    for (model, mode), wids in ids.items():
        for degree, wid in enumerate(wids, 1):
            rs = runs.get(wid) or runs4.get(wid, [])
            v = mean_over_reps([r.mean_throughput(steady_epoch(r)) for r in rs])
            rows.append(dict(model=model, mode=mode, degree=degree, metric="samples_per_s", reproduced=v, paper=paper("samples_per_s", mode, model, degree)))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, ((model, mode), _) in enumerate(ids.items()):
        sub = df[(df.model == model) & (df["mode"] == mode)]
        ax.plot(sub.degree, sub.reproduced, marker=ps.MARKERS[i], color=ps.COLOURS[i], lw=2.5, label=f"{model} {mode}")
        ax.plot(sub.degree, sub.paper, marker=ps.MARKERS[i], color=ps.COLOURS[i], lw=1, ls=":", alpha=0.7)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1x", "2x", "3x", "4x"])
    ax.set_xlabel("Collocation degree")
    ax.set_ylabel("Samples/s (per model)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="solid: reproduced\ndotted: paper")
    if title:
        ax.set_title(title)
    save(fig, exp)
    return comparison(exp, df)


# ---------------------------------------------------------------------- Table 3
def tab03(exp: str = "tab03_a100_data_movement") -> pd.DataFrame | None:
    runs = by_workload(load_runs(exp))
    if not runs:
        return None
    paper = _paper_lookup("tab03")
    rows = []
    for mode, wid in (("Baseline", "6"), ("Shared", "61")):
        rs = runs.get(wid, [])
        if not rs:
            continue
        io = mean_over_reps([r.monitor_mean("IO - MB read/s", r.window(steady_epoch(r, 99))) for r in rs])
        rows.append(dict(mode=mode, gpu="all", metric="io_MBps", reproduced=io, paper=paper("io_MBps", "all", mode)))
        for g in rs[0].gpu_ids():
            label = f"GPU {g}"
            for metric, key, scale in (
                ("pcie_MBps", "DCGMI - PCIE RX Bytes", 1e-6),
                ("nvlink_MBps", "DCGMI - NVlink RX Bytes", 1e-6),
                ("vram_GB", "SMI - Mem Used", 1 / 1024),
            ):
                v = mean_over_reps([r.monitor_mean(key, r.window(steady_epoch(r, 99)), g) * scale for r in rs])
                rows.append(dict(mode=mode, gpu=label, metric=metric, reproduced=v, paper=paper(metric, label, mode)))
    df = pd.DataFrame(rows)
    return comparison(exp, df)


# -------------------------------------------------------------------- Figure 10
def fig10() -> pd.DataFrame | None:
    runs = by_workload(load_runs("fig10_h100_flexible"))
    if not runs:
        return None
    paper = _paper_lookup("fig10")
    rows = []
    for mode, wid in (("Default", "100"), ("Flexible", "105")):
        s = _timm_summary(runs.get(wid, []))
        for metric in ("samples_per_s", "cpu_util_pct"):
            rows.append(dict(mode=mode, metric=metric, reproduced=s[metric], paper=paper(metric, mode)))
    df = pd.DataFrame(rows)

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    for ax, metric, ylabel in zip(axs, ("samples_per_s", "cpu_util_pct"), ("Samples/s (per model)", "% CPU Utilization")):
        sub = df[df.metric == metric]
        for i, mode in enumerate(("Default", "Flexible")):
            r = sub[sub["mode"] == mode]
            ax.bar(i, r.reproduced.iloc[0], 0.6, color=ps.COLOURS[i], hatch=ps.HATCHES[i])
            ax.plot(i, r.paper.iloc[0], **ps.PAPER_STYLE)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Default", "Flexible"])
        ax.set_ylabel(ylabel)
    axs[1].set_ylim(0, 100)
    axs[1].legend(handles=[_paper_handle()])
    fig.suptitle("Figure 10: three MobileNet S on the H100, batch 128 vs 128/192/224")
    save(fig, "fig10_h100_flexible")
    return comparison("fig10_h100_flexible", df)


# -------------------------------------------------------------------- Figure 11
INSTANCES = ["g5.2xlarge", "g5.4xlarge", "g5.8xlarge"]
INSTANCE_CPUS = {"g5.2xlarge": 8, "g5.4xlarge": 16, "g5.8xlarge": 32}


def fig11() -> pd.DataFrame | None:
    exp = "fig11_g5_clmr"
    present = [t for t in INSTANCES if t in tags(exp)]
    if not present:
        return None
    paper = _paper_lookup("fig11")
    rows = []
    for inst in present:
        runs = by_workload(load_runs(exp, tag=inst))
        for mode, wid_streams, wid_mps in (("Non-shared", "4", "204"), ("Shared", "104", "304")):
            for col, wid in (("no_mps", wid_streams), ("mps", wid_mps)):
                v = mean_over_reps([r.mean_throughput(None) for r in runs.get(wid, [])])
                rows.append(dict(instance=inst, mode=mode, collocation=col, metric="samples_per_s", reproduced=v, paper=paper("samples_per_s", mode, inst, col)))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(present))
    for i, mode in enumerate(("Non-shared", "Shared")):
        pos = x + (i - 0.5) * 0.35
        for col, alpha, z in (("mps", 0.35, 2), ("no_mps", 1.0, 3)):
            sub = df[(df["mode"] == mode) & (df.collocation == col)]
            vals = [sub[sub.instance == t].reproduced.iloc[0] for t in present]
            ax.bar(pos, vals, 0.35, color=ps.COLOURS[i], hatch=ps.HATCHES[i], alpha=alpha, zorder=z,
                   label=f"{mode}{' (MPS)' if col == 'mps' else ''}")
            ax.plot(pos, [sub[sub.instance == t].paper.iloc[0] for t in present], **(ps.PAPER_STYLE | {"zorder": 6}))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{INSTANCE_CPUS[t]} vCPUs\n({t})" for t in present])
    ax.set_ylabel("Samples/s (per model)")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [_paper_handle()], loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Figure 11: 4x CLMR on AWS G5, streams (solid) and MPS (light)")
    save(fig, "fig11_g5_clmr")
    return comparison("fig11_g5_clmr", df)


# -------------------------------------------------------------------- Figure 12
def fig12() -> pd.DataFrame | None:
    runs = by_workload(load_runs("fig12_h100_dalle2"))
    if not runs:
        return None
    paper = _paper_lookup("fig12")
    rows = []
    for mode, ids in (("Non-shared", ("11", "22", "44")), ("Shared", ("111", "122", "144"))):
        for degree, wid in zip((1, 2, 4), ids):
            rs = runs.get(wid, [])
            v = mean_over_reps([r.mean_throughput(steady_epoch(r)) for r in rs])
            wall = mean_over_reps([np.mean([p.throughput(steady_epoch(r), "ML - Epoch Samples/s") for p in r.trainers]) for r in rs])
            rows.append(dict(mode=mode, degree=degree, metric="samples_per_s", reproduced=v, paper=paper("samples_per_s", mode, "", degree)))
            if not R.FAST:  # end-of-epoch metric, absent from shortened runs
                rows.append(dict(mode=mode, degree=degree, metric="wallclock_samples_per_s", reproduced=wall, paper=np.nan))
    df = pd.DataFrame(rows)

    sub = df[df.metric == "samples_per_s"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cats = ["1x", "2x", "4x"]
    series = {m: list(sub[sub["mode"] == m].reproduced) for m in ("Non-shared", "Shared")}
    paper_s = {m: list(sub[sub["mode"] == m].paper) for m in ("Non-shared", "Shared")}
    _bars(ax, cats, series, paper_s, width=0.35, ylabel="Samples/s (per model)")
    ax.set_xlabel("Collocation degree")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [_paper_handle()])
    ax.set_title("Figure 12: online DALL-E 2 prior training on the H100")
    save(fig, "fig12_h100_dalle2")
    return comparison("fig12_h100_dalle2", df)


# -------------------------------------------------------------------- Figure 13
def _aggregate_series(run: Run) -> pd.DataFrame:
    """Summed throughput of the run's trainers over elapsed time."""
    series = [p.series(THROUGHPUT).sort_values("timestamp")[["timestamp", "value"]] for p in run.trainers]
    series = [s for s in series if len(s)]
    if not series:
        return pd.DataFrame(columns=["elapsed_s", "value"])
    base = series[0].rename(columns={"value": "v0"})
    for i, s in enumerate(series[1:], 1):
        base = pd.merge_asof(base, s.rename(columns={"value": f"v{i}"}), on="timestamp", direction="nearest")
    base["value"] = base.filter(like="v").sum(axis=1)
    base["elapsed_s"] = (base.timestamp - base.timestamp.iloc[0]) / 1000
    return base[["elapsed_s", "value"]]


def fig13() -> pd.DataFrame | None:
    exp = "fig13_g5_model_selection"
    present = [t for t in INSTANCES if t in tags(exp)]
    if not present:
        return None
    paper = _paper_lookup("fig13")
    original = pd.read_csv(PAPER / "fig13_original_series.csv")
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for i, inst in enumerate(INSTANCES):
        runs = by_workload(load_runs(exp, tag=inst)) if inst in present else {}
        for ax, (mode, wid) in zip(axs, (("Non-shared", "1"), ("Shared", "101"))):
            o = original[(original.instance == inst) & (original["mode"] == mode)]
            ax.plot(o.elapsed_s, o.aggregate_samples_per_s, color=ps.COLOURS[i], lw=1, ls=":", alpha=0.8)
            rs = runs.get(wid, [])
            if not rs:
                continue
            s = _aggregate_series(rs[0])
            ax.plot(s.elapsed_s, s.value, color=ps.COLOURS[i], lw=2.5, label=f"{inst} ({INSTANCE_CPUS[inst]} vCPUs)")
            agg = mean_over_reps([_aggregate_series(r).value.mean() for r in rs])
            runtime = mean_over_reps([(r.meta["end_ms"] - r.meta["start_ms"]) / 1000 for r in rs])
            rows.append(dict(instance=inst, mode=mode, metric="aggregate_samples_per_s", reproduced=agg, paper=paper("aggregate_samples_per_s", mode, inst)))
            rows.append(dict(instance=inst, mode=mode, metric="runtime_s", reproduced=runtime, paper=paper("runtime_s", mode, inst)))
    for ax, title in zip(axs, ("(a) Non-shared", "(b) Shared")):
        ax.set_title(title)
        ax.set_xlabel("Elapsed time (s)")
    axs[0].set_ylabel("Aggregate samples/s")
    axs[1].legend(title="solid: reproduced, dotted: paper")
    fig.suptitle("Figure 13: RegNetX 2 + RegNetX 4 collocated on AWS G5")
    save(fig, "fig13_g5_model_selection")
    return comparison("fig13_g5_model_selection", pd.DataFrame(rows))


# ---------------------------------------------------------------------- Table 4
def tab04(exp: str = "tab04_a100_llm") -> pd.DataFrame | None:
    runs = by_workload(load_runs(exp))
    if not runs:
        return None
    paper = _paper_lookup("tab04")
    rows = []
    for mode, wid in (("Baseline", "1"), ("Shared", "101")):
        rs = runs.get(wid, [])
        if not rs:
            continue
        r0 = rs[0]
        single_gpu = len(r0.gpu_ids()) == 1
        for k, p in enumerate(r0.trainers):
            g = int(p.devices.split(",")[0])
            # On one GPU, model k stands in for the paper's model on GPU k+1.
            label, paper_label = (f"model {k + 1}", f"GPU {k + 1}") if single_gpu else (f"GPU {g}", f"GPU {g}")
            v = mean_over_reps([r.proc_median(next(q for q in r.trainers if q.index == p.index), "ML - Tokens/s", skip=5) for r in rs])
            rows.append(dict(mode=mode, gpu=label, metric="tokens_per_s", reproduced=v, paper=paper("tokens_per_s", paper_label, mode)))
        for g in r0.gpu_ids():
            for metric, key, scale in (
                ("pcie_MBps", "DCGMI - PCIE RX Bytes", 1e-6),
                ("nvlink_KBps", "DCGMI - NVlink RX Bytes", 1e-3),
                ("vram_GB", "SMI - Mem Used", 1 / 1024),
            ):
                v = mean_over_reps([r.monitor_mean(key, r.window(None), g) * scale for r in rs])
                rows.append(dict(mode=mode, gpu=f"GPU {g}", metric=metric, reproduced=v, paper=paper(metric, f"GPU {g}", mode)))
    return comparison(exp, pd.DataFrame(rows))


# -------------------------------------------------------------------- Figure 14
def _top_cpu(path: Path, skip_s: int = 60) -> float:
    """Mean summed %CPU per snapshot of a ``top -b -d 1`` log."""
    totals, current, in_table = [], 0.0, False
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("top - "):
            if in_table:
                totals.append(current)
            current, in_table = 0.0, False
        elif line.lstrip().startswith("PID"):
            in_table = True
        elif in_table and line.strip():
            parts = line.split()
            try:
                current += float(parts[8].replace(",", "."))
            except (IndexError, ValueError):
                pass
    if in_table:
        totals.append(current)
    totals = totals[skip_s:] or totals
    return float(np.mean(totals)) if totals else float("nan")


def _coordl_results(exp: str = "fig14_a100_coordl") -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, list[float]]] = {}
    for log in sorted((R.RESULTS / exp).glob("*/n*/rep*/log.txt")):
        n = int(re.search(r"/n(\d+)/", str(log)).group(1))
        text = log.read_text(errors="replace")
        perf = re.findall(r"##Perf\s*:?\s*([0-9.]+)", text)
        if not perf:
            # Runs stopped early (fast mode) never print ##Perf; use the last running average.
            perf = re.findall(r"Speed\s+[0-9.]+\s+\(([0-9.]+)\)", text)
        if not perf:
            continue
        d = out.setdefault(n, {"thr": [], "cpu": []})
        d["thr"].append(float(perf[-1]))
        d["cpu"].append(_top_cpu(log.with_name("top.txt")))
    return {n: {k: mean_over_reps(v) for k, v in d.items()} for n, d in out.items()}


def fig14(
    exp: str = "fig14_a100_coordl_ts",
    coordl_exp: str = "fig14_a100_coordl",
    title: str = "Figure 14: scaling of baseline, TensorSocket and CoorDL (ResNet18, A100)",
    name: str = "fig14_a100_coordl",
) -> pd.DataFrame | None:
    runs = by_workload(load_runs(exp))
    coordl = _coordl_results(coordl_exp)
    if not runs and not coordl:
        return None
    paper = _paper_lookup("fig14")
    raw = {}
    for label, base in (("Baseline", 609), ("TensorSocket", 619)):
        for k in (1, 2, 3, 4):
            rs = runs.get(str(base + k), [])
            thr = mean_over_reps([r.mean_throughput(steady_epoch(r)) for r in rs])
            cpu = mean_over_reps(
                [
                    r.monitor_mean(
                        "TOP - CPU Utilization",
                        r.window(None) if R.FAST else (r.meta["start_ms"] + 60_000, r.meta["end_ms"]),
                    )
                    for r in rs
                ]
            )
            raw[(label, k)] = (thr, cpu)
    for k, d in coordl.items():
        raw[("CoorDL", k)] = (d["thr"], d["cpu"])

    rows = []
    for (label, k), (thr, cpu) in sorted(raw.items()):
        base_thr, base_cpu = raw.get((label, 1), (np.nan, np.nan))
        rows.append(dict(method=label, degree=k, metric="norm_throughput", reproduced=thr / base_thr, paper=paper("norm_throughput", label, "", k), raw=thr))
        rows.append(dict(method=label, degree=k, metric="norm_cpu", reproduced=cpu / base_cpu, paper=paper("norm_cpu", label, "", k), raw=cpu))
    df = pd.DataFrame(rows)

    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    for ax, metric, ylabel in zip(axs, ("norm_cpu", "norm_throughput"), ("x CPU Utilization", "x Model throughput")):
        for i, label in enumerate(("Baseline", "TensorSocket", "CoorDL")):
            sub = df[(df.method == label) & (df.metric == metric)]
            if sub.reproduced.notna().any():
                ax.plot(sub.degree, sub.reproduced, marker=ps.MARKERS[i], color=ps.COLOURS[i], lw=3, ms=9, label=label)
            ax.plot(sub.degree, sub.paper, marker=ps.MARKERS[i], color=ps.COLOURS[i], lw=1, ls=":", alpha=0.7)
        ax.set_ylabel(ylabel)
    axs[1].set_ylim(0, 1.1)
    axs[1].set_xticks([1, 2, 3, 4])
    axs[1].set_xticklabels(["1x", "2x", "3x", "4x"])
    axs[1].set_xlabel("Collocation degree")
    axs[0].legend(title="solid: reproduced, dotted: paper")
    fig.suptitle(title)
    save(fig, name)
    return comparison(name, df)


# -------------------------------------------------------------------- Figure 15
def fig15() -> pd.DataFrame | None:
    ts = by_workload(load_runs("fig15_h100_joader_ts"))
    jo = by_workload(load_runs("fig15_h100_joader"))
    if not ts and not jo:
        return None
    paper = _paper_lookup("fig15")

    def per_model_median(rs: list[Run]) -> float:
        return mean_over_reps([np.mean([r.proc_median(p) for p in r.trainers]) for r in rs])

    rows = []
    for k in range(1, 9):
        for label, src, wid in (("Baseline", ts, 310 + k), ("TensorSocket", ts, 320 + k), ("Joader", jo, 300 + k)):
            rows.append(dict(method=label, models=k, metric="samples_per_s", reproduced=per_model_median(src.get(str(wid), [])), paper=paper("samples_per_s", label, "", k)))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    cats = [str(k) for k in range(1, 9)]
    series = {m: list(df[df.method == m].reproduced) for m in ("Baseline", "TensorSocket", "Joader")}
    paper_s = {m: list(df[df.method == m].paper) for m in ("Baseline", "TensorSocket", "Joader")}
    _bars(ax, cats, series, paper_s, width=0.28, ylabel="Samples/s (per model)")
    ax.set_xticklabels(cats, rotation=0, ha="center")
    ax.set_xlabel("Collocated MobileNet Small models")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [_paper_handle()], ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Figure 15: baseline vs TensorSocket vs Joader on the H100 (8 data-loading workers)", pad=32)
    save(fig, "fig15_h100_joader")
    return comparison("fig15_h100_joader", df)


# ------------------------------------------------------------------------ smoke
def smoke() -> pd.DataFrame | None:
    runs = by_workload(load_runs("smoke"))
    if not runs:
        return None
    rows = []
    for wid, mode in (("1", "Non-shared"), ("2", "Shared")):
        for r in runs.get(wid, []):
            e = steady_epoch(r, 1)
            rows.append(
                dict(
                    mode=mode,
                    tag=r.meta["tag"],
                    epochs_logged=sorted({int(s) for p in r.trainers for s in p.series(THROUGHPUT).step.unique()}),
                    samples_per_s=r.mean_throughput(e),
                    cpu_util_pct=r.cpu_util_pct(r.window(e)),
                    monitors=sorted(r.monitors.key.unique()),
                )
            )
    df = pd.DataFrame(rows)
    FIGURES.mkdir(exist_ok=True)
    df.to_csv(FIGURES / "smoke.csv", index=False)
    return df


ALL = {
    "fig01": fig01,
    "fig08": fig08,
    "fig09": fig09,
    "tab03": tab03,
    "fig10": fig10,
    "fig11": fig11,
    "fig12": fig12,
    "fig13": fig13,
    "tab04": tab04,
    "fig14": fig14,
    "fig15": fig15,
    # Single-GPU stand-ins for the A100 experiments (all models share one GPU via MPS).
    "fig08-1gpu": lambda: fig08("fig08_h100_1gpu", "Figure 8 stand-in: 4 models sharing one H100 via MPS (paper: 4x A100, one model per GPU)"),
    "fig09-1gpu": lambda: fig09("fig09_h100_1gpu", "fig08_h100_1gpu", "Figure 9 stand-in: models sharing one H100 (dotted: paper, 1 model per A100)"),
    "tab03-1gpu": lambda: tab03("tab03_h100_1gpu"),
    "tab04-1gpu": lambda: tab04("tab04_h100_1gpu"),
    "fig14-1gpu": lambda: fig14(
        "fig14_h100_1gpu",
        "fig14_h100_1gpu_coordl",
        "Figure 14 stand-in: ResNet18 models sharing one H100 (paper: one model per A100)",
        "fig14_h100_1gpu",
    ),
}


def main(argv: list[str]) -> int:
    global FIGURES
    if "--fast" in argv:
        argv = [a for a in argv if a != "--fast"]
        R.set_fast(True)
        FIGURES = FIGURES_FAST
    which = argv or ["all"]
    if which == ["all"]:
        which = list(ALL)
    funcs = ALL | {"smoke": smoke}
    report = [
        "# Reproduced results" + (" (fast mode)" if R.FAST else "") + "\n",
        "Generated by `python -m tsrepro.plot`. `ratio` = reproduced / paper.\n",
    ]
    pd.set_option("display.width", 200)
    for name in which:
        if name not in funcs:
            print(f"unknown figure '{name}'; choose from {', '.join(funcs)}")
            return 1
        df = funcs[name]()
        if df is None:
            if name != "fig01":
                print(f"[{name}] no results yet - skipped")
            else:
                print("[fig01] figures/fig01_cloud_cpu_gpu_ratio.pdf")
            continue
        print(f"\n[{name}]")
        print(df.round(3).to_string(index=False))
        report += [f"\n## {name}\n", df.round(3).to_markdown(index=False) if _has_tabulate() else df.round(3).to_string(index=False), "\n"]
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "REPORT.md").write_text("\n".join(report))
    hand_back(FIGURES)
    print(f"\nfigures and tables written to {FIGURES}")
    return 0


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
