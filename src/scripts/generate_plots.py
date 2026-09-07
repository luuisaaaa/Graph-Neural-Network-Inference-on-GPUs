#!/usr/bin/env python3
"""Generate report-ready SVG plots from GCN benchmark/profiling text files.

The script intentionally uses only the Python standard library so it works even
on machines without matplotlib/seaborn. Outputs are written to results/plots by
default.
"""

import argparse
import html
import math
import re
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

MAIN_DATASETS = ("Cora", "BarabasiAlbert_100k", "ogbn-arxiv")
ARCH_IMPLEMENTATIONS = (
    "sequential",
    "sequential-dense",
    "cpu-dense-parallel",
    "cpu-vertex-parallel",
    "cpu-edge-parallel",
    "cpu-message-batching",
    "cuda-vertex-basic",
    "cuda-vertex-improved",
    "cuda-edge-parallel-basic",
    "cuda-edge-parallel-improved",
    "cuda-message-batching-basic",
    "cuda-message-batching-improved",
)
SCALABILITY_IMPLEMENTATIONS = (
    "cpu-vertex-parallel",
    "cpu-edge-parallel",
    "cpu-message-batching",
    "cuda-vertex-improved",
    "cuda-edge-parallel-improved",
    "cuda-message-batching-improved",
)
TOPOLOGY_IMPLEMENTATIONS = (
    "cpu-vertex-parallel",
    "cuda-vertex-improved",
    "cuda-edge-parallel-improved",
    "cuda-message-batching-improved",
)
GPU_PROFILE_IMPLEMENTATIONS = (
    "cuda-vertex-basic",
    "cuda-vertex-improved",
    "cuda-edge-parallel-basic",
    "cuda-edge-parallel-improved",
    "cuda-message-batching-basic",
    "cuda-message-batching-improved",
)

COLORS = {
    "sequential": "#4c566a",
    "sequential-dense": "#7b8794",
    "cpu-dense-parallel": "#8f6b32",
    "cpu-vertex-parallel": "#2563eb",
    "cpu-edge-parallel": "#16a34a",
    "cpu-message-batching": "#0891b2",
    "cuda-vertex-basic": "#c2410c",
    "cuda-vertex-improved": "#ea580c",
    "cuda-edge-parallel-basic": "#9333ea",
    "cuda-edge-parallel-improved": "#7c3aed",
    "cuda-message-batching-basic": "#db2777",
    "cuda-message-batching-improved": "#be185d",
    "cuda-vertex-compressed-fp16": "#64748b",
}


def esc(value):
    return html.escape(str(value), quote=True)


def slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def parse_number(value):
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_benchmark_results(root):
    records = []
    for path in sorted(root.glob("src/GCN/**/benchmark_result*.txt")):
        section = ""
        current = None
        for line_no, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("-"):
                title = text.strip("- ").strip()
                if title and title not in {
                    "Descrizione Configurazione",
                    "Risultati Medi (3 runs)",
                    "Parametri",
                    "Esecuzioni",
                }:
                    section = title
                continue

            match = re.match(
                r"Dataset: (.+?) \(Nodi: (\d+), Archi: (\d+), Dim\. Feature: (\d+)\)",
                text,
            )
            if match:
                if current:
                    records.append(current)
                current = {
                    "source_file": str(path.relative_to(root)),
                    "source_line": line_no,
                    "section": section,
                    "dataset": match.group(1),
                    "nodes": int(match.group(2)),
                    "edges": int(match.group(3)),
                    "feature_dim": int(match.group(4)),
                }
                continue
            if current is None:
                continue

            mode = re.match(r"Modalita'?\s*:\s*(.+)", text)
            if mode:
                current["implementation"] = mode.group(1).strip()
                continue
            for key, label in (
                ("hidden_dim", "Hidden dimension"),
                ("num_classes", "Classi"),
                ("num_layers", "Livelli"),
            ):
                if text.startswith(label + ":"):
                    value = text.split(":", 1)[1].strip()
                    current[key] = int(value) if value.isdigit() else value
            if "=" in text:
                key, value = text.split("=", 1)
                current[key.strip()] = parse_number(value.strip())
        if current:
            records.append(current)
    return records


def parse_profiling_results(root):
    records = []
    for path in sorted(root.glob("src/GCN/**/profiling_result*.txt")):
        section = ""
        current = None
        for line_no, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("-"):
                title = text.strip("- ").strip()
                if title and title not in {"Parametri", "Esecuzioni"}:
                    section = title
                continue

            match = re.match(
                r"Dataset: (.+?) \(Nodi: (\d+), Archi: (\d+), Dim\. Feature: (\d+)\)",
                text,
            )
            if match:
                if current:
                    records.append(current)
                current = {
                    "source_file": str(path.relative_to(root)),
                    "source_line": line_no,
                    "section": section,
                    "dataset": match.group(1),
                    "nodes": int(match.group(2)),
                    "edges": int(match.group(3)),
                    "feature_dim": int(match.group(4)),
                }
                continue
            if current is None:
                continue

            impl = re.match(r"RESULT implementation:(.+)", text)
            if impl:
                current["implementation"] = impl.group(1).strip().removesuffix("-profiling")
                continue
            for key, label in (
                ("hidden_dim", "Hidden dimension"),
                ("num_classes", "Classi"),
                ("num_layers", "Livelli"),
            ):
                if text.startswith(label + ":"):
                    value = text.split(":", 1)[1].strip()
                    current[key] = int(value) if value.isdigit() else value
            metric = re.match(r"([^:]+):\s*([0-9.]+)%?$", text)
            if metric:
                current[metric.group(1).strip()] = float(metric.group(2))
        if current:
            records.append(current)
    return records


def is_excluded(record):
    """Return a short exclusion reason for known suspicious benchmark points."""
    if (
        record.get("implementation") == "cpu-edge-parallel"
        and record.get("dataset") == "ErdosRenyi_500k"
        and record.get("hidden_dim") == 64
        and record.get("num_layers") == 2
    ):
        return "excluded: prediction checksum mismatch"
    if (
        record.get("implementation") == "cpu-message-batching"
        and record.get("dataset") == "ogbn-arxiv"
        and record.get("hidden_dim") == 64
        and record.get("num_layers") == 2
        and record.get("inference_ms", 0) > 2000
    ):
        return "excluded: timing outlier"
    return ""


def write_svg(path, width, height, elements):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(elements)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: #111827; }}
.title {{ font-size: 20px; font-weight: 700; }}
.subtitle {{ font-size: 12px; fill: #4b5563; }}
.axis {{ stroke: #374151; stroke-width: 1; }}
.grid {{ stroke: #e5e7eb; stroke-width: 1; }}
.tick {{ font-size: 11px; fill: #4b5563; }}
.label {{ font-size: 12px; fill: #111827; }}
.legend {{ font-size: 12px; fill: #111827; }}
.note {{ font-size: 11px; fill: #6b7280; }}
</style>
{content}
</svg>
""",
        encoding="utf-8",
    )


def title_block(title, subtitle, x=28, y=34):
    elements = [f'<text x="{x}" y="{y}" class="title">{esc(title)}</text>']
    if subtitle:
        elements.append(f'<text x="{x}" y="{y + 20}" class="subtitle">{esc(subtitle)}</text>')
    return elements


def log_ticks(min_value, max_value):
    if min_value <= 0 or max_value <= 0:
        return []
    start = math.floor(math.log10(min_value))
    end = math.ceil(math.log10(max_value))
    ticks = []
    for exp in range(start, end + 1):
        for multiplier in (1, 2, 5):
            tick = multiplier * (10 ** exp)
            if min_value <= tick <= max_value:
                ticks.append(tick)
    return ticks


def format_number(value):
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def draw_bar_chart(
    path,
    title,
    subtitle,
    labels,
    values,
    colors,
    y_label,
    log_scale=False,
    width=1120,
    height=650,
    notes=None,
):
    if not values:
        return
    left, right, top, bottom = 86, 28, 84, 155
    plot_w = width - left - right
    plot_h = height - top - bottom
    elements = title_block(title, subtitle)
    values_for_scale = [v for v in values if v > 0]
    if log_scale:
        y_min = min(values_for_scale) * 0.8
        y_max = max(values_for_scale) * 1.25
        ticks = log_ticks(y_min, y_max)

        def y_pos(value):
            return top + plot_h * (math.log10(y_max) - math.log10(value)) / (
                math.log10(y_max) - math.log10(y_min)
            )

    else:
        y_min = 0.0
        y_max = max(values) * 1.18 if max(values) > 0 else 1.0
        ticks = [y_max * i / 5 for i in range(6)]

        def y_pos(value):
            return top + plot_h * (1.0 - (value - y_min) / (y_max - y_min))

    for tick in ticks:
        y = y_pos(tick)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        elements.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{esc(format_number(tick))}</text>'
        )
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    elements.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" y2="{top + plot_h}" class="axis"/>')
    elements.append(
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" class="label">{esc(y_label)}</text>'
    )

    gap = 8
    bar_w = max(12, (plot_w - gap * (len(values) + 1)) / len(values))
    for idx, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + gap + idx * (bar_w + gap)
        y = y_pos(value)
        h = top + plot_h - y
        elements.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>'
        )
        elements.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" class="tick">{esc(format_number(value))}</text>'
        )
        tx = x + bar_w / 2
        ty = top + plot_h + 16
        elements.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="end" transform="rotate(-45 {tx:.1f} {ty:.1f})" class="tick">{esc(label)}</text>'
        )

    if notes:
        for idx, note in enumerate(notes):
            elements.append(f'<text x="{left}" y="{height - 36 + 15 * idx}" class="note">{esc(note)}</text>')
    write_svg(path, width, height, elements)


def draw_grouped_bars(path, title, subtitle, groups, series, values, y_label, width=1120, height=650):
    left, right, top, bottom = 78, 210, 84, 115
    plot_w = width - left - right
    plot_h = height - top - bottom
    elements = title_block(title, subtitle)
    max_value = max([value for row in values.values() for value in row.values()] or [1])
    y_max = max_value * 1.2

    def y_pos(value):
        return top + plot_h * (1.0 - value / y_max)

    for i in range(6):
        tick = y_max * i / 5
        y = y_pos(tick)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        elements.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{esc(format_number(tick))}</text>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    elements.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" y2="{top + plot_h}" class="axis"/>')
    elements.append(
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" class="label">{esc(y_label)}</text>'
    )

    group_w = plot_w / len(groups)
    inner_gap = 4
    bar_w = max(8, (group_w * 0.76 - inner_gap * (len(series) - 1)) / len(series))
    for gi, group in enumerate(groups):
        group_x = left + gi * group_w + group_w * 0.12
        for si, impl in enumerate(series):
            value = values.get(group, {}).get(impl)
            if value is None:
                continue
            x = group_x + si * (bar_w + inner_gap)
            y = y_pos(value)
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" fill="{COLORS.get(impl, "#6b7280")}"/>'
            )
        elements.append(
            f'<text x="{left + gi * group_w + group_w / 2:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{esc(group)}</text>'
        )

    legend_x = width - right + 28
    for idx, impl in enumerate(series):
        y = top + idx * 22
        elements.append(f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{COLORS.get(impl, "#6b7280")}"/>')
        elements.append(f'<text x="{legend_x + 18}" y="{y}" class="legend">{esc(impl)}</text>')
    write_svg(path, width, height, elements)


def draw_line_chart(path, title, subtitle, series_points, x_label, y_label, width=1120, height=650):
    left, right, top, bottom = 82, 220, 84, 84
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = sorted({x for points in series_points.values() for x, _ in points})
    ys = [y for points in series_points.values() for _, y in points]
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys) * 1.2

    def x_pos(value):
        return left + plot_w * (value - x_min) / (x_max - x_min)

    def y_pos(value):
        return top + plot_h * (1.0 - (value - y_min) / (y_max - y_min))

    elements = title_block(title, subtitle)
    for i in range(6):
        tick = y_max * i / 5
        y = y_pos(tick)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        elements.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{esc(format_number(tick))}</text>')
    for x in xs:
        px = x_pos(x)
        elements.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}" class="grid"/>')
        elements.append(f'<text x="{px:.1f}" y="{top + plot_h + 20}" text-anchor="middle" class="tick">{esc(format_number(x))}</text>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    elements.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" y2="{top + plot_h}" class="axis"/>')
    elements.append(f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" class="label">{esc(x_label)}</text>')
    elements.append(
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" class="label">{esc(y_label)}</text>'
    )

    for impl, points in series_points.items():
        color = COLORS.get(impl, "#6b7280")
        coords = " ".join(f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in sorted(points))
        elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            elements.append(f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="4" fill="{color}"/>')

    legend_x = width - right + 28
    for idx, impl in enumerate(series_points):
        y = top + idx * 22
        color = COLORS.get(impl, "#6b7280")
        elements.append(f'<line x1="{legend_x}" y1="{y - 5}" x2="{legend_x + 14}" y2="{y - 5}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 20}" y="{y}" class="legend">{esc(impl)}</text>')
    elements.append(f'<text x="{left}" y="{height - 10}" class="note">{esc("Note: cpu-edge-parallel on ErdosRenyi_500k is excluded until checksum is verified.")}</text>')
    write_svg(path, width, height, elements)


def heat_color(value, min_value, max_value):
    if max_value <= min_value:
        ratio = 0.0
    else:
        ratio = (value - min_value) / (max_value - min_value)
    ratio = max(0.0, min(1.0, ratio))
    r = int(241 - ratio * 68)
    g = int(245 - ratio * 157)
    b = int(249 - ratio * 106)
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_heatmap(path, title, subtitle, x_labels, y_labels, values, width=780, height=560):
    left, right, top, bottom = 112, 42, 88, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    cell_w = plot_w / len(x_labels)
    cell_h = plot_h / len(y_labels)
    all_values = [value for value in values.values() if value is not None]
    if not all_values:
        return
    min_value, max_value = min(all_values), max(all_values)
    elements = title_block(title, subtitle)
    for xi, x_label in enumerate(x_labels):
        x = left + xi * cell_w
        elements.append(f'<text x="{x + cell_w / 2:.1f}" y="{top - 14}" text-anchor="middle" class="label">{esc(x_label)}</text>')
    for yi, y_label in enumerate(y_labels):
        y = top + yi * cell_h
        elements.append(f'<text x="{left - 12}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end" class="label">{esc(y_label)}</text>')
    for yi, y_label in enumerate(y_labels):
        for xi, x_label in enumerate(x_labels):
            value = values.get((x_label, y_label))
            x = left + xi * cell_w
            y = top + yi * cell_h
            fill = heat_color(value, min_value, max_value) if value is not None else "#f3f4f6"
            elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            if value is not None:
                text_color = "#ffffff" if value > (min_value + max_value) / 2 else "#111827"
                elements.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" text-anchor="middle" font-size="13" fill="{text_color}">{esc(format_number(value))}</text>'
                )
    elements.append(f'<text x="{left + plot_w / 2}" y="{height - 26}" text-anchor="middle" class="label">{esc("hidden dimension")}</text>')
    elements.append(
        f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" class="label">{esc("layers")}</text>'
    )
    write_svg(path, width, height, elements)


def best_record(records, dataset, implementation, section=None, hidden_dim=None, num_layers=None):
    candidates = [
        r
        for r in records
        if r.get("dataset") == dataset
        and r.get("implementation") == implementation
        and not is_excluded(r)
    ]
    if section is not None:
        candidates = [r for r in candidates if r.get("section") == section]
    if hidden_dim is not None:
        candidates = [r for r in candidates if r.get("hidden_dim") == hidden_dim]
    if num_layers is not None:
        candidates = [r for r in candidates if r.get("num_layers") == num_layers]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.get("inference_ms", float("inf")))


def plot_architecture(records, out_dir):
    for dataset in MAIN_DATASETS:
        labels, values, colors = [], [], []
        for implementation in ARCH_IMPLEMENTATIONS:
            record = best_record(
                records,
                dataset,
                implementation,
                section="Confronto architetturale",
                hidden_dim=64,
                num_layers=2,
            )
            if not record:
                continue
            labels.append(implementation)
            values.append(float(record["inference_ms"]))
            colors.append(COLORS.get(implementation, "#6b7280"))
        draw_bar_chart(
            out_dir / f"01_architectural_comparison_{slug(dataset)}.svg",
            f"Architectural comparison - {dataset}",
            "Inference time, H=64, L=2. Lower is better. FP16 is plotted separately.",
            labels,
            values,
            colors,
            "inference time (ms, log scale)",
            log_scale=True,
        )


def plot_speedup(records, out_dir):
    groups = []
    series = (
        "cpu-vertex-parallel",
        "cpu-edge-parallel",
        "cpu-message-batching",
        "cuda-vertex-improved",
        "cuda-edge-parallel-improved",
        "cuda-message-batching-improved",
    )
    values = {}
    for dataset in MAIN_DATASETS:
        baseline = best_record(
            records,
            dataset,
            "sequential",
            section="Confronto architetturale",
            hidden_dim=64,
            num_layers=2,
        )
        if not baseline:
            continue
        groups.append(dataset)
        values[dataset] = {}
        base_time = float(baseline["inference_ms"])
        for implementation in series:
            record = best_record(
                records,
                dataset,
                implementation,
                section="Confronto architetturale",
                hidden_dim=64,
                num_layers=2,
            )
            if record:
                values[dataset][implementation] = base_time / float(record["inference_ms"])
    draw_grouped_bars(
        out_dir / "02_speedup_vs_sequential.svg",
        "Speedup vs sequential",
        "H=64, L=2. Higher is better.",
        groups,
        series,
        values,
        "speedup",
    )


def plot_scalability(records, out_dir):
    series_points = {}
    for implementation in SCALABILITY_IMPLEMENTATIONS:
        points = []
        for record in records:
            if (
                record.get("implementation") == implementation
                and record.get("dataset", "").startswith("ErdosRenyi_")
                and record.get("hidden_dim") == 64
                and record.get("num_layers") == 2
                and not is_excluded(record)
            ):
                if record.get("section") not in {"Scalabilità", "Scalabilità sulla dimensione del grafo"}:
                    continue
                points.append((int(record["nodes"]), float(record["inference_ms"])))
        if points:
            by_node = {}
            for node_count, time_ms in points:
                by_node[node_count] = min(time_ms, by_node.get(node_count, float("inf")))
            series_points[implementation] = sorted(by_node.items())
    draw_line_chart(
        out_dir / "03_scalability_erdosrenyi.svg",
        "Scalability on Erdos-Renyi graphs",
        "Inference time as graph size grows, H=64, L=2.",
        series_points,
        "nodes",
        "inference time (ms)",
    )


def plot_topology(records, out_dir):
    groups = ("ErdosRenyi_100k", "BarabasiAlbert_100k", "WattsStrogatz_100k")
    values = {}
    for dataset in groups:
        values[dataset] = {}
        for implementation in TOPOLOGY_IMPLEMENTATIONS:
            record = best_record(
                records,
                dataset,
                implementation,
                section="Analisi topologica e skewed degree distribution",
                hidden_dim=64,
                num_layers=2,
            )
            if record:
                values[dataset][implementation] = float(record["inference_ms"])
    draw_grouped_bars(
        out_dir / "04_topology_impact.svg",
        "Topology impact",
        "Same order of magnitude in nodes/edges, H=64, L=2. Lower is better.",
        groups,
        TOPOLOGY_IMPLEMENTATIONS,
        values,
        "inference time (ms)",
    )


def plot_hyperparameters(records, out_dir):
    for dataset in ("ogbn-arxiv", "BarabasiAlbert_100k"):
        for implementation in ("cuda-edge-parallel-improved", "cpu-vertex-parallel"):
            x_labels = ("H=64", "H=128", "H=256")
            y_labels = ("L=2", "L=4", "L=8")
            values = {}
            for h in (64, 128, 256):
                for layers in (2, 4, 8):
                    record = best_record(
                        records,
                        dataset,
                        implementation,
                        section="Impatto della profondità e dimensione nascosta sul modello",
                        hidden_dim=h,
                        num_layers=layers,
                    )
                    values[(f"H={h}", f"L={layers}")] = (
                        float(record["inference_ms"]) if record else None
                    )
            draw_heatmap(
                out_dir / f"05_hyperparameters_{slug(implementation)}_{slug(dataset)}.svg",
                f"Hidden dimension and depth - {implementation}",
                f"{dataset}. Cell values are inference time in ms.",
                x_labels,
                y_labels,
                values,
            )


def plot_memory(records, out_dir):
    dataset = "ogbn-arxiv"
    labels, values, colors = [], [], []
    implementations = (
        "sequential",
        "cpu-vertex-parallel",
        "cpu-edge-parallel",
        "cpu-message-batching",
        "cuda-vertex-improved",
        "cuda-edge-parallel-improved",
        "cuda-message-batching-improved",
        "cuda-vertex-compressed-fp16",
    )
    for implementation in implementations:
        record = best_record(
            records,
            dataset,
            implementation,
            section="Confronto architetturale",
            hidden_dim=64,
            num_layers=2,
        )
        if not record:
            continue
        labels.append(implementation)
        values.append(float(record["estimated_total_memory_bytes"]) / 1_000_000.0)
        colors.append(COLORS.get(implementation, "#6b7280"))
    draw_bar_chart(
        out_dir / "06_memory_ogbn_arxiv.svg",
        "Estimated memory footprint - ogbn-arxiv",
        "H=64, L=2. FP16 is included only as memory/compression comparison.",
        labels,
        values,
        colors,
        "estimated memory (MB)",
        log_scale=False,
        notes=["FP16 changes prediction_checksum and should not be treated as FP32-equivalent."],
    )


def plot_gpu_profiling(records, out_dir):
    dataset = "ogbn-arxiv"
    labels, sm_values, sm_colors = [], [], []
    ipc_values, ipc_colors = [], []
    for implementation in GPU_PROFILE_IMPLEMENTATIONS:
        candidates = [
            record
            for record in records
            if record.get("section") == "Confronto architetturale"
            and record.get("dataset") == dataset
            and record.get("implementation") == implementation
            and record.get("hidden_dim") == 64
            and record.get("num_layers") == 2
        ]
        if not candidates:
            continue
        record = candidates[0]
        labels.append(implementation)
        sm_values.append(float(record["SM Throughput"]))
        ipc_values.append(float(record["IPC (per SM)"]))
        sm_colors.append(COLORS.get(implementation, "#6b7280"))
        ipc_colors.append(COLORS.get(implementation, "#6b7280"))
    draw_bar_chart(
        out_dir / "07_gpu_profiling_sm_throughput_ogbn_arxiv.svg",
        "GPU profiling - SM throughput",
        "ogbn-arxiv, H=64, L=2. Higher is better.",
        labels,
        sm_values,
        sm_colors,
        "SM throughput (%)",
    )
    draw_bar_chart(
        out_dir / "08_gpu_profiling_ipc_ogbn_arxiv.svg",
        "GPU profiling - IPC per SM",
        "ogbn-arxiv, H=64, L=2. Higher is better.",
        labels,
        ipc_values,
        ipc_colors,
        "IPC per SM",
    )


def write_exclusion_report(path, benchmark_records):
    excluded = [(record, is_excluded(record)) for record in benchmark_records if is_excluded(record)]
    lines = [
        "# Excluded benchmark points",
        "",
        "These points are intentionally omitted from plots that would otherwise use them.",
        "",
    ]
    if not excluded:
        lines.append("No points excluded.")
    else:
        for record, reason in excluded:
            lines.append(
                "- "
                + f"{record.get('implementation')} / {record.get('dataset')} "
                + f"H={record.get('hidden_dim')} L={record.get('num_layers')} "
                + f"inference_ms={record.get('inference_ms')} "
                + f"prediction_checksum={record.get('prediction_checksum')} "
                + f"({reason}; source {record.get('source_file')}:{record.get('source_line')})"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results" / "plots")
    args = parser.parse_args()

    benchmark_records = parse_benchmark_results(args.root)
    profiling_records = parse_profiling_results(args.root)
    if not benchmark_records:
        parser.error("no benchmark_results.txt files found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_architecture(benchmark_records, args.output_dir)
    plot_speedup(benchmark_records, args.output_dir)
    plot_scalability(benchmark_records, args.output_dir)
    plot_topology(benchmark_records, args.output_dir)
    plot_hyperparameters(benchmark_records, args.output_dir)
    plot_memory(benchmark_records, args.output_dir)
    plot_gpu_profiling(profiling_records, args.output_dir)
    write_exclusion_report(args.output_dir / "excluded_points.md", benchmark_records)

    generated = sorted(args.output_dir.glob("*.svg"))
    print(f"Generated {len(generated)} SVG plots in {args.output_dir}")
    for path in generated:
        print(f"  {path.relative_to(args.root)}")
    print(f"  {(args.output_dir / 'excluded_points.md').relative_to(args.root)}")


if __name__ == "__main__":
    main()
