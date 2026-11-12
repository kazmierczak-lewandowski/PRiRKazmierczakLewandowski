import csv
import os
import random
from multiprocessing import cpu_count

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import cp, tabu_search, tabu_search_cuda

SIZES = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
MAX_ITERATIONS = 50
TABU_TENURE = 25
SEED = 123
OUTPUT_DIR = "wykresy"

COLORS = {"seq": "blue", "par": "red", "cuda": "green"}
MARKERS = {"seq": "o", "par": "s", "cuda": "^"}
TEXT_PRIMARY = "black"
TEXT_SECONDARY = "gray"
GRID = "lightgray"
SURFACE = "white"


def generate_cities(n):
    random.seed(SEED)
    return [(random.randint(0, 1000), random.randint(0, 1000)) for _ in range(n)]


def run_benchmark():
    num_workers = cpu_count()
    rows = []
    for n in SIZES:
        cities = generate_cities(n)
        row = {"n": n, "moves": n * (n - 1) // 2}

        random.seed(SEED)
        _, row["seq_distance"], _, row["seq_time"] = tabu_search(
            cities, max_iterations=MAX_ITERATIONS, tabu_tenure=TABU_TENURE, num_workers=1
        )
        random.seed(SEED)
        _, row["par_distance"], _, row["par_time"] = tabu_search(
            cities, max_iterations=MAX_ITERATIONS, tabu_tenure=TABU_TENURE, num_workers=num_workers
        )
        random.seed(SEED)
        _, row["cuda_distance"], _, row["cuda_time"] = tabu_search_cuda(
            cities, max_iterations=MAX_ITERATIONS, tabu_tenure=TABU_TENURE
        )

        print(
            f"n={n:4d}  seq={row['seq_time']:8.3f}s  par={row['par_time']:7.3f}s  "
            f"cuda={row['cuda_time']:6.3f}s  dystans={row['seq_distance']:.2f}/"
            f"{row['par_distance']:.2f}/{row['cuda_distance']:.2f}"
        )
        rows.append(row)
    return rows, num_workers


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=13, pad=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_xticks(SIZES)


def plot_series(ax, sizes, values, key, label):
    ax.plot(
        sizes, values, color=COLORS[key], marker=MARKERS[key], linewidth=2,
        markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label,
    )
    ax.annotate(
        label, (sizes[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
        va="center", color=TEXT_PRIMARY, fontsize=10,
    )


def plot_times(rows, num_workers, path):
    sizes = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    plot_series(ax, sizes, [r["seq_time"] for r in rows], "seq", "Sekwencyjnie")
    plot_series(ax, sizes, [r["par_time"] for r in rows], "par", f"CPU {num_workers} proc.")
    plot_series(ax, sizes, [r["cuda_time"] for r in rows], "cuda", "CUDA")
    ax.set_yscale("log")
    style_axes(
        ax, f"Czas wykonania Tabu Search ({MAX_ITERATIONS} iteracji)",
        "Liczba miast", "Czas [s] (skala logarytmiczna)",
    )
    ax.set_xlim(sizes[0] - 5, sizes[-1] + 30)
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speedup(rows, num_workers, path):
    sizes = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    plot_series(ax, sizes, [r["seq_time"] / r["par_time"] for r in rows], "par", f"CPU {num_workers} proc.")
    plot_series(ax, sizes, [r["seq_time"] / r["cuda_time"] for r in rows], "cuda", "CUDA")
    ax.axhline(1, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.annotate("1x = wersja sekwencyjna", (sizes[-1], 1), xytext=(0, 6), textcoords="offset points",
                ha="right", color=TEXT_SECONDARY, fontsize=9)
    ax.set_yscale("log")
    style_axes(
        ax, "Przyspieszenie względem wersji sekwencyjnej",
        "Liczba miast", "Przyspieszenie [x] (skala logarytmiczna)",
    )
    ax.set_xlim(sizes[0] - 5, sizes[-1] + 30)
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows, num_workers = run_benchmark()
    save_csv(rows, os.path.join(OUTPUT_DIR, "wyniki.csv"))
    plot_times(rows, num_workers, os.path.join(OUTPUT_DIR, "czas_wykonania.png"))
    plot_speedup(rows, num_workers, os.path.join(OUTPUT_DIR, "przyspieszenie.png"))
    print(f"\nZapisano wykresy i wyniki w katalogu '{OUTPUT_DIR}/'")
