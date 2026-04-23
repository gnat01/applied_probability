import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIG
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"

MAX_PAGES = 250
DEFAULT_SIMULATIONS = 8000
THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Baseline hidden Markov tired-typist model used for comparison.
MARKOV_PARAMS = {
    "p_fresh": 0.01,
    "p_tired": 0.12,
    "p_fresh_to_tired": 0.03,
    "p_tired_to_tired": 0.92,
    "initial_tired": 0.05,
}

# Page-time Hawkes-like models. The recursion is:
#   excitation_k = decay * excitation_{k-1} + jump * X_{k-1}
#   lambda_k = baseline + excitation_k
#   P(X_k = 1 | history) = 1 - exp(-lambda_k)
HAWKES_SCENARIOS = [
    {
        "name": "mild",
        "baseline": 0.012,
        "jump": 0.10,
        "decay": 0.45,
    },
    {
        "name": "baseline",
        "baseline": 0.014,
        "jump": 0.22,
        "decay": 0.58,
    },
    {
        "name": "volatile",
        "baseline": 0.010,
        "jump": 0.35,
        "decay": 0.72,
    },
]


# ============================================================
# SHARED HELPERS
# ============================================================
def probability_from_intensity(intensity):
    return 1.0 - np.exp(-intensity)


def crossover_page(prob_at_least_one_typo, threshold):
    crossings = np.flatnonzero(prob_at_least_one_typo > threshold)
    if len(crossings) == 0:
        return len(prob_at_least_one_typo)
    return int(crossings[0] + 1)


def inter_typo_gaps(typos):
    typo_pages = np.flatnonzero(typos) + 1
    if len(typo_pages) < 2:
        return np.array([], dtype=int)
    return np.diff(typo_pages)


def burst_sizes(typos, max_gap=2):
    """
    Count clusters of typos where consecutive typo pages are at most max_gap apart.
    """
    typo_pages = np.flatnonzero(typos) + 1
    if len(typo_pages) == 0:
        return np.array([], dtype=int)

    sizes = []
    current_size = 1
    for gap in np.diff(typo_pages):
        if gap <= max_gap:
            current_size += 1
        else:
            sizes.append(current_size)
            current_size = 1
    sizes.append(current_size)
    return np.array(sizes, dtype=int)


def summarize_sample_paths(name, paths):
    typo_counts = paths.sum(axis=1)
    gaps = []
    bursts = []

    for path in paths:
        gaps.extend(inter_typo_gaps(path))
        bursts.extend(burst_sizes(path))

    gaps = np.array(gaps, dtype=int)
    bursts = np.array(bursts, dtype=int)

    return {
        "model": name,
        "mean_typos": float(np.mean(typo_counts)),
        "median_typos": float(np.median(typo_counts)),
        "prob_zero_typos": float(np.mean(typo_counts == 0)),
        "mean_gap": float(np.mean(gaps)) if len(gaps) else np.nan,
        "median_gap": float(np.median(gaps)) if len(gaps) else np.nan,
        "prob_gap_le_2": float(np.mean(gaps <= 2)) if len(gaps) else np.nan,
        "mean_burst_size": float(np.mean(bursts)) if len(bursts) else np.nan,
        "prob_burst_size_ge_2": (
            float(np.mean(bursts >= 2)) if len(bursts) else np.nan
        ),
    }


# ============================================================
# INDEPENDENT MODEL
# ============================================================
def independent_probability_curve(p_page, max_pages):
    pages = np.arange(1, max_pages + 1)
    return 1.0 - np.power(1.0 - p_page, pages)


def simulate_independent_paths(p_page, simulations, max_pages, rng):
    return rng.random((simulations, max_pages)) < p_page


# ============================================================
# MARKOVIAN TIRED-TYPIST MODEL
# ============================================================
def transition_matrix(p_fresh_to_tired, p_tired_to_tired):
    return np.array(
        [
            [1.0 - p_fresh_to_tired, p_fresh_to_tired],
            [1.0 - p_tired_to_tired, p_tired_to_tired],
        ],
        dtype=float,
    )


def markov_no_typo_survival_curve(params, max_pages):
    state = np.array([1.0 - params["initial_tired"], params["initial_tired"]])
    no_typo_emission = np.array([1.0 - params["p_fresh"], 1.0 - params["p_tired"]])
    transition = transition_matrix(
        params["p_fresh_to_tired"],
        params["p_tired_to_tired"],
    )

    survival = np.empty(max_pages, dtype=float)
    for page in range(max_pages):
        state = state * no_typo_emission
        survival[page] = np.sum(state)
        state = state @ transition
    return survival


def markov_probability_curve(params, max_pages):
    return 1.0 - markov_no_typo_survival_curve(params, max_pages)


def simulate_markov_paths(params, simulations, max_pages, rng):
    paths = np.empty((simulations, max_pages), dtype=bool)

    for simulation in range(simulations):
        tired = rng.random() < params["initial_tired"]
        for page in range(max_pages):
            typo_probability = params["p_tired"] if tired else params["p_fresh"]
            paths[simulation, page] = rng.random() < typo_probability

            if tired:
                tired = rng.random() < params["p_tired_to_tired"]
            else:
                tired = rng.random() < params["p_fresh_to_tired"]

    return paths


def stationary_markov_typo_probability(params):
    tired_probability = params["p_fresh_to_tired"] / (
        params["p_fresh_to_tired"] + 1.0 - params["p_tired_to_tired"]
    )
    return (
        (1.0 - tired_probability) * params["p_fresh"]
        + tired_probability * params["p_tired"]
    )


# ============================================================
# HAWKES-LIKE PAGE PROCESS
# ============================================================
def hawkes_branching_ratio(params):
    """
    Expected total excitation mass from one event in this discrete kernel.
    Values below 1 are the natural stable regime.
    """
    return params["jump"] * params["decay"] / (1.0 - params["decay"])


def approximate_hawkes_stationary_probability(params):
    ratio = hawkes_branching_ratio(params)
    if ratio >= 1.0:
        return np.nan
    mean_intensity = params["baseline"] / (1.0 - ratio)
    return probability_from_intensity(mean_intensity)


def simulate_hawkes_paths(params, simulations, max_pages, rng):
    paths = np.empty((simulations, max_pages), dtype=bool)
    probabilities = np.empty((simulations, max_pages), dtype=float)
    intensities = np.empty((simulations, max_pages), dtype=float)

    for simulation in range(simulations):
        excitation = 0.0
        for page in range(max_pages):
            intensity = params["baseline"] + excitation
            probability = probability_from_intensity(intensity)
            typo = rng.random() < probability

            paths[simulation, page] = typo
            probabilities[simulation, page] = probability
            intensities[simulation, page] = intensity

            excitation = params["decay"] * excitation + params["jump"] * typo

    return paths, probabilities, intensities


def empirical_cumulative_any_typo(paths):
    return np.mean(np.maximum.accumulate(paths, axis=1), axis=0)


# ============================================================
# CSV WRITERS
# ============================================================
def write_comparison_summary(filepath, rows):
    fieldnames = [
        "model",
        "mean_typos",
        "median_typos",
        "prob_zero_typos",
        "mean_gap",
        "median_gap",
        "prob_gap_le_2",
        "mean_burst_size",
        "prob_burst_size_ge_2",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_crossover_summary(filepath, rows):
    fieldnames = ["model"] + [f"threshold_{threshold:.2f}" for threshold in THRESHOLDS]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# PLOTS
# ============================================================
def plot_hawkes_sample_path(params, filepath, seed):
    rng = np.random.default_rng(seed)
    paths, probabilities, intensities = simulate_hawkes_paths(
        params,
        simulations=1,
        max_pages=MAX_PAGES,
        rng=rng,
    )
    typos = paths[0]
    pages = np.arange(1, MAX_PAGES + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(pages, intensities[0], color="#2c7fb8", linewidth=1.8)
    axes[0].set_ylabel("intensity")
    axes[0].set_title(
        "Hawkes-like typo process sample path\n"
        f"scenario={params['name']}, "
        f"mu={params['baseline']:.3f}, "
        f"jump={params['jump']:.2f}, "
        f"decay={params['decay']:.2f}"
    )

    axes[1].plot(pages, probabilities[0], color="#41ab5d", linewidth=1.8)
    axes[1].set_ylabel("page typo prob.")

    typo_pages = pages[typos]
    axes[2].eventplot(
        typo_pages,
        lineoffsets=0.5,
        linelengths=0.75,
        linewidths=2.2,
        colors="#d7191c",
    )
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_yticks([])
    axes[2].set_ylabel("typos")
    axes[2].set_xlabel("page")

    for axis in axes:
        axis.grid(True, axis="x", alpha=0.2)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_hawkes_scenario_probability_curves(hawkes_curves, filepath):
    pages = np.arange(1, MAX_PAGES + 1)
    plt.figure(figsize=(11, 7))

    for name, curve in hawkes_curves.items():
        plt.plot(pages, curve, linewidth=2.0, label=name)

    for threshold in THRESHOLDS:
        plt.axhline(threshold, color="black", alpha=0.12, linewidth=1.0)

    plt.xlabel("pages read")
    plt.ylabel("empirical P(at least one typo)")
    plt.title("Hawkes-like typo process: cumulative typo probability")
    plt.ylim(0.0, 1.01)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_model_comparison_curves(curves, filepath):
    pages = np.arange(1, MAX_PAGES + 1)
    plt.figure(figsize=(11, 7))

    for name, curve in curves.items():
        plt.plot(pages, curve, linewidth=2.0, label=name)

    for threshold in THRESHOLDS:
        plt.axhline(threshold, color="black", alpha=0.12, linewidth=1.0)

    plt.xlabel("pages read")
    plt.ylabel("P(at least one typo)")
    plt.title("Independent vs Markovian vs Hawkes-like typo accumulation")
    plt.ylim(0.0, 1.01)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_gap_histograms(model_paths, filepath):
    plt.figure(figsize=(11, 7))
    bins = np.arange(1, 41) - 0.5

    for name, paths in model_paths.items():
        gaps = []
        for path in paths:
            gaps.extend(inter_typo_gaps(path))
        if len(gaps) == 0:
            continue
        plt.hist(
            gaps,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=name,
        )

    plt.xlabel("inter-typo gap in pages")
    plt.ylabel("density")
    plt.title("Inter-typo gap distribution")
    plt.xlim(0.5, 40.5)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_burst_size_histograms(model_paths, filepath):
    plt.figure(figsize=(11, 7))
    bins = np.arange(1, 11) - 0.5

    for name, paths in model_paths.items():
        sizes = []
        for path in paths:
            sizes.extend(burst_sizes(path))
        if len(sizes) == 0:
            continue
        plt.hist(
            sizes,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=name,
        )

    plt.xlabel("burst size")
    plt.ylabel("density")
    plt.title("Burst size distribution, grouping typo gaps <= 2 pages")
    plt.xticks(range(1, 10))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_crossover_comparison(crossover_rows, filepath):
    labels = [row["model"] for row in crossover_rows]
    x = np.arange(len(labels))
    width = 0.15

    plt.figure(figsize=(12, 7))
    for idx, threshold in enumerate(THRESHOLDS):
        y = [row[f"threshold_{threshold:.2f}"] for row in crossover_rows]
        plt.bar(
            x + (idx - len(THRESHOLDS) / 2) * width + width / 2,
            y,
            width=width,
            label=f"threshold={threshold:.2f}",
        )

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("crossover page n*")
    plt.title("Crossover pages by model")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Hawkes-style typo process comparisons."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for generated CSV and PNG files. "
            "Defaults to typos_on_a_page/outputs/."
        ),
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help=f"Monte Carlo paths per simulated model. Default: {DEFAULT_SIMULATIONS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed for reproducible simulations.",
    )
    return parser.parse_args()


def main(output_dir, simulations, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    markov_stationary_p = stationary_markov_typo_probability(MARKOV_PARAMS)
    independent_curve = independent_probability_curve(markov_stationary_p, MAX_PAGES)
    independent_paths = simulate_independent_paths(
        markov_stationary_p,
        simulations,
        MAX_PAGES,
        rng,
    )

    markov_curve = markov_probability_curve(MARKOV_PARAMS, MAX_PAGES)
    markov_paths = simulate_markov_paths(MARKOV_PARAMS, simulations, MAX_PAGES, rng)

    hawkes_curves = {}
    hawkes_paths_by_name = {}
    baseline_hawkes_paths = None
    baseline_hawkes_curve = None

    for scenario in HAWKES_SCENARIOS:
        paths, _, _ = simulate_hawkes_paths(scenario, simulations, MAX_PAGES, rng)
        curve = empirical_cumulative_any_typo(paths)
        hawkes_curves[scenario["name"]] = curve
        hawkes_paths_by_name[scenario["name"]] = paths
        if scenario["name"] == "baseline":
            baseline_hawkes_paths = paths
            baseline_hawkes_curve = curve

        plot_hawkes_sample_path(
            scenario,
            output_dir / f"hawkes_sample_path_{scenario['name']}.png",
            seed + len(hawkes_paths_by_name),
        )

    comparison_curves = {
        "independent calibrated": independent_curve,
        "markov tired typist": markov_curve,
        "hawkes baseline": baseline_hawkes_curve,
    }
    comparison_paths = {
        "independent calibrated": independent_paths,
        "markov tired typist": markov_paths,
        "hawkes baseline": baseline_hawkes_paths,
    }

    crossover_rows = []
    for name, curve in comparison_curves.items():
        row = {"model": name}
        for threshold in THRESHOLDS:
            row[f"threshold_{threshold:.2f}"] = crossover_page(curve, threshold)
        crossover_rows.append(row)

    summary_rows = [
        summarize_sample_paths(name, paths)
        for name, paths in comparison_paths.items()
    ]

    write_comparison_summary(
        output_dir / "hawkes_comparison_summary.csv",
        summary_rows,
    )
    write_crossover_summary(
        output_dir / "hawkes_crossover_comparison.csv",
        crossover_rows,
    )

    plot_hawkes_scenario_probability_curves(
        hawkes_curves,
        output_dir / "hawkes_probability_curves.png",
    )
    plot_model_comparison_curves(
        comparison_curves,
        output_dir / "hawkes_model_comparison_curves.png",
    )
    plot_gap_histograms(
        comparison_paths,
        output_dir / "hawkes_inter_typo_gap_comparison.png",
    )
    plot_burst_size_histograms(
        comparison_paths,
        output_dir / "hawkes_burst_size_comparison.png",
    )
    plot_crossover_comparison(
        crossover_rows,
        output_dir / "hawkes_crossover_comparison.png",
    )

    print("Hawkes scenario stability diagnostics:")
    for scenario in HAWKES_SCENARIOS:
        print(
            f"  {scenario['name']}: "
            f"branching_ratio={hawkes_branching_ratio(scenario):.3f}, "
            f"approx_stationary_p="
            f"{approximate_hawkes_stationary_probability(scenario):.4f}"
        )

    print("\nWrote Hawkes typo process outputs to:")
    print(f"  {output_dir.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    main(args.output_dir, args.simulations, args.seed)
