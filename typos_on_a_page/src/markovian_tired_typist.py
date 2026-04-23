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

# Hidden Markov model states:
#   F = fresh typist
#   T = tired typist
#
# The tired state has a higher page typo probability and tends to persist.
P_TYPO_FRESH = 0.01
P_TYPO_TIRED = 0.12
P_FRESH_TO_TIRED = 0.03

# Initial state distribution: mostly fresh at page 1.
INITIAL_FRESH_PROB = 0.95

MAX_PAGES = 250
THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Sweep the tired-state persistence parameter:
#   a = P(tired on page k+1 | tired on page k)
TIRED_PERSISTENCE_VALUES = np.linspace(0.10, 0.98, 89)

# Scenarios make the plots more interpretable than changing one parameter only.
SCENARIOS = [
    {
        "name": "mild",
        "p_fresh": 0.01,
        "p_tired": 0.06,
        "p_fresh_to_tired": 0.02,
        "initial_tired": 0.02,
    },
    {
        "name": "baseline",
        "p_fresh": P_TYPO_FRESH,
        "p_tired": P_TYPO_TIRED,
        "p_fresh_to_tired": P_FRESH_TO_TIRED,
        "initial_tired": 1.0 - INITIAL_FRESH_PROB,
    },
    {
        "name": "fragile",
        "p_fresh": 0.01,
        "p_tired": 0.18,
        "p_fresh_to_tired": 0.06,
        "initial_tired": 0.08,
    },
]


# ============================================================
# HIDDEN MARKOV MODEL
# ============================================================
def transition_matrix(p_fresh_to_tired, p_tired_to_tired):
    """
    Row-stochastic transition matrix for states [fresh, tired].

    rows: current state
    cols: next state
    """
    return np.array(
        [
            [1.0 - p_fresh_to_tired, p_fresh_to_tired],
            [1.0 - p_tired_to_tired, p_tired_to_tired],
        ],
        dtype=float,
    )


def no_typo_survival_curve(
    max_pages,
    p_fresh,
    p_tired,
    p_fresh_to_tired,
    p_tired_to_tired,
    initial_tired,
):
    """
    Return S[n-1] = P(no typo in pages 1..n).

    The vector tracks joint probability:
        [P(no typo so far and fresh now), P(no typo so far and tired now)]

    For each page:
        1. emit a no-typo observation conditional on current state;
        2. transition the hidden typist state to the next page.
    """
    state = np.array([1.0 - initial_tired, initial_tired], dtype=float)
    no_typo_emission = np.array([1.0 - p_fresh, 1.0 - p_tired], dtype=float)
    transition = transition_matrix(p_fresh_to_tired, p_tired_to_tired)

    survival = np.empty(max_pages, dtype=float)

    for page in range(max_pages):
        state = state * no_typo_emission
        survival[page] = np.sum(state)
        state = state @ transition

    return survival


def typo_probability_curve(*args, **kwargs):
    return 1.0 - no_typo_survival_curve(*args, **kwargs)


def crossover_page(prob_at_least_one_typo, threshold):
    """
    Return the smallest page count n with P(at least one typo by n) > threshold.
    """
    crossings = np.flatnonzero(prob_at_least_one_typo > threshold)
    if len(crossings) == 0:
        return len(prob_at_least_one_typo)
    return int(crossings[0] + 1)


def expected_run_lengths(p_fresh_to_tired, p_tired_to_tired):
    """
    Expected state dwell times in pages for a two-state Markov chain.
    """
    fresh_run = np.inf if p_fresh_to_tired == 0 else 1.0 / p_fresh_to_tired
    tired_exit = 1.0 - p_tired_to_tired
    tired_run = np.inf if tired_exit == 0 else 1.0 / tired_exit
    return fresh_run, tired_run


def stationary_tired_probability(p_fresh_to_tired, p_tired_to_tired):
    denominator = p_fresh_to_tired + (1.0 - p_tired_to_tired)
    if denominator == 0:
        return np.nan
    return p_fresh_to_tired / denominator


def simulate_typo_sequence(
    pages,
    p_fresh,
    p_tired,
    p_fresh_to_tired,
    p_tired_to_tired,
    initial_tired,
    rng,
):
    """
    Simulate hidden typist states and observed typo indicators.
    """
    tired = rng.random() < initial_tired
    states = np.empty(pages, dtype=bool)
    typos = np.empty(pages, dtype=bool)

    for page in range(pages):
        states[page] = tired
        typo_probability = p_tired if tired else p_fresh
        typos[page] = rng.random() < typo_probability

        if tired:
            tired = rng.random() < p_tired_to_tired
        else:
            tired = rng.random() < p_fresh_to_tired

    return states, typos


# ============================================================
# TABLES
# ============================================================
def compute_crossover_grid(scenario):
    rows = []
    for persistence in TIRED_PERSISTENCE_VALUES:
        curve = typo_probability_curve(
            MAX_PAGES,
            scenario["p_fresh"],
            scenario["p_tired"],
            scenario["p_fresh_to_tired"],
            persistence,
            scenario["initial_tired"],
        )

        fresh_run, tired_run = expected_run_lengths(
            scenario["p_fresh_to_tired"],
            persistence,
        )

        row = {
            "scenario": scenario["name"],
            "p_fresh": scenario["p_fresh"],
            "p_tired": scenario["p_tired"],
            "p_fresh_to_tired": scenario["p_fresh_to_tired"],
            "p_tired_to_tired": persistence,
            "initial_tired": scenario["initial_tired"],
            "stationary_tired": stationary_tired_probability(
                scenario["p_fresh_to_tired"],
                persistence,
            ),
            "expected_fresh_run": fresh_run,
            "expected_tired_run": tired_run,
        }
        for threshold in THRESHOLDS:
            row[f"threshold_{threshold:.2f}"] = crossover_page(curve, threshold)
        rows.append(row)

    return rows


def write_crossover_csv(rows, filepath):
    fieldnames = [
        "scenario",
        "p_fresh",
        "p_tired",
        "p_fresh_to_tired",
        "p_tired_to_tired",
        "initial_tired",
        "stationary_tired",
        "expected_fresh_run",
        "expected_tired_run",
    ] + [f"threshold_{threshold:.2f}" for threshold in THRESHOLDS]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# PLOTS
# ============================================================
def plot_probability_curves(scenario, filepath):
    pages = np.arange(1, MAX_PAGES + 1)
    persistence_values = [0.10, 0.50, 0.80, 0.92, 0.98]

    plt.figure(figsize=(11, 7))

    for persistence in persistence_values:
        curve = typo_probability_curve(
            MAX_PAGES,
            scenario["p_fresh"],
            scenario["p_tired"],
            scenario["p_fresh_to_tired"],
            persistence,
            scenario["initial_tired"],
        )
        _, tired_run = expected_run_lengths(
            scenario["p_fresh_to_tired"],
            persistence,
        )
        label = f"P(T->T)={persistence:.2f}, E[tired run]={tired_run:.1f}"
        plt.plot(pages, curve, linewidth=2.0, label=label)

    for threshold in THRESHOLDS:
        plt.axhline(threshold, color="black", alpha=0.12, linewidth=1.0)

    plt.xlabel("pages read")
    plt.ylabel("P(at least one typo)")
    plt.title(
        "Tired-typist Markov model: cumulative typo probability\n"
        f"scenario={scenario['name']}, "
        f"p_fresh={scenario['p_fresh']:.2f}, "
        f"p_tired={scenario['p_tired']:.2f}, "
        f"P(F->T)={scenario['p_fresh_to_tired']:.2f}"
    )
    plt.ylim(0.0, 1.01)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_crossover_heatmap(rows, scenario_name, filepath):
    scenario_rows = [row for row in rows if row["scenario"] == scenario_name]
    persistence = np.array([row["p_tired_to_tired"] for row in scenario_rows])
    heatmap = np.array(
        [
            [row[f"threshold_{threshold:.2f}"] for row in scenario_rows]
            for threshold in THRESHOLDS
        ],
        dtype=float,
    )

    plt.figure(figsize=(11, 6))
    image = plt.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[
            persistence.min(),
            persistence.max(),
            THRESHOLDS[0],
            THRESHOLDS[-1],
        ],
        cmap="viridis_r",
    )
    cbar = plt.colorbar(image)
    cbar.set_label("crossover page n*")

    plt.xlabel("tired-state persistence P(T at k+1 | T at k)")
    plt.ylabel("probability threshold")
    plt.title(f"Crossover page heatmap for scenario={scenario_name}")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_crossover_lines(rows, scenario_name, filepath):
    scenario_rows = [row for row in rows if row["scenario"] == scenario_name]
    persistence = np.array([row["p_tired_to_tired"] for row in scenario_rows])

    plt.figure(figsize=(11, 7))

    for threshold in THRESHOLDS:
        y = np.array(
            [row[f"threshold_{threshold:.2f}"] for row in scenario_rows],
            dtype=float,
        )
        plt.plot(
            persistence,
            y,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=f"threshold={threshold:.2f}",
        )

    plt.xlabel("tired-state persistence P(T at k+1 | T at k)")
    plt.ylabel("crossover page n*")
    plt.title(f"Crossover page vs tiredness persistence for scenario={scenario_name}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_stationary_summary(filepath):
    plt.figure(figsize=(11, 7))
    persistence = TIRED_PERSISTENCE_VALUES

    for scenario in SCENARIOS:
        stationary = np.array(
            [
                stationary_tired_probability(
                    scenario["p_fresh_to_tired"],
                    tired_persistence,
                )
                for tired_persistence in persistence
            ],
            dtype=float,
        )
        effective_page_typo_rate = (
            (1.0 - stationary) * scenario["p_fresh"]
            + stationary * scenario["p_tired"]
        )
        plt.plot(
            persistence,
            effective_page_typo_rate,
            linewidth=2.0,
            label=scenario["name"],
        )

    plt.xlabel("tired-state persistence P(T at k+1 | T at k)")
    plt.ylabel("stationary mean page typo probability")
    plt.title("Long-run typo rate induced by persistent tiredness")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_sample_bursts(scenario, filepath):
    pages = 180
    persistence_values = [0.10, 0.50, 0.80, 0.92, 0.98]
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(
        len(persistence_values),
        1,
        figsize=(12, 8),
        sharex=True,
    )

    for axis, persistence in zip(axes, persistence_values):
        states, typos = simulate_typo_sequence(
            pages,
            scenario["p_fresh"],
            scenario["p_tired"],
            scenario["p_fresh_to_tired"],
            persistence,
            scenario["initial_tired"],
            rng,
        )

        page_numbers = np.arange(1, pages + 1)
        tired_pages = page_numbers[states]
        typo_pages = page_numbers[typos]

        axis.eventplot(
            tired_pages,
            lineoffsets=0.15,
            linelengths=0.22,
            linewidths=1.4,
            colors="#f1a340",
        )
        axis.eventplot(
            typo_pages,
            lineoffsets=0.65,
            linelengths=0.55,
            linewidths=2.2,
            colors="#d7191c",
        )

        _, tired_run = expected_run_lengths(
            scenario["p_fresh_to_tired"],
            persistence,
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([0.15, 0.65])
        axis.set_yticklabels(["tired", "typo"])
        axis.set_ylabel(f"T->T={persistence:.2f}\nE={tired_run:.1f}")
        axis.grid(True, axis="x", alpha=0.2)

    axes[-1].set_xlabel("page")
    fig.suptitle(
        "Sample burst patterns from the tired-typist Markov model\n"
        f"scenario={scenario['name']}, red marks are typo pages",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots and CSVs for the Markovian tired-typist model."
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
    return parser.parse_args()


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for scenario in SCENARIOS:
        all_rows.extend(compute_crossover_grid(scenario))
        plot_probability_curves(
            scenario,
            output_dir / f"markovian_probability_curves_{scenario['name']}.png",
        )
        plot_sample_bursts(
            scenario,
            output_dir / f"markovian_sample_bursts_{scenario['name']}.png",
        )

    write_crossover_csv(all_rows, output_dir / "markovian_crossover_values.csv")

    for scenario in SCENARIOS:
        plot_crossover_lines(
            all_rows,
            scenario["name"],
            output_dir / f"markovian_crossover_lines_{scenario['name']}.png",
        )
        plot_crossover_heatmap(
            all_rows,
            scenario["name"],
            output_dir / f"markovian_crossover_heatmap_{scenario['name']}.png",
        )

    plot_stationary_summary(output_dir / "markovian_stationary_typo_rate.png")

    print("Wrote Markovian typo model outputs to:")
    print(f"  {output_dir.resolve()}")


if __name__ == "__main__":
    main(parse_args().output_dir)
