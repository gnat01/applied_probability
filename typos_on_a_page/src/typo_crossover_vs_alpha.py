import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"

# alphas to evaluate
alphas = np.arange(0.0001, 0.0011, 0.0001)

max_pages = 1000
threshold_log = np.log(0.5)

def find_crossover(alpha):
    log_prob_no_typo = 0.0

    for n in range(1, max_pages + 1):
        p_m = 0.01 + alpha * n

        # sanity check
        if p_m >= 1:
            return n

        log_prob_no_typo += np.log(1 - p_m)

        if log_prob_no_typo < threshold_log:
            return n

    return max_pages


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the independent typo crossover for threshold 0.5."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for generated PNG files. "
            "Defaults to typos_on_a_page/outputs/."
        ),
    )
    return parser.parse_args()


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # compute crossover points
    crossover_points = [find_crossover(alpha) for alpha in alphas]

    # plot
    plt.figure()
    plt.plot(alphas, crossover_points, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("n* (pages for P(≥1 typo) > 0.5)")
    plt.title("Crossover point vs increasing typo probability slope")
    plt.grid()

    output_path = output_dir / "typo_crossover_vs_alpha.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main(parse_args().output_dir)
