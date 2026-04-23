import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============================================================
# CONFIG
# ============================================================
# Dense alpha grid for stable fitting
ALPHAS = np.linspace(0.0001, 0.0010, 100)

# Probability thresholds
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

MAX_PAGES = 1000

# Output files
VALUES_CSV = "crossover_values.csv"
FITS_CSV = "fit_parameters.csv"
PLOT_DATA_ONLY = "crossover_curves.png"
PLOT_WITH_FITS = "crossover_curves_with_fits.png"


# ============================================================
# CORE PROBABILITY / CROSSOVER COMPUTATION
# ============================================================
def find_crossover(alpha: float, threshold: float, max_pages: int = MAX_PAGES) -> int:
    """
    Return the smallest n such that:
        P(at least one typo in pages 1..n) > threshold

    where page m has typo probability:
        p_m = 0.01 + alpha * m

    and page typo events are independent.
    """
    log_prob_no_typo = 0.0
    threshold_log = np.log(1.0 - threshold)

    for n in range(1, max_pages + 1):
        p_m = 0.01 + alpha * n

        # If p_m >= 1, then a typo is certain on this page.
        if p_m >= 1.0:
            return n

        log_prob_no_typo += np.log1p(-p_m)

        if log_prob_no_typo < threshold_log:
            return n

    return max_pages


def compute_crossover_table(alphas, thresholds):
    """
    Returns:
        results: dict mapping threshold -> np.array of crossover values
    """
    results = {}
    for threshold in thresholds:
        results[threshold] = np.array(
            [find_crossover(alpha, threshold) for alpha in alphas],
            dtype=float
        )
    return results


# ============================================================
# FITTING
# ============================================================
# Robust choice:
#   n*(alpha) = A * alpha^(-B)
#
# This is still a power law, just without the badly identified +C term.
def power_law_no_offset(alpha, A, B):
    return A * np.power(alpha, -B)


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def fit_single_curve(alphas, y_vals):
    """
    Fit y = A * alpha^(-B)
    Returns dict with A, B, R2, RMSE.
    """
    x = np.asarray(alphas, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    # Log-log linear regression for a stable initial guess:
    # log y = log A - B log x
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    A0 = np.exp(intercept)
    B0 = -slope

    # Nonlinear refinement with bounds
    params, _ = curve_fit(
        power_law_no_offset,
        x,
        y,
        p0=[A0, B0],
        bounds=([0.0, 0.0], [np.inf, 5.0]),
        maxfev=50000
    )

    A_hat, B_hat = params
    y_hat = power_law_no_offset(x, A_hat, B_hat)
    r2 = compute_r2(y, y_hat)
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    return {
        "A": float(A_hat),
        "B": float(B_hat),
        "R2": float(r2),
        "RMSE": rmse,
        "y_hat": y_hat,
    }


def fit_all_curves(alphas, results):
    """
    Returns:
        fit_results: dict mapping threshold -> fit summary dict
    """
    fit_results = {}
    for threshold, y_vals in results.items():
        fit_results[threshold] = fit_single_curve(alphas, y_vals)
    return fit_results


# ============================================================
# CSV WRITERS
# ============================================================
def write_crossover_csv(filepath, alphas, thresholds, results):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha"] + [f"threshold_{t:.1f}" for t in thresholds])

        for i, alpha in enumerate(alphas):
            row = [f"{alpha:.7f}"] + [int(results[t][i]) for t in thresholds]
            writer.writerow(row)


def write_fit_csv(filepath, thresholds, fit_results):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "A", "B", "R2", "RMSE"])

        for threshold in thresholds:
            fr = fit_results[threshold]
            writer.writerow([
                f"{threshold:.1f}",
                f"{fr['A']:.10f}",
                f"{fr['B']:.10f}",
                f"{fr['R2']:.10f}",
                f"{fr['RMSE']:.10f}",
            ])


# ============================================================
# PLOTTING
# ============================================================
def plot_data_only(alphas, thresholds, results, filepath):
    plt.figure(figsize=(10, 6))

    for threshold in thresholds:
        plt.plot(
            alphas,
            results[threshold],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"threshold = {threshold:.1f}"
        )

    plt.xlabel("alpha")
    plt.ylabel("n* (smallest pages such that P(≥1 typo) > threshold)")
    plt.title("Crossover point n* vs alpha")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_data_and_fits(alphas, thresholds, results, fit_results, filepath):
    plt.figure(figsize=(11, 7))
    alpha_fine = np.linspace(np.min(alphas), np.max(alphas), 500)

    for threshold in thresholds:
        y = results[threshold]
        fr = fit_results[threshold]
        y_fit_fine = power_law_no_offset(alpha_fine, fr["A"], fr["B"])

        plt.plot(
            alphas,
            y,
            marker="o",
            linestyle="None",
            markersize=3,
            label=f"data t={threshold:.1f}"
        )
        plt.plot(
            alpha_fine,
            y_fit_fine,
            linewidth=1.6,
            label=f"fit t={threshold:.1f}"
        )

    plt.xlabel("alpha")
    plt.ylabel("n* (smallest pages such that P(≥1 typo) > threshold)")
    plt.title("Crossover point n* vs alpha with power-law fits")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


# ============================================================
# CONSOLE SUMMARY
# ============================================================
def print_summary(thresholds, fit_results):
    print("\nFitted model for each threshold:")
    print("n*(alpha) = A * alpha^(-B)\n")

    for threshold in thresholds:
        fr = fit_results[threshold]
        print(
            f"threshold={threshold:.1f}: "
            f"A={fr['A']:.8f}, "
            f"B={fr['B']:.8f}, "
            f"R^2={fr['R2']:.8f}, "
            f"RMSE={fr['RMSE']:.8f}"
        )


# ============================================================
# MAIN
# ============================================================
def main():
    results = compute_crossover_table(ALPHAS, THRESHOLDS)
    fit_results = fit_all_curves(ALPHAS, results)

    write_crossover_csv(VALUES_CSV, ALPHAS, THRESHOLDS, results)
    write_fit_csv(FITS_CSV, THRESHOLDS, fit_results)

    plot_data_only(ALPHAS, THRESHOLDS, results, PLOT_DATA_ONLY)
    plot_data_and_fits(ALPHAS, THRESHOLDS, results, fit_results, PLOT_WITH_FITS)

    print_summary(THRESHOLDS, fit_results)

    print("\nWrote files:")
    print(f"  - {VALUES_CSV}")
    print(f"  - {FITS_CSV}")
    print(f"  - {PLOT_DATA_ONLY}")
    print(f"  - {PLOT_WITH_FITS}")


if __name__ == "__main__":
    main()
