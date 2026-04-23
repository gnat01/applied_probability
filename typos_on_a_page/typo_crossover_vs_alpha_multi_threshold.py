import numpy as np
import matplotlib.pyplot as plt

# alphas to evaluate
alphas = np.arange(0.0001, 0.0011, 0.0001)

# probability thresholds to evaluate
thresholds = np.arange(0.1, 0.9, 0.1)

max_pages = 1000

def find_crossover(alpha, threshold):
    """
    Return the smallest n such that:
        P(at least one typo in first n pages) > threshold
    where page m has typo probability:
        p_m = 0.01 + alpha * m
    and pages are independent.
    """
    log_prob_no_typo = 0.0
    threshold_log = np.log(1 - threshold)

    for n in range(1, max_pages + 1):
        p_m = 0.01 + alpha * n

        # If typo probability becomes 1 or more, then from this page onward
        # at least one typo is guaranteed.
        if p_m >= 1:
            return n

        log_prob_no_typo += np.log(1 - p_m)

        if log_prob_no_typo < threshold_log:
            return n

    return max_pages


# plot all thresholds on the same axes
plt.figure(figsize=(10, 6))

for threshold in thresholds:
    crossover_points = [find_crossover(alpha, threshold) for alpha in alphas]
    plt.plot(alphas, crossover_points, marker='o', label=f"threshold = {threshold:.1f}")

plt.xlabel("alpha")
plt.ylabel("n* (smallest pages such that P(≥1 typo) > threshold)")
plt.title("Crossover point n* vs alpha for multiple typo-probability thresholds")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("typo_crossover_vs_alpha_multi_threshold.png", dpi=300)
plt.show()
