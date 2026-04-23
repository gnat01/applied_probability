# Hawkes-Like Typo Process

This is the third model in the typo-on-a-page toy project.

The three model stories are:

```text
independent model:
  every page has its own typo probability

hidden Markov model:
  a latent tired state persists across pages

Hawkes-like model:
  typo events directly raise near-future typo risk
```

## Discrete Page-Time Hawkes Model

Let:

```text
X_k = 1 if page k has at least one typo
X_k = 0 otherwise
```

The model keeps a page-level typo intensity:

```text
lambda_k = mu + excitation_k
```

The excitation term is updated from the previous page:

```text
excitation_k = decay * excitation_{k-1} + jump * X_{k-1}
```

Then intensity is converted into a probability:

```text
P(X_k = 1 | history) = 1 - exp(-lambda_k)
```

This is a discrete-time analogue of a self-exciting point process.

## Parameters

```text
mu    = baseline intensity
jump  = immediate increase in future intensity after a typo
decay = persistence of excitation from page to page
```

If a typo occurs, the next page's typo probability rises. If no further typo
occurs, the excitation decays geometrically.

## Stability Diagnostic

A useful rough diagnostic is the branching ratio:

```text
branching ratio = jump * decay / (1 - decay)
```

Values below `1` are the natural stable regime. Higher values mean each typo
can trigger many descendants, leading to long clusters.

This is only an approximation because the script uses Bernoulli page events
with probability:

```text
1 - exp(-lambda_k)
```

but it is still a good sanity check.

## Scenarios

The script uses three Hawkes-like scenarios:

```text
mild:
  mu = 0.012
  jump = 0.10
  decay = 0.45

baseline:
  mu = 0.014
  jump = 0.22
  decay = 0.58

volatile:
  mu = 0.010
  jump = 0.35
  decay = 0.72
```

The volatile case has the strongest self-excitation and should show the most
obvious bursts.

## What the Comparison Script Does

The comparison script generates:

```text
independent calibrated model
hidden Markov tired-typist model
baseline Hawkes-like model
```

The independent model is calibrated to the Markov model's long-run average page
typo probability. That makes it a useful non-bursty comparison point.

The script then compares:

```text
cumulative probability of at least one typo by page n
inter-typo gap distributions
burst size distributions
crossover page n* by threshold
sample Hawkes paths with intensity and event marks
```

## Important Interpretation

The cumulative probability plot is useful, but it is not the sharpest test of
burstiness. Many different models can reach similar values for:

```text
P(at least one typo by page n)
```

The better diagnostics are:

```text
inter-typo gaps
burst sizes
sample event rasters
```

Independent page typos tend to have roughly geometric gaps. Hawkes-like typos
should produce more very short gaps because a typo raises the near-future typo
probability.

## Why This Complements the Markov Model

The hidden Markov model says:

```text
typos cluster because the typist enters a persistent tired state
```

The Hawkes-like model says:

```text
typos cluster because typo events excite more typo events
```

In real applied work, either story could be a proxy for the same phenomenon.
For this toy problem, the distinction is useful because it shows two different
ways to generate burstiness:

```text
latent-state persistence
self-excitation
```

