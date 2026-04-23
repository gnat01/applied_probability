# Typos as a Point Process

The current Markovian model says:

```text
hidden tiredness creates bursts of typos
```

A point-process view says:

```text
typos are events in page-time
```

Instead of modeling every page as independent, or introducing a hidden fresh /
tired state, we can model the typo events themselves as a stochastic process.

## Page-Time Events

Let:

```text
X_k = 1 if page k has a typo
X_k = 0 otherwise
```

The independent model uses something like:

```text
P(X_k = 1) = p_k
```

The point-process model makes the probability depend on the event history:

```text
P(X_k = 1 | X_1, ..., X_{k-1})
```

So a typo on page `k` can change the probability of typos on later pages.

## Hawkes-Like Model

A natural discrete-page Hawkes-style model is:

```text
lambda_k = mu + sum_{i < k} alpha exp(-beta (k - i)) X_i
```

where:

```text
mu    = baseline typo intensity
alpha = excitation size after a typo
beta  = decay rate of that excitation
```

Then convert intensity into a page typo probability:

```text
P(X_k = 1 | history) = 1 - exp(-lambda_k)
```

This keeps the probability between `0` and `1`.

## Interpretation

If a typo happens, the future typo intensity jumps:

```text
typo -> higher near-future typo probability
```

Then that effect decays over later pages:

```text
near pages: stronger effect
far pages: weaker effect
```

This creates bursts directly from the observed typo process.

## Contrast with the Markovian Tired-Typist Model

The two models can produce visually similar clusters, but the causal stories
are different.

Markov tired typist:

```text
hidden tired state persists
tired state raises typo probability
typos are evidence of tiredness
```

Hawkes-style point process:

```text
typo event raises future typo intensity
intensity decays with page distance
typos excite more typos
```

So:

```text
Markov model: burstiness comes from latent state persistence
Hawkes model: burstiness comes from self-excitation
```

## What Would Be Worth Comparing

If we add this as a third model, useful comparisons would be:

```text
independent Bernoulli baseline
hidden Markov tiredness model
self-exciting Hawkes-like model
```

Good plots:

```text
cumulative P(at least one typo by page n)
sample event rasters
inter-typo gap distribution
burst size distribution
crossover page n* by probability threshold
```

The inter-typo gap distribution may be the sharpest diagnostic:

```text
independent model: roughly geometric gaps
Markov model: many short gaps during tired runs, longer fresh gaps
Hawkes model: many very short gaps after events, with decaying excitation
```

## Why This Is Appealing

The point-process framing treats typos as the primary objects, not just page
labels. That makes it natural to ask questions like:

```text
How clustered are typos?
How long do quiet periods last?
How quickly does typo risk decay after a typo?
What is the expected burst size?
```

For the current toy project, this would make a clean third chapter after the
independent model and the hidden-state Markov model.

