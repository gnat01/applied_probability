# Markovian Typos

The original typo model treats pages independently. Page `m` has some typo
probability `p_m`, and the probability of no typo across the first `n` pages is
a product:

```text
P(no typo in pages 1..n) = product_m (1 - p_m)
```

That creates isolated typo events. It does not naturally create bursts.

The Markovian model introduces a hidden typist state:

```text
F = fresh typist
T = tired typist
```

The page typo probability depends on the hidden state:

```text
P(typo on page k | F) = p_F
P(typo on page k | T) = p_T
```

with:

```text
p_T > p_F
```

The state evolves as a Markov chain:

```text
P(T at k+1 | F at k) = b
P(F at k+1 | F at k) = 1 - b

P(T at k+1 | T at k) = a
P(F at k+1 | T at k) = 1 - a
```

Here:

```text
a = tired-state persistence
b = fresh-to-tired transition probability
```

Burstiness comes from choosing high `a`. If the typist becomes tired, tiredness
tends to persist for several pages, and typo probabilities stay elevated during
that run.

## Transition Matrix

Using state order `[F, T]`, the transition matrix is:

```text
        next F    next T
F     1 - b       b
T     1 - a       a
```

The expected length of a tired run is:

```text
E[tired run] = 1 / (1 - a)
```

So:

```text
a = 0.50  -> E[tired run] = 2 pages
a = 0.80  -> E[tired run] = 5 pages
a = 0.92  -> E[tired run] = 12.5 pages
a = 0.98  -> E[tired run] = 50 pages
```

That is the main burst-control knob.

## Computing No-Typo Survival

The scalar product from the independent model is replaced by a two-entry
probability vector:

```text
v_k = [
  P(no typo so far and typist is fresh at page k),
  P(no typo so far and typist is tired at page k)
]
```

At each page:

1. Multiply by the no-typo emission probabilities:

```text
[1 - p_F, 1 - p_T]
```

2. Sum the vector to get:

```text
P(no typo in pages 1..k)
```

3. Apply the Markov transition matrix to move to the next page.

Then:

```text
P(at least one typo in pages 1..k) = 1 - P(no typo in pages 1..k)
```

The crossover page `n*` is the smallest `n` such that:

```text
P(at least one typo in pages 1..n) > threshold
```

## Scenarios in the Script

The script uses three scenarios:

```text
mild:
  p_F = 0.01
  p_T = 0.06
  b = 0.02
  initial tired probability = 0.02

baseline:
  p_F = 0.01
  p_T = 0.12
  b = 0.03
  initial tired probability = 0.05

fragile:
  p_F = 0.01
  p_T = 0.18
  b = 0.06
  initial tired probability = 0.08
```

For each scenario, the script sweeps:

```text
a = P(T at k+1 | T at k)
```

from `0.10` to `0.98`.

## Interpretation

Increasing tired-state persistence has two effects:

1. Once tiredness starts, typo-prone stretches last longer.
2. The long-run fraction of tired pages increases.

The long-run tired probability is:

```text
pi_T = b / (b + 1 - a)
```

The corresponding long-run average page typo probability is:

```text
(1 - pi_T) p_F + pi_T p_T
```

That is not the whole story, because burstiness also changes the clustering of
typos, but it explains why the cumulative probability curves rise faster as
`a` increases.

## Sample Burst Plot

The script also simulates page-by-page typo sequences with a fixed random seed.
Those plots are named:

```text
outputs/markovian_sample_bursts_mild.png
outputs/markovian_sample_bursts_baseline.png
outputs/markovian_sample_bursts_fragile.png
```

Each row uses a different tired-state persistence value. Orange marks show
pages where the hidden typist state is tired. Red marks show pages where a typo
appears. As `a` increases, the orange tired-state runs become longer, and the
red typo marks become more clustered.
