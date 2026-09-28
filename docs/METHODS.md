# Reproduction methods

All entropies and logarithms use natural-log units. Root level is 1. Every
figure is generated from code and bundled inputs; the plotting scripts do not
read or crop pre-rendered manuscript figures.

## The two LLM summaries

For a story with cumulative information array `TI`, its rate is the slope of
ordinary least squares on `arange(len(TI))`. Figure 3 takes the mean and
population standard deviation of these per-story slopes over the 925 paired
IDs. Figure 1a instead fits all token-index/information observations pooled
over the full 1,000-story cache. No intercept is forced to zero.

## Tree score and terminal nodes

Let `Z_K(n) = binom(n+K-1, K-1)`. At each internal node, add `ln Z_K(n)` to
the negative log score. A terminal leaf of size `n>1` also contributes
`ln Z_K(n)`; sizes 0 and 1 contribute zero. Sum over the partition and divide
by its root mass. This is exactly the scoring convention in the original
925-tree entropy script.

It treats a terminal multi-token leaf as a labelled no-split outcome. It is
not the likelihood obtained by marginalizing over the K possible positions
of the nonempty bin in such an outcome. This distinction matters when
comparing other definitions of tree probability.

For a random weak composition, recursion stops when at most one part is
positive. Successful compositions recurse into all positive parts. Empty
parts need no simulation because their score is zero. The expected score obeys

```text
H(0) = H(1) = 0
H(N) = ln Z_K(N) + K / Z_K(N) * sum[m=2..N-1] Z_(K-1)(N-m) H(m).
```

`analysis/rtm.py` evaluates the convolution by repeated cumulative sums in
`O(K*N)` time. Tests compare this implementation with the direct recurrence
and the closed-form binary solution. Figure 1b shows `H(5000)/5000`; Figure 3a
shows the simulation mean and empirical 95% quantiles. The dashed value
`2.507174205174` is retained from the original plotting
scripts and is not used to fit, rescale or force agreement with simulations.

## Chunk-size histogram and KL

Each integer leaf or internal node contributes its size to its own level;
leaves are not carried forward. Pool `n/N` only if the requested level is
strictly shallower than that particular tree's maximum depth.

For Figure 2b, retain samples within the saved level-specific windows, use 15
equal-width bins, and normalize the histogram inside the window. Interpolate
the supplied theoretical density at the bin centers; multiply both densities
by bin width and separately normalize to probability vectors. Clip each
probability to at least `1e-12`, renormalize, and evaluate `sum p*ln(p/q)`.
Levels 2–10 are plotted; levels 2–9 enter the reported mean KL. This reproduces
the original correlations-repository estimator, including its window choice.

## Typicality bins

Use 12 bins with log-spaced edges between the smaller of the two minimum
lengths and one plus the larger maximum length. Intervals are left-closed and
right-open. At least five observations are required per bin. Each mean is
plotted at the mean observed length in that bin; the shading is one population
standard deviation (`ddof=0`). LLM lengths and partition lengths are retained
separately.

## Log-size theory and standardization

Let `R ~ Beta(1,K-1)` and `s_L = product(R_1,...,R_(L-1))`. In the variable
`t=-ln(s)`, the single-factor density is

```text
g_2(t) = (K-1) exp(-t) [1-exp(-t)]^(K-2), t >= 0.
```

Repeated linear FFT convolutions produce `g_L`; the ordinary size density is
`f_L(s)=g_L(-ln(s))/s`. The grid has 65,536 points and `0 <= t <= -ln(1e-20)`.
Negative roundoff is clipped to zero. Curves are shown above a relative
`1e-10` noise floor in `g`, and the retained probability mass of each level is
recorded in the numerical output. Level densities are not separately
renormalized after truncation.

For `K=4`, the exact log moments are

```text
mu_L = -(L-1) * sum[j=1..3] 1/j
sigma_L^2 = (L-1) * sum[j=1..3] 1/j^2.
```

The transformed density is `sigma_L * g_L` versus
`(ln(s)-mu_L)/sigma_L`. Tests compare the two-factor FFT density against an
independent Mellin-convolution integral at four size fractions.

For the 925-tree empirical panels, discard zero size fractions, take log
sizes, and estimate each level's mean and population SD. Histograms use six
bins per level, with edges spanning the observed log-size or standardized
range. In the log-size density panel, divide the density in `ln(s)` by `s`
at the geometric bin centers. The normal reference is not fitted to the
histogram. These transformations preserve the specified empirical cohort.

## Validation

`validate.py` checks every archive member hash and ID, nonnegative integer
leaves, partition masses, saved-token counts, paired-score IDs, finite score
arrays, and exact equality of the 925 score arrays to their entries in the
1,000-story cache. It preserves and reports the one known partition/text
coverage mismatch. `validate.py --results` additionally checks the regression,
KL and paired-entropy reference values and the existence of all figure files.

Tests exercise numerical identities independently of the plotting code,
including the absorbing-state Markov calculation against the original scalar
implementation. The optional chunker is preserved by its source SHA-256 and
is separate from these offline checks.
