# Manuscript plotting settings

`analysis/common.py::style()` resets Matplotlib defaults, then selects
**DejaVu Sans, 20 pt**, with **STIX** mathematical text. All four axis spines
are visible. PDFs embed TrueType fonts and retain vector lines, markers and
text. PNGs use 130 dpi, except the chunk-size grid, which uses the reference
script's 110 dpi. Explicit canvas sizes and `tight_layout()` preserve the
font-to-panel proportions; saving does not crop the canvas.

## Source mapping

The correlations repository source version is
`ca5cbeccba06fd800ef4040c9c95858a79f0fed3`.

| Reproduction | Style source | Settings |
|---|---|---|
| `analysis/figure2.py` | `reddit1000/make_chunksize_figure.py --clean` | 15 × 12 inch grid; 10 pt empirical markers; 5 pt red dashed theory; identical per-level y limits; dashed grids at alpha 0.3; no legend or KL annotations |
| `analysis/figure3.py`, panels b–c | `reddit1000/make_entropy_figure.py` | Scatter area 50 pt², alpha 0.2, black LLM marker edges; 2.5 pt binned lines and 6 pt markers; 2 pt red dashed theory; 12 pt legends; y range 0–5.1 |
| `analysis/figure1.py`, panel a | `compute_information_utilities_together_6May2026.py` and manuscript `Entropy_curve.pdf` | 925-story paired cache; one-based token positions; length-ranked Mako palette, 1.5 pt cumulative traces, 2 pt fit/reference lines, compact upper-left legend, text-length colorbar |
| `analysis/figure5.py` | `tree_universal.ipynb`, `infinite_N_chunk_size_distribution_6Aug2026.ipynb` and manuscript `tree_universal.pdf` | Ten Blues colors sampled from 0.3 to 0.9; 3 pt level curves at alpha 0.8; 5 pt standardized empirical markers; 4 pt dashed normal reference; separate discrete colorbars for levels 2–11 |

The composite figures use manuscript panel letters and labels. Figure 1
retains the green/blue headings, serif tree-entropy panel and colored tree
nodes. Figure 3 uses wrapped entropy-rate axis labels. Figure 5 uses the manuscript's displayed density
range (`10^-4` to `10^4`) and standardized range (`-5` to `5`), with its four
included panels arranged in a two-by-two grid.

Figure 2's complete plotting configuration follows the original clean script.
Its PNG was checked against a fresh run of that script in the same Matplotlib
environment and matched pixel-for-pixel.
For the assembled multi-panel figures, spacing is set in the reproduction
scripts so that the original type sizes and axis labels fit on the canvas.
The figure-specific plotting parameters are explicit next to the relevant
plot calls; there is no external style-file dependency. Seaborn supplies only
the original Mako palette and does not set the plotting theme.

These are presentation settings. Input cohorts, histogram bins, estimators,
theory calculations and simulation seeds are specified in
[Methods](METHODS.md) and the numerical output files. The finite-size
expectation is implemented in `analysis/rtm.py`; Figure 3a displays the
manuscript's simulation mean, quantile band and dashed theory reference.

Figure 1a is also exported on a 6.54 x 4 inch canvas (9% wider at the same
height), with 16 pt axis/colorbar text and a 9.6 pt legend (20% smaller).
Its colorbar spans 0 to 2500. Figure 5c and 5d have
separate 6 x 4 inch exports following notebook cells 9 and 4. The density
panel uses six linear bins in size fraction s, with arithmetic bin centers.
