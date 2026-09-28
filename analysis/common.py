"""Data access and the scoring conventions of the 925-tree analysis."""
from pathlib import Path
import gzip
import json
import math
import tarfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RATE = 2.507174205174  # reference used in the original plotting scripts
LEVELS = range(2, 12)


def read_json(path):
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def iter_trees(archive=None):
    """Yield original story IDs and full, unmodified result JSONs without extraction."""
    archive = archive or ROOT / 'data/trees_cut30.tar.gz'
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            if member.isfile() and member.name.endswith('_result.json'):
                yield Path(member.name).name.split('_')[0], json.load(tf.extractfile(member))


def tree_size(node):
    return node if isinstance(node, int) else sum(tree_size(c) for c in node)


def sizes_by_level(node):
    """Root is level 1. Terminal leaves are not copied into subsequent levels."""
    levels = {}
    def visit(branch, level):
        size = branch if isinstance(branch, int) else sum(visit(c, level+1) for c in branch)
        levels.setdefault(level, []).append(size)
        return size
    visit(node, 1)
    return levels


def pooled_sizes(trees):
    """Normalize by partition mass and exclude each tree's deepest level."""
    result = {level: [] for level in LEVELS}
    for _, tree in trees:
        levels = sizes_by_level(tree['partition'])
        n, depth = levels[1][0], max(levels)
        for level in LEVELS:
            if level in levels and level < depth:
                result[level].extend(size/n for size in levels[level])
    return {level: np.asarray(values) for level, values in result.items()}


def size_and_logprob(node, k=4):
    """Original RTM score: ln Z_k(n) for internal nodes AND multi-token leaves.

    A terminal multi-token leaf is scored as one labelled no-split outcome,
    without marginalizing the k possible positions of the nonempty bin.
    """
    if isinstance(node, int):
        return node, -math.log(math.comb(node+k-1, k-1)) if node > 1 else 0.0
    children = [size_and_logprob(c, k) for c in node]
    n = sum(v[0] for v in children)
    return n, sum(v[1] for v in children) - math.log(math.comb(n+k-1, k-1))


def entropy_records(trees):
    scores = read_json(ROOT/'data/llm_entropy_reddit925.json.gz')
    assert {story_id for story_id, _ in trees} == set(scores)
    rows = []
    for story_id, tree in trees:
        ti = np.asarray(scores[story_id]['TI_cumulative_token'])
        slope, intercept = np.polyfit(np.arange(ti.size), ti, 1)
        n, logp = size_and_logprob(tree['partition'])
        rows.append({'story_id':story_id, 'N_tree':n, 'N_llm':int(ti.size),
                     'tree_rate':-logp/n, 'llm_rate':float(slope),
                     'llm_intercept':float(intercept)})
    return rows


def binned(n, y, edges, minimum=5):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        keep = (n >= lo) & (n < hi)
        if keep.sum() >= minimum:
            rows.append([n[keep].mean(), y[keep].mean(), y[keep].std(ddof=0), keep.sum()])
    return np.asarray(rows)


def style():
    """Original manuscript scripts: DejaVu Sans 20 pt and STIX mathematics."""
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans', 'mathtext.fontset':'stix',
                         'font.size':20, 'savefig.dpi':130,
                         'pdf.fonttype':42, 'ps.fonttype':42})


def panel_label(ax, letter, x=-.28):
    ax.text(x, 1.08, f'({letter})', transform=ax.transAxes,
            fontsize=26, va='bottom', ha='left')


def save(fig, out, stem, dpi=130):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'{stem}.{ext}', dpi=dpi)
    plt.close(fig)


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False)+'\n', encoding='utf-8')
