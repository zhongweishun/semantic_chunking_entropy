"""Rebuild the supported main-text figure analyses from this checkout alone."""
from pathlib import Path
import argparse
import time
from analysis import figure1, figure2, figure3, figure5
from analysis.common import ROOT, iter_trees, style


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',nargs='+',choices=['1','2','3','5'],default=['1','2','3','5'])
    parser.add_argument('--out-dir',type=Path,default=ROOT/'figures')
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--seed',type=int,default=20260928)
    parser.add_argument('--simulation-replicates',type=int,default=128)
    args=parser.parse_args()
    if args.simulation_replicates<2:parser.error('simulation-replicates must be at least 2')
    args.results_dir.mkdir(parents=True,exist_ok=True)
    style()
    start=time.perf_counter()
    trees=list(iter_trees()) if any(f in args.figure for f in ['2','3','5']) else None
    for number in args.figure:
        print(f'Rebuilding figure {number}...',flush=True)
        if number=='1':figure1.make(args.out_dir,args.results_dir)
        if number=='2':figure2.make(trees,args.out_dir,args.results_dir)
        if number=='3':figure3.make(trees,args.out_dir,args.results_dir,args.simulation_replicates,args.seed)
        if number=='5':figure5.make(trees,args.out_dir,args.results_dir)
    print(f'Done in {time.perf_counter()-start:.1f}s. Figures: {args.out_dir}',flush=True)


if __name__=='__main__':main()
