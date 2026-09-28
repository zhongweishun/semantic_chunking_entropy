"""Figure 2b: original 925-tree windows, 15 bins, finite-N RTM overlay."""
import numpy as np
import matplotlib.pyplot as plt
from .common import ROOT, read_json, pooled_sizes, save, write_json


def make(trees,out,results):
    pooled=pooled_sizes(trees)
    theory=read_json(ROOT/'theory/theory_dist_k=4.json')
    windows=read_json(ROOT/'data/windows_nb100.json')
    fig,axes=plt.subplots(3,3,figsize=(11,9),layout='constrained')
    rows=[]
    for ax,level in zip(axes.flat,range(2,11)):
        lo,hi=windows[str(level)]['min'],windows[str(level)]['max']
        values=pooled[level]; selected=values[(values>=lo)&(values<=hi)]
        edges=np.linspace(lo,hi,16); centers=(edges[:-1]+edges[1:])/2
        empirical,_=np.histogram(selected,bins=edges,density=True)
        pn=np.asarray(theory[str(level)]['distn'][-1]); xt=np.arange(1,pn.size+1)/2500; yt=pn*2500
        xs=np.linspace(lo,hi,400); ys=np.interp(xs,xt,yt,left=0,right=0)
        p=empirical*np.diff(edges);p/=p.sum()
        q=np.interp(centers,xt,yt,left=0,right=0)*np.diff(edges);q/=q.sum()
        pc=np.clip(p,1e-12,None);pc/=pc.sum();qc=np.clip(q,1e-12,None);qc/=qc.sum()
        kl=float(np.sum(pc*np.log(pc/qc)))
        keep=empirical>0; ax.semilogy(centers[keep],empirical[keep],'o',ms=5,label='925 trees')
        keep=ys>0;ax.semilogy(xs[keep],ys[keep],'r--',lw=2,label='RTM, K=4')
        ax.set(xlabel=r'$s=n/N$',ylabel=r'$f_L(s)$',title=f'Level {level}')
        ax.grid(alpha=.18)
        if level==2:ax.legend(fontsize=8)
        rows.append({'level':level,'samples':len(values),'in_window':len(selected),'window':[lo,hi],
                     'bin_edges':edges.tolist(),'density':empirical.tolist(),'KL':kl})
    save(fig,out,'figure2b_chunk_sizes')
    write_json(results/'figure2b.json',{'trees':len(trees),'theory_N':2500,'bins':15,'levels':rows,'mean_KL_L2_9':float(np.mean([r['KL'] for r in rows if r['level']<=9]))})
