"""Figure 3: seeded RTM realizations and the paired 925-story entropy analysis."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from .common import REFERENCE_RATE, entropy_records, binned, save, write_json
from .rtm import typicality, expected_entropy


def make(trees,out,results,replicates=128,seed=20260928):
    rows=entropy_records(trees)
    nt=np.asarray([r['N_tree'] for r in rows]); nl=np.asarray([r['N_llm'] for r in rows])
    tr=np.asarray([r['tree_rate'] for r in rows]); ll=np.asarray([r['llm_rate'] for r in rows])
    sizes,sims=typicality(replicates,seed)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    ax=axes[0]; lo,hi=np.quantile(sims,[.025,.975],axis=1)
    ax.fill_between(sizes,lo,hi,color='tab:blue',alpha=.2,label='95% empirical quantiles')
    ax.plot(sizes,sims.mean(axis=1),color='tab:blue',lw=2,label='Simulation mean')
    exact=expected_entropy(4,int(sizes[-1]))
    ax.plot(sizes,exact[sizes]/sizes,color='black',ls=':',lw=1.2,label='Finite-N expectation')
    ax.set(xscale='log',xlabel='N (tokens)',ylabel='Entropy rate (nats/token)',title='(a) Random trees')
    ax=axes[1]
    ax.scatter(nl,ll,s=15,color='tab:green',alpha=.2,label='LLM',rasterized=True)
    ax.scatter(nt,tr,s=15,color='tab:blue',alpha=.2,label='Tree model',rasterized=True)
    ax.set(xlabel='N (tokens)',title='(b) 925 stories',ylim=(0,5.1))
    edges=np.geomspace(min(nt.min(),nl.min()),max(nt.max(),nl.max())+1,13)
    summaries={}
    ax=axes[2]
    for name,n,y,color in [('LLM',nl,ll,'tab:green'),('Tree model',nt,tr,'tab:blue')]:
        b=binned(n,y,edges)
        ax.plot(b[:,0],b[:,1],'-o',color=color,ms=4,label=name)
        ax.fill_between(b[:,0],b[:,1]-b[:,2],b[:,1]+b[:,2],color=color,alpha=.2,lw=0)
        summaries[name]=b.tolist()
    ax.set(xscale='log',xlabel='N (tokens)',title='(c) Mean and standard deviation',ylim=(0,5.1))
    for ax in axes:
        ax.axhline(REFERENCE_RATE,color='red',ls='--',lw=1.5,label='Manuscript RTM reference')
        ax.legend(fontsize=7,loc='lower right')
    save(fig,out,'figure3_typicality')
    with (results/'entropy_rates_925.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(results/'rtm_simulation.npz',N=sizes,rates=sims,seed=seed,replicates=replicates)
    write_json(results/'figure3.json',{'trees':len(rows),'LLM_mean':float(ll.mean()),'LLM_std':float(ll.std()),'tree_mean':float(tr.mean()),'tree_std':float(tr.std()),'correlation':float(np.corrcoef(ll,tr)[0,1]),'bin_edges':edges.tolist(),'binned_columns':['mean_N','mean_rate','population_std','count'],'binned':summaries,'simulation_seed':seed,'replicates_per_N':replicates,'simulation_N':sizes.tolist(),'simulation_mean':sims.mean(axis=1).tolist(),'reference_rate':REFERENCE_RATE})
