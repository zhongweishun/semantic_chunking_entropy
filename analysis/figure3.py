"""Figure 3: seeded RTM realizations and the paired 925-story entropy analysis."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from .common import REFERENCE_RATE, entropy_records, binned, save, write_json, panel_label
from .rtm import typicality


def make(trees,out,results,replicates=128,seed=20260928):
    rows=entropy_records(trees)
    nt=np.asarray([r['N_tree'] for r in rows]); nl=np.asarray([r['N_llm'] for r in rows])
    tr=np.asarray([r['tree_rate'] for r in rows]); ll=np.asarray([r['llm_rate'] for r in rows])
    sizes,sims=typicality(replicates,seed)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    ax=axes[0]; lo,hi=np.quantile(sims,[.025,.975],axis=1)
    ax.fill_between(sizes,lo,hi,color='tab:blue',alpha=.2,label='95% quantiles')
    ax.plot(sizes,sims.mean(axis=1),color='tab:blue',lw=2.5,label='mean')
    ax.set(xscale='log',xlabel=r'$N$',xlim=(10,10000),ylim=(1,3.25))
    ax.set_xticks([10,100,1000,10000],labels=[r'$10^1$',r'$10^2$',r'$10^3$',r'$10^4$'])
    ax=axes[1]
    ax.scatter(nl,ll,s=50,color='tab:green',edgecolors='black',alpha=.2,label='LLM')
    ax.scatter(nt,tr,s=50,marker='o',color='tab:blue',alpha=.2,label='Tree model')
    ax.set(xlabel=r'$N$ (token)',ylim=(0,5.1))
    edges=np.geomspace(min(nt.min(),nl.min()),max(nt.max(),nl.max())+1,13)
    summaries={}
    ax=axes[2]
    for name,n,y,color in [('LLM',nl,ll,'tab:green'),('Tree model',nt,tr,'tab:blue')]:
        b=binned(n,y,edges)
        ax.plot(b[:,0],b[:,1],'-o',color=color,lw=2.5,ms=6,label=name)
        ax.fill_between(b[:,0],b[:,1]-b[:,2],b[:,1]+b[:,2],color=color,alpha=.2,lw=0)
        summaries[name]=b.tolist()
    ax.set(xscale='log',xlabel=r'$N$ (token)',ylim=(0,5.1))
    for letter,ax in zip('abc',axes):
        ax.set_ylabel('Entropy rate estimate\n(nats/token)')
        ax.axhline(REFERENCE_RATE,color='red',ls='--',lw=2,label='Theory '+r'$(K=4)$')
        leg=ax.legend(fontsize=12,loc='lower right')
        if letter=='b':
            for handle in leg.legend_handles:
                handle.set_alpha(1.)
        panel_label(ax,letter)
    fig.tight_layout()
    save(fig,out,'figure3_typicality')
    with (results/'entropy_rates_925.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(results/'rtm_simulation.npz',N=sizes,rates=sims,seed=seed,replicates=replicates)
    write_json(results/'figure3.json',{'trees':len(rows),'LLM_mean':float(ll.mean()),'LLM_std':float(ll.std()),'tree_mean':float(tr.mean()),'tree_std':float(tr.std()),'correlation':float(np.corrcoef(ll,tr)[0,1]),'bin_edges':edges.tolist(),'binned_columns':['mean_N','mean_rate','population_std','count'],'binned':summaries,'simulation_seed':seed,'replicates_per_N':replicates,'simulation_N':sizes.tolist(),'simulation_mean':sims.mean(axis=1).tolist(),'reference_rate':REFERENCE_RATE})
