"""Figure 1: cumulative surprisal, finite-N RTM rate, illustrative semantic tree."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .common import ROOT, REFERENCE_RATE, read_json, save, write_json
from .rtm import expected_entropy


def make(out, results):
    cache = read_json(ROOT/'data/llm_entropy_reddit1000.json.gz')
    curves = [np.asarray(v['TI_cumulative_token']) for v in cache.values()]
    x = np.concatenate([np.arange(len(y)) for y in curves])
    y = np.concatenate(curves)
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1-np.sum((y-(slope*x+intercept))**2)/np.sum((y-y.mean())**2)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    ax = axes[0]
    norm = Normalize(0, 2500)
    for values in sorted(curves, key=len, reverse=True):
        ax.plot(np.arange(values.size), values, color=plt.cm.viridis(norm(values.size)), alpha=.12, lw=.55, rasterized=True)
    xx = np.array([0,max(map(len,curves))-1])
    ax.plot(xx,slope*xx+intercept,'b-.',lw=1.8,label=f'Fit: h={slope:.3f}, R²={r2:.3f}')
    ax.plot(xx,REFERENCE_RATE*xx+intercept,'r--',lw=1.6,label='RTM reference (K=4)')
    ax.set(xlabel='Token number',ylabel='Cumulative surprisal (nats)',title='(a) LLM cross-entropy')
    ax.legend(fontsize=8,loc='upper left')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,label='Text length (tokens)',shrink=.8)
    ks = np.arange(2, 10)
    finite_n = 5000
    rates = [expected_entropy(int(k),finite_n)[-1]/finite_n for k in ks]
    ax=axes[1]
    ax.plot(ks,rates,'o-',color='tomato',label=f'H(N)/N, N={finite_n:,}')
    ax.set(xlabel='Maximum chunks (K)',ylabel='RTM entropy (nats/token)',title='(b) Semantic-tree entropy',xticks=[2,4,6,8])
    ax.legend(fontsize=9)
    # Fixed illustration transcribed from the manuscript, not an LLM experiment.
    nodes={'r':(4,4,'10'), 'a':(1.5,3,'4'),'b':(4,3,'2'),'c':(7,3,'4'),
           'd':(1,2,'3'),'fox':(2.4,2,'fox'),'jumps':(3.6,2,'jumps'),'over':(4.5,2,'over'),
           'e':(6.6,2,'3'),'dot':(8.2,2,'.'),'the1':(.3,1,'The'),'f':(1.6,1,'2'),
           'g':(6.1,1,'2'),'dog':(7.4,1,'dog'),'quick':(1.1,0,'quick'),'brown':(2.1,0,'brown'),
           'the2':(5.6,0,'the'),'lazy':(6.6,0,'lazy')}
    edges=[('r','a'),('r','b'),('r','c'),('a','d'),('a','fox'),('b','jumps'),('b','over'),
           ('c','e'),('c','dot'),('d','the1'),('d','f'),('e','g'),('e','dog'),('f','quick'),('f','brown'),('g','the2'),('g','lazy')]
    ax=axes[2]
    for u,v in edges:
        ax.plot([nodes[u][0],nodes[v][0]],[nodes[u][1],nodes[v][1]],color='black',lw=.8,zorder=1)
    for px,py,label in nodes.values():
        ax.scatter([px],[py],s=620,color='#3ce9b4' if label.isdigit() else '#43c5e5',edgecolor='black',lw=.7,zorder=2)
        ax.text(px,py,label,ha='center',va='center',fontsize=9,zorder=3)
    ax.set(xlim=(-.3,8.8),ylim=(-.6,4.7),title='(c) Semantic-tree illustration')
    ax.axis('off')
    save(fig,out,'figure1_entropy')
    write_json(results/'figure1.json',{'llm_stories':len(curves),'pooled_token_observations':len(x),'slope':slope,'intercept':intercept,'r_squared':r2,'reference_rate':REFERENCE_RATE,'rtm_finite_N':finite_n,'rtm_K':ks.tolist(),'rtm_H_over_N':rates})
