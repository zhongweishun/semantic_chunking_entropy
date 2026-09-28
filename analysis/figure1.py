"""Figure 1: cumulative surprisal, finite-N RTM rate, illustrative semantic tree."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8),
                             gridspec_kw={'width_ratios':[1.15,1,1.35]})
    ax = axes[0]
    norm = Normalize(0, 2500)
    cmap = sns.color_palette('mako', as_cmap=True)
    for values in sorted(curves, key=len):
        ax.plot(np.arange(values.size), values, color=cmap(norm(values.size)), lw=1.5)
    xx = np.array([0,max(map(len,curves))-1])
    ax.plot(xx,slope*xx+intercept,'b-.',lw=2,label=f'$h={slope:.3f}$\n$R^2={r2:.3f}$')
    ax.plot(xx,REFERENCE_RATE*xx+intercept,'r--',lw=2,label='Theory '+r'$(K=4)$')
    ax.set(xlabel='Token Number',ylabel='Cumulative surprisal (nats)')
    ax.legend(fontsize=12,loc='upper left',handlelength=1.,handletextpad=.4)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,
                 label=r'Text Length $(N)$',ticks=np.arange(0,2501,500))
    ks = np.arange(2, 10)
    finite_n = 5000
    rates = [expected_entropy(int(k),finite_n)[-1]/finite_n for k in ks]
    ax=axes[1]
    ax.plot(ks,rates,'o-',color='tomato',lw=2.5,ms=6,
            label=r'Semantic Tree Entropy $h_K$')
    ax.set(xlabel=r'Max. Chunks $(K)$',ylabel=r'Entropy rate $h$ (nats/token)',
           xticks=[2,4,6,8],ylim=(1,3.5))
    # The manuscript's tree-entropy panel uses serif lettering.
    for text in [ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
        text.set_fontfamily('STIXGeneral')
    ax.legend(prop={'family':'STIXGeneral','size':12},loc='lower right')
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
        ax.plot([nodes[u][0],nodes[v][0]],[nodes[u][1],nodes[v][1]],color='black',lw=1.2,zorder=1)
    for px,py,label in nodes.values():
        ax.scatter([px],[py],s=720,color='#3ce9b4' if label.isdigit() else '#43c5e5',edgecolor='black',lw=1.,zorder=2)
        ax.text(px,py,label,ha='center',va='center',fontsize=20 if label.isdigit() else 11,
                fontfamily='STIXGeneral',zorder=3)
    ax.set(xlim=(-.3,8.8),ylim=(-.6,5.1))
    ax.text(4.25,4.8,'[[The quick brown] fox] [jumps over] [[the lazy dog] .]',
            ha='center',va='center',fontsize=13)
    ax.axis('off')
    for ax,letter,title,color in zip(axes,'abc',
            ['LLM Cross-Entropy','Semantic Tree Entropy','Semantic Tree'],
            ['#32b34a','#3ba5ff','#3ba5ff']):
        ax.set_title(title,loc='left',fontsize=22,fontweight='bold',color=color,pad=22)
        ax.text(-.17 if letter!='c' else -.15,1.08,f'({letter})',
                transform=ax.transAxes,fontsize=24,va='bottom')
    fig.tight_layout(w_pad=1.2)
    save(fig,out,'figure1_entropy')
    write_json(results/'figure1.json',{'llm_stories':len(curves),'pooled_token_observations':len(x),'slope':slope,'intercept':intercept,'r_squared':r2,'reference_rate':REFERENCE_RATE,'rtm_finite_N':finite_n,'rtm_K':ks.tolist(),'rtm_H_over_N':rates})
