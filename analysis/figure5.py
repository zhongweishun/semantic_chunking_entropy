"""Figure 5a-d: Beta-product theory and log-size scaling of the 925 trees."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.stats import norm
from .common import LEVELS, sizes_by_level, save, write_json, panel_label
from .rtm import scaling_densities



def notebook_samples(trees):
    """tree_universal.ipynb: n / len(saved tokens), excluding deepest level.

    This follows tree_analysis's num_tokens and collect_level_data literally.
    Figure 2's source script instead normalizes by partition mass; these differ
    for the single retained partial-coverage record 100426.
    """
    pooled = {level: [] for level in LEVELS}
    for _, tree in sorted(trees,key=lambda item:len(item[1]['tokens'])):
        levels = sizes_by_level(tree['partition'])
        n = len(tree['tokens'])
        for level in LEVELS:
            if level < max(levels) and level < levels[1][0]:
                pooled[level].extend(size/n for size in levels[level])
    return {level:np.asarray(values) for level,values in pooled.items()}


def empirical_histograms(values, bins=6):
    """Notebook cells 9 and 4: uniform s bins and standardized log-s bins."""
    values = np.asarray(values)
    values = values[values>0]
    edges = np.linspace(values.min(),values.max(),bins+1)
    density,_ = np.histogram(values,bins=edges,density=True)
    z = np.log(values)
    mu,sd = float(z.mean()),float(z.std())
    standard = (z-mu)/sd
    xedges = np.linspace(standard.min(),standard.max(),bins+1)
    hx,_ = np.histogram(standard,bins=xedges,density=True)
    return {'positive_chunk_samples':int(values.size),'mean_log_s':mu,'std_log_s':sd,
            'size_bin_edges':edges.tolist(),'size_bin_centers':((edges[:-1]+edges[1:])/2).tolist(),
            'size_density':density.tolist(),'standardized_bin_edges':xedges.tolist(),
            'standardized_density':hx.tolist()}


def plot_empirical_density(ax, records, colors):
    for record,color in zip(records,colors):
        h = np.asarray(record['size_density'])
        x = np.asarray(record['size_bin_centers'])
        keep = h>0
        ax.loglog(x[keep],h[keep],'o-',linewidth=3,color=color,alpha=.8)
    ax.set(xlabel=r'$s$',ylabel=r'$\hat{f}_{L}(s)$',ylim=(1e-4,1e4))
    # The notebook asks for (0,1) on a log axis. Keep its autoscaled positive
    # lower bound and set only the upper bound, avoiding the invalid zero.
    ax.set_xlim(right=1)
    ax.yaxis.set_minor_locator(LogLocator(base=10.,subs=np.arange(2,10)*.1,numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_empirical_collapse(ax, records, colors):
    for record,color in zip(records,colors):
        edges = np.asarray(record['standardized_bin_edges'])
        ax.plot((edges[:-1]+edges[1:])/2,record['standardized_density'],
                'o',ms=5,color=color,alpha=1)
    xx = np.linspace(-5,5,100)
    ax.plot(xx,norm.pdf(xx),'k--',lw=4,label=r'$\mathcal{N}(0,1)$')
    ax.set(xlabel=r'$(\ln (s)-\hat{\mu}_L)/\hat{\sigma}_L$',
           ylabel=r'$\hat{\sigma}_L s \hat{f}_{L}(s)$',xlim=(-5,5),ylim=(0,.725))
    ax.legend(loc='upper right')


def level_colorbar(fig,ax,colors):
    sm=plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(1.5,12.5),10),
                            cmap=matplotlib_colors(colors))
    fig.colorbar(sm,ax=ax,ticks=list(LEVELS),label=r'Level$(L)$')

def make(trees,out,results):
    t,densities=scaling_densities()
    s=np.exp(-t)
    colors=plt.cm.Blues(np.linspace(.3,.9,10))
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    pooled=notebook_samples(trees)
    report=[]
    harmonic=sum(1/j for j in range(1,4))
    harmonic2=sum(1/j**2 for j in range(1,4))
    for i,level in enumerate(LEVELS):
        g=densities[level]; f=g/s
        # Numerical FFT noise is removed using a floor in log-density space.
        valid=g>g.max()*1e-10
        axes[0,0].loglog(s[valid],f[valid],color=colors[i],lw=3,alpha=.8)
        mu=-(level-1)*harmonic; sigma=np.sqrt((level-1)*harmonic2)
        xx=(-t-mu)/sigma
        axes[0,1].plot(xx[valid],sigma*g[valid],color=colors[i],lw=3,alpha=.8)
        record = empirical_histograms(pooled[level])
        record.update(level=level,theory_window_mass=float(np.trapz(g,t)))
        report.append(record)
    plot_empirical_density(axes[1,0],report,colors)
    plot_empirical_collapse(axes[1,1],report,colors)
    for ax in [axes[0,1]]:
        xx=np.linspace(-5,5,400);ax.plot(xx,norm.pdf(xx),'k--',lw=4,label=r'$\mathcal{N}(0,1)$')
        ax.set(xlim=(-5,5),ylim=(0,.725),xticks=[-5,-2.5,0,2.5,5])
        ax.legend(loc='upper right')
    axes[0,0].set(xlabel=r'$s$',ylabel=r'$f_L(s)$',xlim=(2e-4,1),ylim=(1e-4,1e4))
    axes[0,1].set(xlabel=r'$(\ln(s)-\mu_L)/\sigma_L$',ylabel=r'$\sigma_L s f_L(s)$')
    for letter,ax in zip('abcd',axes.flat):
        level_colorbar(fig,ax,colors)
        panel_label(ax,letter)
    for ax in axes[:,0]:
        ax.set_yticks([1e-4,1e-2,1,1e2,1e4])
        ax.yaxis.set_minor_locator(LogLocator(base=10.,subs=np.arange(2,10)*.1,numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    save(fig,out,'figure5abcd_scaling')
    for stem,plot in [('figure5c_density_925',plot_empirical_density),
                      ('figure5d_collapse_925',plot_empirical_collapse)]:
        fig,ax=plt.subplots(figsize=(6,4))
        plot(ax,report,colors)
        level_colorbar(fig,ax,colors)
        fig.tight_layout()
        save(fig,out,stem)
    write_json(results/'figure5abcd.json',{'trees':len(trees),'K':4,'FFT_points':len(t),'log_window':[0,float(t[-1])],'bins_per_empirical_level':6,'empirical_size_binning':'uniform in s','normalization':'n / len(saved tokens)','notebook_cells':[4,9],'levels':report})


def matplotlib_colors(colors):
    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)
