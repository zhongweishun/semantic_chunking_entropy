"""Figure 5a-d: Beta-product theory and log-size scaling of the 925 trees."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.stats import norm
from .common import LEVELS, pooled_sizes, save, write_json, panel_label
from .rtm import scaling_densities


def make(trees,out,results):
    t,densities=scaling_densities()
    s=np.exp(-t)
    colors=plt.cm.Blues(np.linspace(.3,.9,10))
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    pooled=pooled_sizes(trees)
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
        values=pooled[level]; values=values[values>0]
        z=np.log(values); mu_emp=float(z.mean()); sd_emp=float(z.std())
        # Log-spaced s bins correspond to equal-width bins in ln(s).
        edges=np.linspace(z.min(),z.max(),7); centers=(edges[:-1]+edges[1:])/2
        gz,_=np.histogram(z,bins=edges,density=True)
        keep=gz>0
        axes[1,0].loglog(np.exp(centers[keep]),gz[keep]/np.exp(centers[keep]),'o-',color=colors[i],ms=6,lw=3,alpha=.8)
        standard=(z-mu_emp)/sd_emp
        xedges=np.linspace(standard.min(),standard.max(),7)
        hx,_=np.histogram(standard,bins=xedges,density=True)
        axes[1,1].plot((xedges[:-1]+xedges[1:])/2,hx,'o',color=colors[i],ms=5)
        report.append({'level':level,'positive_chunk_samples':int(values.size),'mean_log_s':mu_emp,'std_log_s':sd_emp,'theory_window_mass':float(np.trapz(g,t)),'standardized_bin_edges':xedges.tolist(),'standardized_density':hx.tolist()})
    for ax in [axes[0,1],axes[1,1]]:
        xx=np.linspace(-5,5,400);ax.plot(xx,norm.pdf(xx),'k--',lw=4,label=r'$\mathcal{N}(0,1)$')
        ax.set(xlim=(-5,5),ylim=(0,.725),xticks=[-5,-2.5,0,2.5,5])
        ax.legend(loc='upper right')
    axes[0,0].set(xlabel=r'$s$',ylabel=r'$f_L(s)$',xlim=(2e-4,1),ylim=(1e-4,1e4))
    axes[0,1].set(xlabel=r'$(\ln(s)-\mu_L)/\sigma_L$',ylabel=r'$\sigma_L s f_L(s)$')
    axes[1,0].set(xlabel=r'$s$',ylabel=r'$\hat f_L(s)$',xlim=(2e-4,1),ylim=(1e-4,1e4))
    axes[1,1].set(xlabel=r'$(\ln(s)-\hat\mu_L)/\hat\sigma_L$',ylabel=r'$\hat\sigma_L s\hat f_L(s)$')
    sm=plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(1.5,12.5),10),cmap=matplotlib_colors(colors))
    for letter,ax in zip('abcd',axes.flat):
        fig.colorbar(sm,ax=ax,ticks=list(LEVELS),label=r'Level$(L)$')
        panel_label(ax,letter)
    for ax in axes[:,0]:
        ax.set_yticks([1e-4,1e-2,1,1e2,1e4])
        ax.yaxis.set_minor_locator(LogLocator(base=10.,subs=np.arange(2,10)*.1,numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    save(fig,out,'figure5abcd_scaling')
    write_json(results/'figure5abcd.json',{'trees':len(trees),'K':4,'FFT_points':len(t),'log_window':[0,float(t[-1])],'bins_per_empirical_level':6,'levels':report})


def matplotlib_colors(colors):
    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)
