"""Vector extension figures from frozen outputs, with pure RGB and neutral grays."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/extensions';F=ROOT/'figures'
BLUE,RED,MAGENTA,GRAY,BLACK='#0000ff','#ff0000','#ff00ff','#808080','#000000'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
 'axes.titlesize':11,'axes.titleweight':'bold','xtick.labelsize':9,'ytick.labelsize':9,
 'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.7,'lines.linewidth':1.9,'pdf.fonttype':42,'ps.fonttype':42,
 'savefig.facecolor':'white','figure.facecolor':'white'})

def panel(ax,title):
    ax.set_title(title,loc='left',pad=13);ax.grid(axis='y',color=GRAY,alpha=.17,lw=.6);ax.set_axisbelow(True)

def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight',metadata={'Creator':None,'Producer':None,'CreationDate':None,'ModDate':None})
    fig.savefig(F/(name+'.png'),bbox_inches='tight',dpi=220);plt.close(fig)

def weighted_horizon():
    w=pd.read_csv(R/'weights.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.8));fig.subplots_adjust(left=.10,right=.98,top=.85,bottom=.27,wspace=.37)
    ax=axes[0];panel(ax,'A   Service metric')
    names=['Absolute','Reciprocal','Reciprocal square'];colors=[BLACK,BLUE,MAGENTA];markers=['s','o','D']
    for beta in range(3):
        data=w[(w.beta==beta)&(w.alpha==.25)]
        ax.scatter(data.unweighted_shortfall,data.jain,color=GRAY,alpha=.27,s=13,edgecolors='none')
        ax.scatter(data.unweighted_shortfall.mean(),data.jain.mean(),color=colors[beta],s=65,marker=markers[beta],edgecolor=BLACK,lw=.5,zorder=4)
    ax.set_xlabel('Unweighted shortfall');ax.set_ylabel('Mean Jain index');ax.set_ylim(.5,1.025)
    handles=[Line2D([],[],color=c,marker=m,ls='None',label=n,markersize=6) for c,m,n in zip(colors,markers,names)]
    ax.legend(handles=handles,loc='lower right',frameon=False,fontsize=8)
    ax=axes[1];panel(ax,'B   Horizon cost gap')
    c=pd.read_csv(ROOT/'results/comparisons/clairvoyant.csv');c=c[c.mu==3.].sort_values('block')
    x=c.block.to_numpy();actual=c.gap/c.phi_star_upper*100;bound=c.sharp_bound/c.phi_star_upper*100
    ax.vlines(x,actual,bound,color=GRAY,alpha=.5,lw=1)
    ax.scatter(x,actual,color=BLUE,s=21,marker='o')
    ax.scatter(x,bound,color=BLACK,s=28,marker='_',linewidths=1.7)
    ax.set_xlabel('Electricity block');ax.set_ylabel('Gap / clairvoyant cost (%)');ax.set_xticks([0,10,20,29]);ax.set_ylim(0,1.32*bound.max())
    ax.legend(handles=[Line2D([],[],color=BLUE,marker='o',ls='None',label='Feedback gap'),Line2D([],[],color=BLACK,marker='_',ls='None',label='Certificate')],loc='upper left',ncol=2,frameon=False,fontsize=8)
    save(fig,'weighted_horizon')

def closed_loop():
    d=pd.read_csv(R/'ramps.csv');rec=pd.read_csv(R/'static_recovery_epochs.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.65));fig.subplots_adjust(left=.10,right=.98,top=.85,bottom=.27,wspace=.35)
    ax=axes[0];panel(ax,'A   Repeated feasibility')
    for alpha,color,marker in [(.25,BLUE,'o'),(1.,BLACK,'s')]:
        z=d[d.alpha==alpha].groupby('ramp').mean(numeric_only=True);x=np.arange(len(z))
        ax.plot(x,z.hard_feasible_fraction*100,color=color,marker=marker,label='Feedback' if alpha==.25 else 'Myopic')
        ax.plot(x,z.full_priority_fraction*100,color=color,marker=marker,ls=':',lw=1.7)
    ax.set_xticks([0,1,2],['0.1','0.5','2']);ax.set_xlabel('Ramp limit per recipient');ax.set_ylabel('Observations (%)');ax.set_ylim(0,104)
    ax.legend(frameon=False,loc='lower right',fontsize=8)
    ax.text(.04,.97,'Solid: hard feasible\nDotted: full priorities',transform=ax.transAxes,va='top',fontsize=8)
    ax=axes[1];panel(ax,'B   Two recovery events')
    z=rec[(rec.ramp==.1)&(rec.alpha==.25)&(rec['update']<=10)]
    t=np.r_[0,z['update']];total=np.r_[24,z.total];high=np.r_[8,z.high]
    ax.plot(t,total,color=BLACK,marker='s',markersize=4,label='Total allocation')
    ax.plot(t,high,color=BLUE,marker='o',markersize=4,label='High-tier service')
    ax.axhline(12,color=GRAY,ls='--',lw=1);ax.axhline(8,color=GRAY,ls=':',lw=1)
    ax.axvline(4,color=GRAY,alpha=.45,lw=.8);ax.axvline(7,color=GRAY,alpha=.45,lw=.8)
    ax.annotate('Hard feasible',xy=(4,12),xytext=(3.2,19),fontsize=8,arrowprops={'arrowstyle':'-','color':GRAY},ha='left')
    ax.annotate('Full priorities',xy=(7,8),xytext=(6.1,15.7),fontsize=8,arrowprops={'arrowstyle':'-','color':GRAY},ha='left')
    ax.set_xticks([0,1,4,7,10]);ax.set_xlabel('Updates after capacity drop');ax.set_ylabel('Service units');ax.set_ylim(0,26)
    ax.legend(frameon=False,loc='lower right',fontsize=8)
    save(fig,'closed_loop')

if __name__=='__main__':
    F.mkdir(exist_ok=True)
    weighted_horizon();closed_loop()
    paths=['weights.csv','ramps.csv','static_recovery_epochs.csv']
    report={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'input_sha256':{**{p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths},
        'comparisons/clairvoyant.csv':hashlib.sha256((ROOT/'results/comparisons/clairvoyant.csv').read_bytes()).hexdigest()},
      'figures':{n:hashlib.sha256((F/n).read_bytes()).hexdigest() for n in ['weighted_horizon.pdf','closed_loop.pdf']}}
    (R/'figure_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
