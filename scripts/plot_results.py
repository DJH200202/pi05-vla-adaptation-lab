"""Render publication-friendly PNG/SVG charts using existing records only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'demo/assets/data.json').read_text())
OUT=ROOT/'demo/assets/charts'; OUT.mkdir(parents=True,exist_ok=True)
C={'Base':'#9aa5b5','A':'#9b76d3','B':'#e19d38','C':'#128b89','D':'#366bd3'}
plt.rcParams.update({'figure.facecolor':'#ffffff','axes.facecolor':'#ffffff','axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':'#344256','text.color':'#17283c',
 'xtick.color':'#566579','ytick.color':'#566579','font.size':11,'axes.titlesize':15,
 'axes.titleweight':'bold','axes.titlepad':16,'grid.alpha':.15,'savefig.dpi':180,'svg.fonttype':'none'})
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)
def foot(fig,text): fig.text(.06,.015,text,fontsize=9,color='#667085')

fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
for ax,g in zip(axes.ravel(),'ABCD'):
 t=D['training'][g]
 ax.plot(t['step'],t['loss'],color=C[g],alpha=.18,lw=.6,label='Raw batch loss')
 ax.plot(t['step'],t['loss_ema'],color=C[g],lw=1.8,label='EMA (alpha=0.05)')
 ax.set_title(f'{g} | {D["labels"][g]}',fontsize=12)
 ax.set_ylim(bottom=0);ax.grid();ax.legend(fontsize=9)
 ax.set_xlabel('Optimizer update');ax.set_ylabel('Flow-matching training loss')
fig.suptitle('Training dynamics | every recorded update, 1-3000',fontsize=19,fontweight='bold')
fig.subplots_adjust(hspace=.36,top=.88,bottom=.10)
foot(fig,'No held-out validation in train500. Shared y-axis; smoothing is for display only.')
save(fig,'loss-raw')

fig,ax=plt.subplots(figsize=(11,5))
for g in 'ABCD': ax.plot(D['training'][g]['step'],D['training'][g]['loss_ema'],label=g,color=C[g],lw=1.8)
ax.set(title='Training loss comparison',xlabel='Optimizer update',ylabel='Training loss (EMA, alpha=0.05)')
ax.set_ylim(bottom=0);ax.grid();ax.legend(ncol=4)
fig.subplots_adjust(bottom=.18);foot(fig,'All 3000 points per run. This is not validation loss and is not a closed-loop success metric.')
save(fig,'loss-comparison')

fig,axes=plt.subplots(1,2,figsize=(12,4.8))
t=D['training']['C'];axes[0].plot(t['step'],t['lr'],color=C['C'],lw=2)
axes[0].set(title='Shared learning-rate schedule',xlabel='Optimizer update',ylabel='Learning rate');axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[0].grid()
for g in 'ABCD':
 v=D['training'][g]['grad_norm'];sm=[]
 for x in v:sm.append(x if not sm else .05*x+.95*sm[-1])
 axes[1].plot(t['step'],sm,color=C[g],label=g,lw=1.5)
axes[1].set(title='Gradient norm (EMA)',xlabel='Optimizer update',ylabel='Recorded gradient norm');axes[1].legend(ncol=4);axes[1].grid()
fig.subplots_adjust(bottom=.20,wspace=.25);foot(fig,'Schedule and gradients come from metrics.jsonl; no synthetic interpolation or new training.')
save(fig,'optimization')

fig,ax=plt.subplots(figsize=(10,5))
groups=D['groups']; rates=[100*D['evaluation'][g]['success_rate'] for g in groups]
ax.bar(groups,rates,color=[C[g] for g in groups],width=.6)
for g,y in zip(groups,rates):ax.text(g,y+2,f'{y:.1f}%\n{D["evaluation"][g]["successes"]}/500',ha='center',fontsize=11)
ax.set(title='LIBERO Spatial | official 10 tasks x 50 initial states',ylabel='Closed-loop success (%)',ylim=(0,112))
ax.set_yticks([0,20,40,60,80,100]);ax.yaxis.grid()
fig.subplots_adjust(bottom=.16);foot(fig,'Same evaluation manifest and control protocol. E was deferred and has no train500 result.')
save(fig,'success-rates')

fig,ax=plt.subplots(figsize=(12,4.6))
m=np.array([[D['evaluation'][g]['per_task'][str(i)]['success_rate']*100 for i in range(10)] for g in groups])
im=ax.imshow(m,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
ax.set_xticks(range(10),[f'T{i}' for i in range(10)]);ax.set_yticks(range(5),groups)
for j in range(5):
 for i in range(10):ax.text(i,j,f'{m[j,i]:.0f}%',ha='center',va='center',color='white' if m[j,i]>55 else '#17283c')
ax.set_title('Per-task closed-loop success | 50 episodes per cell')
fig.colorbar(im,ax=ax,label='Success (%)',fraction=.03,pad=.02)
fig.subplots_adjust(bottom=.17);foot(fig,'Task IDs follow the pinned LIBERO Spatial task order. Full language prompts are listed in the demo.')
save(fig,'task-heatmap')

fig,ax=plt.subplots(figsize=(10,5.2))
for g in 'ABCD':
 x=D['training'][g]['parameters']['trainable']/1e6;y=D['evaluation'][g]['success_rate']*100
 ax.scatter(x,y,s=150,c=C[g],zorder=4)
 ax.annotate(f'{g}: {x:.1f}M / {y:.1f}%',(x,y),xytext=(0,-24 if g=='C' else 12),textcoords='offset points',ha='center')
ax.set_xscale('log');ax.set_xlim(12,1600);ax.set_ylim(-5,115);ax.set(title='Trainable parameters vs. closed-loop outcome',xlabel='Trainable parameters (millions, log scale)',ylabel='Success (%)');ax.grid()
fig.subplots_adjust(bottom=.18);foot(fig,'Parameter count is not memory or wall-clock cost. Scope changes also affect parameter storage precision.')
save(fig,'parameter-tradeoff')

fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for ax,(pair,counts) in zip(axes,D['paired'].items()):
 a,b=pair.split('_');vals=list(counts.values());labels=['Both succeed',f'{a} only',f'{b} only','Both fail']
 ax.barh(labels,vals,color=['#566579',C[a],C[b],'#b9c2cc'])
 for i,v in enumerate(vals):ax.text(v+3,i,str(v),va='center')
 ax.set_xlim(0,520);ax.set_title(f'{a} vs {b} | paired outcomes');ax.set_xlabel('Episodes');ax.invert_yaxis()
fig.subplots_adjust(wspace=.36,bottom=.18);foot(fig,'C and D disagree on 30 initial states: C wins 14, D wins 16. Overall D leads by 2 episodes.')
save(fig,'paired-outcomes')

old={r['group']:float(r['success_rate'])*100 for r in D['historical']}
fig,ax=plt.subplots(figsize=(10,5));x=np.arange(5)
ax.bar(x-.18,[old[g] for g in groups],.36,color='#bac6d4',label='450 training demonstrations')
ax.bar(x+.18,rates,.36,color=[C[g] for g in groups],label='500 training demonstrations')
ax.set_xticks(x,groups);ax.set(title='Two training recipes | same official 500-episode protocol',ylabel='Success (%)',ylim=(0,115));ax.legend();ax.yaxis.grid()
for i,g in enumerate(groups):ax.text(i+.18,rates[i]+1,f'{rates[i]:.1f}',ha='center',fontsize=9)
fig.subplots_adjust(bottom=.18);foot(fig,'Training data AND normalization differ. This does not isolate a causal effect of adding 50 demonstrations.')
save(fig,'recipe-comparison')

fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for ax,key,title in [(axes[0],'last100_mean','Mean training loss | final 100 updates'),(axes[1],'median_update_seconds','Median update time | observed runs')]:
 vals=[D['training'][g][key] for g in 'ABCD'];ax.bar(list('ABCD'),vals,color=[C[g] for g in 'ABCD'])
 for g,v in zip('ABCD',vals):ax.text(g,v*1.025,f'{v:.5f}' if key=='last100_mean' else f'{v:.2f}s',ha='center',fontsize=10)
 ax.set_title(title,fontsize=13);ax.set_ylim(0,max(vals)*1.22);ax.yaxis.grid()
fig.subplots_adjust(bottom=.21,wspace=.3);foot(fig,'Loss: batch average, not full-dataset validation. Timing: excludes compilation steps 1 and 3; not a controlled speed benchmark.')
save(fig,'loss-and-time')
print('Rendered 9 figures in PNG + SVG:',OUT)
