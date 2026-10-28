"""Shared bars, horizontal bars, scaling lines, provenance and descriptions."""
from __future__ import annotations
import csv,html,json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from . import style as st
from .data import value,signature

class NoData(ValueError):pass

def metric_label(metric,mode,unit):
    unit=('epoch' if mode=='strong' else 'iter') if unit=='auto' else unit
    if metric=='time':return 'Train-only epoch time (s)' if mode=='strong' else 'Median iteration time (ms)'
    if metric=='tail_pct':return 'Exposed tail / training epoch time (%)'
    stem='Cumulative hook work' if metric=='hook' else 'Exposed tail time'
    return stem+{'epoch':' / epoch (s)','iter':' / optimizer step (ms)','call':' / hook call (ms)'}[unit]

def transform_direct(runs,policy):
    out=[]
    for r in runs:
        r=r.copy();m=r['method']
        if m.startswith('direct_t'):
            t=int(m.split('_t')[-1])
            if policy=='omit':continue
            if policy=='gpu':
                if t!=r['ranks']:continue
                r['method']='direct_p'
        out.append(r)
    return out

def select(runs,args):
    rr=[]
    for r in runs:
        if args.system and r['system'] not in args.system:continue
        if args.mode and r['mode']!=args.mode:continue
        if args.model and not any(x.lower() in (r['model']+' '+r['family']).lower() for x in args.model):continue
        if args.backend and r['backend'] not in args.backend:continue
        if args.gpus and r['ranks'] not in args.gpus:continue
        if args.global_batch and r['effective_batch'] not in args.global_batch:continue
        if args.batch_per_rank and r['batch_per_rank'] not in args.batch_per_rank:continue
        if not args.diagnostics and r['algorithm'].startswith('default'):continue
        if args.rates and r['zfp_rate'] and str(r['zfp_rate']) not in args.rates:continue
        rr.append(r)
    rr=transform_direct(rr,args.direct)
    if args.methods:rr=[r for r in rr if r['method'] in args.methods]
    if args.families:rr=[r for r in rr if st.family(r['method']).lower() in args.families]
    if args.prefer_rate8:
        have={signature(r,False)[:-1] for r in rr if r['method']=='ring_zfp_online_coll:8'}
        rr=[r for r in rr if not(r['method']=='ring_zfp_online_coll:10' and signature(r,False)[:-1] in have)]
    return rr

def workload(r,x):
    # Varying x is intentional, but neither model/resolution nor effective
    # workload may silently change. Strong GPT uses accumulated effective batch.
    fixed=r['effective_batch'] if r['mode']=='strong' else r['batch_per_rank']
    if x in ('global_batch','batch_per_rank'):fixed=None
    return (r['model'],r['dataset'],r['image_size'],r['mode'],fixed,r.get('block_size'))

def panel_key(r):return f'{r["system"].title()} · '+('RCCL' if r['system']=='frontier' and r['backend']=='nccl' else r['backend'].upper())
def xvalue(r,x):return r['effective_batch'] if x=='global_batch' else r[x]
def aggregate(runs,metric,unit,x,repeat):
    grouped={};omitted=[]
    for r in runs:
        v=value(r,metric,unit)
        if v is None:omitted.append(dict(id=r['id'],method=r['method'],ranks=r['ranks'],reason='metric not recorded (not zero)',source=r['source']));continue
        key=(panel_key(r),xvalue(r,x),r['method']);grouped.setdefault(key,[]).append((r,v))
    points=[]
    for (panel,xx,m),rv in grouped.items():
        if repeat=='latest':rv=[max(rv,key=lambda t:(Path(t[0]['source']).stat().st_mtime if Path(t[0]['source']).exists() else 0,t[0]['source'],t[0]['source_line']))]
        points.append(dict(panel=panel,x=xx,method=m,value=statistics.median(v for r,v in rv),n_runs=len(rv),sources=[dict(source=r['source'],line=r['source_line'],id=r['id'],raw_hook=r['raw_hook'],epochs=r['epochs'],observed_epochs=r['observed_epochs'],complete=r['complete'],warnings=r.get('warnings',[]),native_fallback=r['native_fallback']) for r,v in rv],run_values=[v for r,v in rv]))
    return points,omitted

def tail_pair_values(r,unit='auto'):
    """Matched post-warmup epochs; both bars use the same wall-time basis.

    Iteration context is mean train-only time per measured optimizer step,
    rather than the unrelated median iteration latency in the timing summary.
    """
    unit=('epoch' if r['mode']=='strong' else 'iter') if unit=='auto' else unit
    if unit=='call':raise ValueError('Paired tails require epoch or iter units, not hook calls')
    pairs=[]
    for e in r['epochs_data']:
        if e['epoch']<=1 or e.get('train_s') is None or e.get('tail_total_ms') is None:continue
        if unit=='epoch':train,tail=e['train_s'],e['tail_total_ms']/1000
        elif e.get('steps'):train,tail=e['train_s']*1000/e['steps'],e['tail_total_ms']/e['steps']
        else:continue
        if math.isfinite(train) and math.isfinite(tail):pairs.append((train,tail))
    if not pairs:return None
    return tuple(statistics.median(p[i] for p in pairs) for i in (0,1))

def aggregate_tail_pairs(runs,unit,x,repeat):
    matched=[];omitted=[]
    for r in runs:
        pair=tail_pair_values(r,unit)
        if pair is None:
            omitted.append(dict(id=r['id'],method=r['method'],ranks=r['ranks'],reason='matched train-only time and tail not recorded (not zero)',source=r['source']))
            continue
        # Feed matched-epoch tail values through the same repeat selection used
        # for ordinary figures; attach train values from precisely those runs.
        clone=r.copy();clone['epochs_data']=[dict(epoch=2,tail_total_ms=pair[1]*1000)]
        clone['_training_value']=pair[0];clone['_tail_value']=pair[1];matched.append(clone)
    points,_=aggregate(matched,'tail','epoch',x,repeat)
    by_id={r['id']:r for r in matched}
    for p in points:
        rv=[by_id[src['id']] for src in p['sources']]
        p['training_value']=statistics.median(r['_training_value'] for r in rv)
        p['training_run_values']=[r['_training_value'] for r in rv]
    return points,omitted

def create(runs,args,output):
    if not runs:raise NoData('No matching completed runs')
    signatures={workload(r,args.x) for r in runs}
    if len(signatures)>1:raise ValueError('Selection contains different models/resolutions, modes, fixed workloads or block sizes. Filter further or use batch runners, which split these into separate figures.')
    if args.x in ('global_batch','batch_per_rank') and len({r['ranks'] for r in runs})>1:raise ValueError('Varying batch on x requires one --gpus value')
    if args.x=='ranks' and len({r['lr'] for r in runs})>1 and runs[0]['mode']=='strong' and args.x=='ranks':raise ValueError('Strong-scaling selection has different learning rates; use --lr or separate inputs')
    mode=runs[0]['mode']
    if mode not in ('strong','weak'):raise ValueError('Specify --mode strong or weak for inputs outside named scaling folders')
    metric=args.metric;unit=args.unit;ylabel=metric_label(metric,mode,unit)
    paired=metric=='tail' and args.layout=='hbar' and getattr(args,'tail_context',True)
    if paired and args.normalize!='absolute':raise ValueError('Paired training/tail bars require absolute time units')
    if paired:
        points,omitted=aggregate_tail_pairs(runs,unit,args.x,args.repeats)
        ylabel='Time per epoch (s)' if (unit=='epoch' or (unit=='auto' and mode=='strong')) else 'Mean time per optimizer step (ms)'
    else:points,omitted=aggregate(runs,metric,unit,args.x,args.repeats)
    if not points:raise NoData(f'No recorded {metric} measurements after epoch 1')
    panels=sorted({p['panel'] for p in points});methods=sorted({p['method'] for p in points},key=st.sort_key);xs=sorted({p['x'] for p in points});lookup={(p['panel'],p['x'],p['method']):p['value'] for p in points}
    layout=args.layout
    if layout=='hbar' and len(xs)!=1:raise ValueError('Horizontal method comparison needs one x value: specify --gpus (or one batch for batch x)')
    if layout=='lines' and args.normalize in ('speedup','efficiency') and (mode!='strong' or args.x!='ranks'):raise ValueError('Scaling speedup/efficiency require strong mode and ranks on x')
    if args.normalize=='baseline':
        for p in points:
            b=lookup.get((p['panel'],p['x'],'none'))
            if b is None:raise ValueError('Baseline normalization requires a matching built-in baseline at every x')
            p['value']/=b
        ylabel='Time / matching built-in baseline (×)'
    elif args.normalize in ('speedup','efficiency'):
        ref=args.reference_gpus or min(xs)
        for p in points:
            b=lookup.get((p['panel'],ref,p['method']))
            if b is None:raise ValueError(f'Missing {p["method"]} at reference GPU count {ref}; filter methods or change reference')
            p['value']=b/p['value']/(p['x']/ref if args.normalize=='efficiency' else 1)
        ylabel=f'{"Parallel efficiency" if args.normalize=="efficiency" else "Speedup"} relative to {ref} GPUs'
    lookup={(p['panel'],p['x'],p['method']):p['value'] for p in points}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':args.font_size,'axes.labelsize':args.font_size,'axes.titlesize':args.font_size+1,'axes.spines.top':False,'axes.spines.right':False,'hatch.linewidth':.85,'svg.fonttype':'none','pdf.fonttype':42})
    cols=min(len(panels),args.columns);rows=math.ceil(len(panels)/cols)
    width=args.width;panel_width=(width-.8)/cols
    if panel_width<4.6:raise ValueError('Panels too narrow for readable text: reduce --columns or increase --width')
    families=[f for f in st.ORDER if any(st.family(m)==f for m in methods)]
    legend_cols = min(len(families), max(1, int((width - 1) / 2.4)))
    legend_rows = 0 if paired else math.ceil(len(families) / legend_cols)
    legend_inches=.55 if paired else 0
    for row in range(legend_rows):
        fs=families[row*legend_cols:(row+1)*legend_cols]
        legend_inches+=.35+max(sum(st.family(m)==f for m in methods) for f in fs)*.22
    panel_height=max(2.8,(.53 if paired else .26)*len(methods)) if layout=='hbar' else 3.0
    xlabel_inches=(1.05 if args.family_labels and layout=='bars' else .58)
    height=args.height or (.85+rows*panel_height+(rows-1)*1.1+xlabel_inches+legend_inches+.45)
    bottom=(legend_inches+.45+xlabel_inches)/height;top=1-.85/height
    fig,axs=plt.subplots(rows,cols,figsize=(width,height),squeeze=False)
    fig.subplots_adjust(left=(.43 if layout=='hbar' else .9/width),right=.98,top=top,bottom=bottom,hspace=.72,wspace=.28)
    xlab={'ranks':'GPU count (ranks)','global_batch':'Effective global batch','batch_per_rank':'Batch / rank'}[args.x]
    for ai,ax in enumerate(axs.flat):
        if ai>=len(panels):ax.set_visible(False);continue
        if layout == 'hbar':
            row,col=divmod(ai,cols);cell_width=width/cols
            if cell_width<8:raise ValueError('Paired tail panels need at least 8 physical inches each; increase --width or reduce --columns')
            available=top-bottom;cell_height=(available-(rows-1)*1.1/height)/rows
            ax.set_position([(col*cell_width+3.8)/width,top-(row+1)*cell_height-row*1.1/height,(cell_width-4.1)/width,cell_height])
        panel=panels[ai];present=[m for m in methods if any((panel,x,m) in lookup for x in xs)]
        if layout=='lines':
            for i,m in enumerate(present):
                s=st.style(m,args.palette=='shades');ys=[lookup.get((panel,x,m),float('nan')) for x in xs]
                ax.plot(xs,ys,color=s['color'],marker=s['marker'],ls=s['linestyle'],lw=2.7 if m=='none' else 1.8,ms=6,label=st.full_label(m))
            ax.set_xticks(xs);ax.set_xlabel(xlab);ax.set_ylabel(ylabel);ax.grid(axis='y',color='#d0d0d0',lw=.6)
            if args.xscale=='log2':ax.set_xscale('log',base=2);ax.set_xticks(xs,labels=[str(x) for x in xs])
        elif layout=='hbar':
            yy=[];texts=[];cursor=0
            for f in families:
                for m in [m for m in present if st.family(m)==f]:
                    v=lookup.get((panel,xs[0],m));s=st.style(m,args.palette=='shades')
                    if paired:
                        train=next(p['training_value'] for p in points if p['panel']==panel and p['x']==xs[0] and p['method']==m)
                        ax.barh(cursor-.21,train,color='#D9D9D9',edgecolor='black',hatch='..',lw=.7,height=.38)
                    yy_tail=cursor+.21 if paired else cursor
                    hh=.38 if paired else .75
                    ax.barh(yy_tail,v,color=s['color'],edgecolor='white',hatch=s['hatch'],lw=0,height=hh);ax.barh(yy_tail,v,facecolor='none',edgecolor='black',lw=.7,height=hh)
                    yy.append(cursor);texts.append(st.full_label(m));cursor+=1
                cursor+=.4
            ax.set_yticks(yy,texts);ax.invert_yaxis();ax.set_xlabel(ylabel);ax.grid(axis='x',color='#d0d0d0',lw=.6)
        else:
            groups=[(f,[m for m in present if st.family(m)==f]) for f in families];groups=[(f,ms) for f,ms in groups if ms];span=len(present)+.8*(len(groups)-1);w=.88/span;cursor=-span/2
            for f,ms in groups:
                start=cursor
                for m in ms:
                    off=(cursor+.5)*w;cursor+=1;s=st.style(m,args.palette=='shades')
                    for j,x in enumerate(xs):
                        v=lookup.get((panel,x,m))
                        if v is None:continue
                        ax.bar(j+off,v,width=w*.94,color=s['color'],edgecolor='white',hatch=s['hatch'],lw=0,zorder=3);ax.bar(j+off,v,width=w*.94,facecolor='none',edgecolor='black',lw=.7,zorder=4)
                center=((start+cursor)/2)*w
                if args.family_labels:
                    for j in range(len(xs)):ax.text(j+center,-.075,st.abbreviation(f),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=args.font_size)
                cursor+=.8
            ax.set_xticks(range(len(xs)),xs);ax.tick_params(axis='x',pad=40 if args.family_labels else 6);ax.set_xlabel(xlab,labelpad=8);ax.set_ylabel(ylabel);ax.grid(axis='y',color='#d0d0d0',lw=.6);ax.set_xlim(-.55,len(xs)-.45)
        ax.set_axisbelow(True);ax.set_title(panel+(f' · {xlab} {xs[0]}' if layout=='hbar' else ''),pad=10)
        if layout=='hbar':ax.set_xlim(left=0)
        else:ax.set_ylim(bottom=0)
        if args.normalize in ('baseline','efficiency'):
            (ax.axvline if layout=='hbar' else ax.axhline)(1,color='black',ls='--',lw=1)
        if args.ymax:
            if layout=='hbar':ax.set_xlim(0,args.ymax)
            else:ax.set_ylim(0,args.ymax)
    if layout == 'hbar' and args.share_y:
        upper=max(ax.get_xlim()[1] for ax in axs.flat if ax.get_visible())
        for ax in axs.flat:
            if ax.get_visible():ax.set_xlim(0,upper)
    if args.share_y and layout!='hbar':
        upper=max(ax.get_ylim()[1] for ax in axs.flat if ax.get_visible())
        for ax in axs.flat:
            if ax.get_visible():ax.set_ylim(0,upper)
    desc=workload(runs[0],args.x);fixed=desc[4]
    workload_text=('effective global batch' if mode=='strong' else 'local batch')+f' {fixed}' if fixed is not None else f'fixed GPU count {runs[0]["ranks"]}'
    display_family={'wideresnet32':'WideResNet 32×32','wideresnet224':'WideResNet 224×224','gpt':'GPT'}.get(runs[0]['family'],runs[0]['family'])
    title=args.title or f'{display_family} · {mode} scaling · {workload_text}'
    fig.text(.9/width,1-.13/height,title,fontsize=args.font_size+3,fontweight='bold',va='top')
    fig.text(.9/width,1-.47/height,'Lower is better' if args.normalize not in ('speedup','efficiency') else 'Higher is better',fontsize=args.font_size,va='top')
    if paired:
        fig.legend([Patch(facecolor='#D9D9D9',edgecolor='black',hatch='..'),Patch(facecolor='#145C32',edgecolor='white',hatch='///')],['Train-only time','Exposed tail (family color and pattern)'],loc='lower center',bbox_to_anchor=(.5,.39/height),ncol=1,frameon=False,fontsize=args.font_size)
    # Legends use explicit family headings and carry the same patterns as bars.
    y=(legend_inches+.36)/height
    for row in range(legend_rows):
        fs=families[row*legend_cols:(row+1)*legend_cols];rowheight=.35+max(sum(st.family(m)==f for m in methods) for f in fs)*.22
        for col,f in enumerate(fs):
            ms=[m for m in methods if st.family(m)==f];handles=[]
            for m in ms:
                s=st.style(m,args.palette=='shades')
                if layout=='lines':
                    from matplotlib.lines import Line2D
                    handles.append(Line2D([],[],color=s['color'],marker=s['marker'],ls=s['linestyle'],lw=1.8))
                else:handles.append(Patch(facecolor=s['color'],edgecolor='white',hatch=s['hatch']))
            leg=fig.legend(handles,[st.label(m) for m in ms],title=f,loc='upper left',bbox_to_anchor=(.9/width+col*(width-1)/legend_cols/width,y),frameon=False,fontsize=args.font_size,title_fontsize=args.font_size,handlelength=2.4,borderaxespad=0,labelspacing=.45);leg.get_title().set_fontweight('bold')
        y-=rowheight/height
    fig.text(.9/width,.13/height,'Epoch 1 excluded · missing metrics are not zero · r = ZFP rate',fontsize=args.font_size,va='bottom')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for fmt in args.formats:fig.savefig(output.with_suffix('.'+fmt),dpi=args.dpi)
    plt.close(fig)
    contrast=st.contrast_report(methods,args.palette=='shades')
    provenance=dict(tail_context=paired,training_context_basis='matched post-warmup epochs; mean time per measured optimizer step for iter units' if paired else None,title=title,metric=metric,unit=unit,layout=layout,width_inches=width,height_inches=height,min_font_pt=args.font_size,mode=mode,x=args.x,normalize=args.normalize,repeats=args.repeats,points=points,omitted=omitted,contrast=contrast,notes=['Hook work is cumulative and can overlap; do not stack it as disjoint wall time.','Tail is the measured exposed tail, not the entire communication cost.','Duplicate aliases and genuinely repeated measurements retain source provenance.','Figure design checks do not certify the final thesis PDF/EPUB.'])
    output.with_suffix('.json').write_text(json.dumps(provenance,indent=2))
    with output.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['panel','x','method','value','n_runs']+(['training_value'] if paired else []));w.writeheader();w.writerows({k:p[k] for k in w.fieldnames} for p in points)
    trend=[]
    for panel in panels:
        for x in xs:
            pp=[p for p in points if p['panel']==panel and p['x']==x];best=min(pp,key=lambda p:p['value']) if args.normalize not in ('speedup','efficiency') else max(pp,key=lambda p:p['value'])
            trend.append(f'{panel}, {xlab} {x}: '+st.full_label(best['method'])+f' has the {"lowest" if args.normalize not in ("speedup","efficiency") else "highest"} plotted value, {best["value"]:.3f}.')
    alt=f'{title}. {ylabel} is compared across {xlab.lower()} values '+', '.join(map(str,xs))+'. '+('Panels compare '+', '.join(panels)+'. ' if len(panels)>1 else panels[0]+'. ')+ 'A detailed description and all plotted values follow.'
    long=f'The figure uses {layout}. '+('Families and methods are identified by text and patterns in addition to color. ' if layout!='lines' else 'Methods use colors, line styles and markers. ')+f'The metric is {ylabel}. '+ ' '.join(trend)+' Epoch 1 is excluded. Values are medians; when repeated runs are available the selected repeat policy is '+args.repeats+'. These are not confidence intervals across repeated jobs. Missing or unrecorded metrics are omitted, not set to zero. '+('Hook work is cumulative instrumented work; it can overlap and is not an additive wall-time partition. ' if metric=='hook' else '')+('Exposed tail is the instrumented tail, not total communication time. ' if metric.startswith('tail') else '')
    if paired:
        long+=' Each method has two separate horizontal bars starting at zero: a light gray dotted bar for train-only time and a family-colored patterned bar for exposed tail. They are not stacked. Both bars use the same post-warmup epochs and repeat selection. In iteration units, both are means per measured optimizer step, not median iteration latency. Shared horizontal limits support comparison across panels. The baseline is omitted when tail instrumentation is unavailable.'
    output.with_suffix('.description.md').write_text(f'# {title}\n\n## Short alternative\n\n{alt}\n\n## Detailed description\n\n{long}\n')
    table_keys=['panel','x','method','value','n_runs']+(['training_value'] if paired else [])
    training_header='<th scope="col">Train-only time</th>' if paired else ''
    trs=''.join('<tr>'+''.join(f'<td>{html.escape(str(p[k]))}</td>' for k in table_keys)+'</tr>' for p in points)
    img=output.name+'.png' if 'png' in args.formats else output.name+'.svg' if 'svg' in args.formats else None
    graphic=f'<figure><img src="{img}" alt="{html.escape(alt,quote=True)}"><figcaption>{html.escape(title)}. <a href="#description">Detailed description and data</a>.</figcaption></figure>' if img else ''
    output.with_suffix('.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title><style>body{{font:18px/1.5 Arial;max-width:65rem;margin:2rem auto;padding:1rem;background:white;color:#111}}img{{max-width:100%}}a{{color:#0645ad}}table{{border-collapse:collapse}}td,th{{border:1px solid #333;padding:.4rem;text-align:left}}</style><h1>{html.escape(title)}</h1>{graphic}<h2 id="description">Detailed description</h2><p>{html.escape(long)}</p><table><caption>All plotted values; {html.escape(ylabel)}</caption><thead><tr><th scope="col">Panel</th><th scope="col">{html.escape(xlab)}</th><th scope="col">Method</th><th scope="col">Value</th><th scope="col">Runs</th>{training_header}</tr></thead><tbody>{trs}</tbody></table></html>')
    return provenance
