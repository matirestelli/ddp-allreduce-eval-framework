"""Discover each experiment matrix; render all available configurations once."""
from __future__ import annotations
import argparse,copy,json,html,sys,hashlib,collections
from pathlib import Path
from .cli import parser,parse,slug
from .data import load_inputs,export_runs
from .render import create,select,workload,NoData

def run(system,argv=None):
    p=parser(description=f'Generate the full {system.title()} thesis plotting tree. Inputs may be code_oct26, system roots, raw files, or exported CSV.')
    p.add_argument('--output-root',type=Path,default=Path('plots_thesis'))
    p.add_argument('--scopes',nargs='+',choices=['all','fixed'],default=['all','fixed'],help='all: GPU sweeps; fixed: each GPU count, plus varying-batch plots at fixed GPU count')
    p.add_argument('--metrics',nargs='+',choices=['time','hook','tail','tail_pct'],default=['time','hook','tail'])
    p.add_argument('--layouts',nargs='+',choices=['bars','lines','hbar'],default=['bars','lines','hbar'])
    p.add_argument('--normalizations',nargs='+',choices=['absolute','baseline','speedup','efficiency'],default=['absolute'])
    p.add_argument('--comparisons',action=argparse.BooleanOptionalAction,default=True,help='Also generate matching MPI/native-backend comparison panels')
    p.add_argument('--max-plots',type=int,help='Optional preview limit; manifest reports truncation')
    p.add_argument('--dry-run',action='store_true',help='Write planned jobs and audit only')
    p.add_argument('--skip-existing',action='store_true',help='Skip a figure when its JSON sidecar exists')
    args=parse(p,argv);args.system=[system]
    runs,issues,allruns=load_inputs(args.inputs,system=args.assume_system,mode=args.mode,include_partial=args.include_partial)
    if args.lr is not None:runs=[r for r in runs if r['lr']==args.lr]
    runs=select(runs,args);runs=[r for r in runs if r['mode'] in ('strong','weak')]
    base=args.output_root/system;base.mkdir(parents=True,exist_ok=True);export_runs([r for r in allruns if r['system']==system],base/'parsed_runs.csv');(base/'input_issues.json').write_text(json.dumps(issues,indent=2))
    groups={}
    for r in runs:
        key=(r['family'],r['mode'],r['backend'])+workload(r,'ranks')+((r['lr'] if r['mode']=='strong' else None),)
        groups.setdefault(key,[]).append(r)
    jobs=[]
    def add(rr,scope,x,tag,layouts):
        r=rr[0]
        for metric in args.metrics:
            for layout in layouts:
                # Default tails compare two bars at a single configuration.
                if metric=='tail' and args.tail_context and layout!='hbar':continue
                if layout=='hbar' and len({(z['effective_batch'] if x=='global_batch' else z[x]) for z in rr})>1:continue
                for norm in args.normalizations:
                    if norm!='absolute' and metric!='time':continue
                    if norm in ('speedup','efficiency') and (layout!='lines' or r['mode']!='strong' or x!='ranks' or scope=='fixed'):continue
                    category='normal' if metric=='time' and layout!='lines' else ('lines' if metric=='time' else 'hooks' if metric=='hook' else 'tails')
                    backend_folder='backend_comparison' if len({z['backend'] for z in rr})>1 else r['backend']
                    folder=base/r['family']/(r['mode']+'Scaling')/backend_folder/category
                    unit=('epoch' if r['mode']=='strong' else 'iter') if args.unit=='auto' else args.unit
                    stem=folder/(tag+'_'+scope+'_'+layout+'_'+metric+'_'+(unit if metric!='time' else norm))
                    jobs.append((rr,dict(scope=scope,x=x,metric=metric,layout=layout,normalize=norm,stem=str(stem))))
    for key,rr in sorted(groups.items(),key=lambda kv:str(kv[0])):
        r=rr[0];tag=('GB'+str(r['effective_batch'])) if r['mode']=='strong' else ('LB'+str(r['batch_per_rank']))
        if 'all' in args.scopes:add(rr,'all','ranks',tag,[l for l in args.layouts if l!='hbar'])
        if 'fixed' in args.scopes:
            for gpu in sorted({z['ranks'] for z in rr}):add([z for z in rr if z['ranks']==gpu],'fixed','ranks',tag+'_gpu'+str(gpu),[l for l in args.layouts if l!='lines'])
    if args.comparisons:
        compare_groups={}
        for r in runs:
            key=workload(r,'ranks')+((r['lr'] if r['mode']=='strong' else None),)
            compare_groups.setdefault(key,[]).append(r)
        for key,rr in sorted(compare_groups.items(),key=lambda kv:str(kv[0])):
            if len({r['backend'] for r in rr})<2:continue
            r=rr[0];tag=('GB'+str(r['effective_batch'])) if r['mode']=='strong' else ('LB'+str(r['batch_per_rank']))
            if 'all' in args.scopes:add(rr,'all','ranks',tag,[l for l in args.layouts if l!='hbar'])
            if 'fixed' in args.scopes:
                for gpu in sorted({z['ranks'] for z in rr}):
                    fixed=[z for z in rr if z['ranks']==gpu]
                    if len({z['backend'] for z in fixed})>1:add(fixed,'fixed','ranks',tag+'_gpu'+str(gpu),[l for l in args.layouts if l!='lines'])
    # At fixed GPU count, compare different batch sizes without mixing models.
    if 'fixed' in args.scopes:
        batchgroups={}
        for r in runs:
            key=(r['system'],r['family'],r['model'],r['dataset'],r['image_size'],r['mode'],r['backend'],r['ranks'],r.get('block_size'))
            batchgroups.setdefault(key,[]).append(r)
        for key,rr in sorted(batchgroups.items(),key=lambda kv:str(kv[0])):
            x='global_batch' if rr[0]['mode']=='strong' else 'batch_per_rank'
            if len({r['effective_batch'] if x=='global_batch' else r[x] for r in rr})>1:add(rr,'fixed',x,'gpu'+str(rr[0]['ranks'])+'_vary_batch',[l for l in args.layouts if l!='hbar'])
    # Disambiguate models/datasets/block sizes sharing short batch tags.
    counts=collections.Counter(j['stem'] for rr,j in jobs)
    for rr,j in jobs:
        if counts[j['stem']]>1:
            configs=sorted({str((r['model'],r['dataset'],r['image_size'],r.get('block_size'),r['lr'] if r['mode']=='strong' else None)) for r in rr})
            suffix=hashlib.sha256(repr(configs).encode()).hexdigest()[:8]
            j['stem']+='__cfg'+suffix
    truncated=args.max_plots is not None and len(jobs)>args.max_plots
    if args.max_plots is not None:jobs=jobs[:args.max_plots]
    report=dict(system=system,inputs=[str(Path(x).resolve()) for x in args.inputs],completed_runs=len(runs),all_runs=len([r for r in allruns if r['system']==system]),planned=len(jobs),truncated=truncated,dry_run=args.dry_run,plots=[])
    for i,(rr,j) in enumerate(jobs):
        item=j.copy();stem=Path(j['stem'])
        if args.dry_run:item['status']='planned'
        elif args.skip_existing and stem.with_suffix('.json').exists():item['status']='existing'
        else:
            aa=copy.copy(args)
            for k in ['x','metric','layout','normalize']:setattr(aa,k,j[k])
            try:
                result=create(rr,aa,stem);item.update(status='written',points=len(result['points']),omitted=len(result['omitted']))
            except NoData as exc:item.update(status='no_metric',reason=str(exc))
            except (ValueError,KeyError) as exc:item.update(status='error',reason=str(exc))
        report['plots'].append(item)
        if (i+1)%20==0:print(f'{system}: {i+1}/{len(jobs)} figures processed',flush=True)
    if not runs:report['notice']='No matching runs; verify input path and system names. No fabricated figures generated.'
    (base/'manifest.json').write_text(json.dumps(report,indent=2))
    links=[]
    for j in report['plots']:
        if j['status'] in ('written','existing'):
            path=Path(j['stem']).relative_to(base).as_posix()+'.html';links.append(f'<li><a href="{html.escape(path)}">{html.escape(path[:-5])}</a></li>')
    (base/'index.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{system.title()} thesis plots</title><style>body{{font:18px/1.5 Arial;max-width:70rem;margin:2rem auto;padding:1rem;background:white;color:#111}}a{{color:#0645ad}}</style><h1>{system.title()} thesis plots</h1><p>{len(runs)} completed/selected runs. See manifest.json and input_issues.json for unavailable metrics, incomplete jobs and provenance. These figure packages do not certify the final thesis document.</p><ul>'+''.join(links)+'</ul></html>')
    from collections import Counter
    counts=Counter(j['status'] for j in report['plots']);print(f'{system}: {dict(counts)}; index: {base/"index.html"}')
    return report
