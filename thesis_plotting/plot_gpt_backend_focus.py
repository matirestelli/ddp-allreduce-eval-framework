#!/usr/bin/env python3
"""Two GPT backend comparisons per system/workload using your existing parser/style.

1. Scaling LINES: baseline; MPI Ring/RD online ZFP rate8; native NCCL/RCCL
   Hierarchical online ZFP rate8; k-ring k4 on Polaris or k8 on Frontier.
2. 32-GPU BARS: all available algorithms (including all fixed-t Direct variants).

Run from the project root on either supercomputer (paths are examples):

  python3 thesis_plotting/plot_gpt_backend_focus.py \
      experiments_frontier experiments_frontier_rccl \
      --system frontier --output-root plots_thesis/gpt_backend_focus

  python3 thesis_plotting/plot_gpt_backend_focus.py \
      experiments_polaris experiments_polaris_nccl \
      --system polaris --output-root plots_thesis/gpt_backend_focus

  # Both machines, separate figures for each:
  python3 thesis_plotting/plot_gpt_backend_focus.py \
      experiments_frontier experiments_frontier_rccl \
      experiments_polaris experiments_polaris_nccl

  # Synthetic logs only; filenames need no special completion overrides:
  python3 thesis_plotting/plot_gpt_backend_focus.py simulation_logs/polaris_gpt \
      --output-root plots_simulation/gpt_backend_focus

Defaults: strong scaling, effective accumulated batch768, 32-GPU zoom, seconds
per training epoch. --zoom-gpus changes the bar-plot GPU count. --direct gpu
reduces bar Direct variants to t=GPU count; default is all. --mode, --system,
--backend, --gpus, --lr, --model, --global-batch, --formats, --config and other
shared filters remain available. Filters apply to both views. --out is an
alternative output root. Workloads and systems are never mixed into one plot.
Each figure has the existing PNG/PDF/data/description/HTML companions; one
shared legend underneath and side-by-side panels with common value scales.
Missing requested methods are listed in focus_manifest.json, never substituted.
Use --combine-simulated to join compatible logged 16/32-GPU and simulated
64-GPU files. Synthetic points retain estimate markers and visible labels;
original model/dataset labels remain in focus_manifest.json. No timings change.
The input logs and existing plotting modules are not modified by this script.
"""
from pathlib import Path
import copy, hashlib, html, inspect, json
from thesisplots.cli import parser, parse
from thesisplots.data import load_inputs, export_runs
from thesisplots.render import create, select, workload, NoData


def wanted(run):
    method=run['method']
    kring={'polaris':'kring_k4','frontier':'kring_k8'}.get(run['system'])
    if method=='none' or (kring and method==kring):return True
    if str(run.get('zfp_rate'))!='8':return False
    if run['backend']=='mpi':
        return run['algorithm'] in ('ring_zfp_online_coll','recursive_doubling_zfp_online_coll')
    if run['backend'] in ('nccl','rccl'):
        return run['algorithm']=='hierarchical_zfp_online_coll'
    return False


def expected(system,backend):
    methods=['none',{'polaris':'kring_k4','frontier':'kring_k8'}[system]]
    if backend=='mpi':methods+=['ring_zfp_online_coll:8','recursive_doubling_zfp_online_coll:8']
    else:methods+=['hierarchical_zfp_online_coll:8']
    return methods


def main(argv=None):
    p=parser('time','lines',__doc__)
    p.add_argument('--output-root',type=Path,default=Path('plots_thesis/gpt_backend_focus'))
    p.add_argument('--zoom-gpus',type=int,default=32)
    p.add_argument('--combine-simulated',action='store_true',help='Combine compatible logged and simulated GPT records in one explicitly labeled comparison; never alters timing values')
    if '--include-predicted' not in p._option_string_actions:
        p.add_argument('--include-predicted',action='store_true')
    p.set_defaults(mode='strong',model=['gpt'],global_batch=[768],direct='all',
                   diagnostics=True,columns=2,width=18,metric='time',layout='lines')
    args=parse(p,argv)
    if args.metric!='time' or args.normalize!='absolute':
        p.error('This preset compares absolute training time; use the general scripts for other metrics/normalizations.')
    options=dict(system=getattr(args,'assume_system',None),mode=args.mode,
                 include_partial=getattr(args,'include_partial',False))
    params=inspect.signature(load_inputs).parameters
    if 'include_predicted' in params:options['include_predicted']=args.include_predicted
    elif args.include_predicted:p.error('--include-predicted requires your prediction-aware data.py')
    try:
        runs,issues,allruns=load_inputs(args.inputs,**options)
        if args.combine_simulated:
            # A synthetic record can have status=SIMULATED rather than numeric
            # execution status. Admit complete numeric synthetic epoch coverage
            # explicitly, without rewriting its original completion evidence.
            included={r['id']:r for r in runs}
            for r in allruns:
                simulated='simulated' in (r['model']+' '+r['dataset']).lower()
                rows=[e for e in r['epochs_data'] if e.get('finished') and e.get('train_s') is not None]
                covered={e['epoch'] for e in rows}
                if (simulated and not r.get('quarantined') and r.get('train_summary_s') is not None
                    and covered==set(range(1,r['epochs']+1))):
                    included[r['id']]=r
            runs=[]
            for original in included.values():
                r=copy.deepcopy(original)
                simulated='simulated' in (r['model']+' '+r['dataset']).lower()
                r['comparison_source_kind']='simulated' if simulated else 'logged'
                r['original_model']=r['model'];r['original_dataset']=r['dataset']
                r['model']=r['model'].removesuffix('_simulated')
                r['dataset']=r['dataset'].removeprefix('simulated_')
                if simulated:
                    r['result_kind']='predicted'  # shared renderer's estimate marker
                    r.setdefault('warnings',[]).append('SIMULATED source explicitly included in mixed comparison; original model='+r['original_model']+'; original dataset='+r['original_dataset']+'; timings unchanged.')
                runs.append(r)
        if args.lr is not None:runs=[r for r in runs if r['lr']==args.lr]
        runs=[r for r in select(runs,args) if r.get('family')=='gpt' and r['system'] in ('frontier','polaris')]
        if not runs:raise NoData('No matching GPT records. Supply the results folders, --mode strong and the correct --global-batch.')
        groups={}
        for r in runs:
            key=(r['system'],)+workload(r,'ranks')+(r.get('lr') if r['mode']=='strong' else None,)
            groups.setdefault(key,[]).append(r)
        root=Path(args.out or args.output_root);root.mkdir(parents=True,exist_ok=True)
        report=dict(inputs=[str(Path(x).resolve()) for x in args.inputs],zoom_gpus=args.zoom_gpus,combine_simulated=args.combine_simulated,figures=[],issues=issues)
        counts={s:sum(key[0]==s for key in groups) for s in ('frontier','polaris')}
        for key,rr in sorted(groups.items(),key=lambda kv:str(kv[0])):
            r=rr[0];system=r['system'];batch=('GB'+str(r['effective_batch'])) if r['mode']=='strong' else ('LB'+str(r['batch_per_rank']))
            if counts[system]>1:batch+='__cfg'+hashlib.sha256(repr(key).encode()).hexdigest()[:8]
            base=root/system/'gpt'/(r['mode']+'Scaling')/'backend_comparison'
            base.mkdir(parents=True,exist_ok=True)
            export_runs(rr,base/(batch+'_input_runs.csv'))
            for view,selected,layout,sub,tag in [
                ('selected_scaling',[z for z in rr if wanted(z)],'lines','lines','selected_hooks'),
                ('all_algorithms_zoom',[z for z in rr if z['ranks']==args.zoom_gpus],'bars','normal','gpu'+str(args.zoom_gpus)+'_all_algorithms'),
            ]:
                stem=base/sub/(batch+'_'+tag)
                item=dict(system=system,view=view,stem=str(stem),layout=layout)
                if not selected:
                    item.update(status='no_data',reason='No requested methods' if layout=='lines' else 'No completed runs at zoom GPU count')
                    report['figures'].append(item);continue
                aa=copy.copy(args);aa.layout=layout;aa.x='ranks';aa.metric='time';aa.normalize='absolute';aa.share_y=True
                n_panels=len({z['backend'] for z in selected});aa.columns=n_panels;aa.width=max(args.width,9*n_panels)
                simulated=any('simulat' in (z['model']+' '+z['dataset']).lower() for z in selected)
                simulation_gpus=sorted({z['ranks'] for z in selected if z.get('comparison_source_kind')=='simulated'})
                if not args.title:
                    aa.title=('SIMULATION · ' if simulated else '')+'GPT · '+system.title()+' · '+r['mode']+' scaling · '+batch.split('__cfg')[0]+(' · all algorithms at '+str(args.zoom_gpus)+' GPUs' if layout=='bars' else ' · selected algorithms')
                if simulation_gpus:
                    aa.title=(args.title or ('GPT · '+system.title()+' · '+r['mode']+' scaling · '+batch.split('__cfg')[0]))+' · simulated GPU counts: '+', '.join(map(str,simulation_gpus))
                    item['simulated_gpu_counts']=simulation_gpus
                item['source_configurations']=[dict(id=z['id'],source=z['source'],line=z['source_line'],gpus=z['ranks'],method=z['method'],kind=z.get('comparison_source_kind','logged'),original_model=z.get('original_model',z['model']),original_dataset=z.get('original_dataset',z['dataset'])) for z in selected]
                item['backends']=sorted({z['backend'] for z in selected})
                if n_panels<2:item['notice']='Only one backend is available in this selection.'
                if layout=='lines':
                    missing=[]
                    for backend in sorted({z['backend'] for z in rr}):
                        for gpu in sorted({z['ranks'] for z in rr if z['backend']==backend}):
                            have={z['method'] for z in selected if z['backend']==backend and z['ranks']==gpu}
                            missing.extend(dict(backend=backend,gpus=gpu,method=m) for m in expected(system,backend) if m not in have)
                    item['missing_requested_points']=missing
                try:
                    result=create(selected,aa,stem)
                    item.update(status='written',points=len(result['points']),methods=sorted({z['method'] for z in selected}))
                except NoData as exc:item.update(status='no_metric',reason=str(exc))
                report['figures'].append(item)
                print(system,view,item['status'],str(stem),flush=True)
        (root/'focus_manifest.json').write_text(json.dumps(report,indent=2))
        links=[]
        for item in report['figures']:
            if item['status']=='written':
                relative=Path(item['stem']).relative_to(root).as_posix()+'.html'
                links.append('<li><a href="'+html.escape(relative)+'">'+html.escape(item['system']+' · '+item['view'])+'</a></li>')
        (root/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>GPT backend focus</title><h1>GPT backend comparisons</h1><p>Selected-method scaling curves and all-algorithm fixed-GPU bars. Missing requested methods are listed in focus_manifest.json.</p><ul>'+''.join(links)+'</ul></html>')
        return report
    except (ValueError,FileNotFoundError) as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
