"""Common command options; every public script exposes the full --help."""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
from .data import load_inputs,export_runs
from .render import create,select,NoData

def parser(default_metric='time',default_layout='bars',description=None):
    p=argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('inputs',nargs='+',help='One or more raw experiment folders, individual .out/.log files, training CSVs, or exported parsed_runs.csv; scanning is recursive')
    p.add_argument('--mode',choices=['strong','weak'],help='Filter scaling mode; supply it when folder names do not identify the mode')
    p.add_argument('--system',nargs='+',choices=['frontier','polaris'],help='Filter systems')
    p.add_argument('--assume-system',choices=['frontier','polaris'],help='Identify files with no system in their path')
    p.add_argument('--backend',nargs='+',choices=['mpi','nccl','rccl','gloo'],help='Filter backend; Frontier NCCL API is displayed as RCCL')
    p.add_argument('--model',nargs='+',help='Model/family substring filters: wideresnet32, wideresnet224, gpt, or actual model name')
    p.add_argument('--gpus',nargs='+',type=int,help='Filter actual rank counts')
    p.add_argument('--global-batch',nargs='+',type=int,help='Filter effective/accumulated global batch (GPT includes accumulation)')
    p.add_argument('--batch-per-rank',nargs='+',type=int,help='Filter local microbatch')
    p.add_argument('--lr',type=float,help='Filter learning rate')
    p.add_argument('--methods',nargs='+',help='Exact canonical IDs; e.g. none ring recursive_doubling_zfp_online_coll:8 hierarchical_zfp_online_coll:8 direct_p')
    p.add_argument('--families',nargs='+',type=str.lower,help='Family labels: Ring, "Recursive doubling", Hierarchical, Direct, "k-ring", Baseline')
    p.add_argument('--rates',nargs='+',help='ZFP rate filter; uncompressed methods remain')
    p.add_argument('--direct',choices=['gpu','all','omit'],default='gpu',help='gpu: use direct_tP at P ranks as Direct t=GPUs; all: separate fixed-t series; omit: hide Direct')
    p.add_argument('--diagnostics',action=argparse.BooleanOptionalAction,default=True,help='Include default/default_sync/default_clone diagnostic hooks')
    p.add_argument('--prefer-rate8',action=argparse.BooleanOptionalAction,default=False,help='Suppress rate10 only when rate8 exists at the identical configuration; never relabel measurements')
    p.add_argument('--include-partial',action='store_true',help='Include incomplete runs (explicit exploratory override; provenance records incompleteness)')
    p.add_argument('--metric',choices=['time','hook','tail','tail_pct'],default=default_metric)
    p.add_argument('--unit',choices=['auto','epoch','iter','call'],default='auto',help='Communication units; auto=seconds/epoch in strong and milliseconds/optimizer step in weak. Hook call is different from optimizer step')
    p.add_argument('--layout',choices=['bars','lines','hbar'],default=default_layout)
    p.add_argument('--tail-context',action=argparse.BooleanOptionalAction,default=True,help='Horizontal tail plots pair gray train-only time with the colored exposed tail; --no-tail-context gives tail alone')
    p.add_argument('--x',choices=['ranks','global_batch','batch_per_rank'],default='ranks',help='Vary ranks at fixed workload, or vary batch at one GPU count')
    p.add_argument('--normalize',choices=['absolute','baseline','speedup','efficiency'],default='absolute')
    p.add_argument('--reference-gpus',type=int,help='Reference for strong speedup/efficiency; default is smallest plotted count')
    p.add_argument('--repeats',choices=['median','latest'],default='median',help='Reduce genuinely different runs; mirrored files are deduplicated first')
    p.add_argument('--palette',choices=['shades','solid'],default='shades')
    p.add_argument('--family-labels',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--share-y',action=argparse.BooleanOptionalAction,default=True,help='Use common y limits across comparison panels')
    p.add_argument('--columns',type=int,default=1,help='Panel columns; use sufficiently large physical --width for multiple columns')
    p.add_argument('--width',type=float,default=9,help='Physical inches; default is for a landscape thesis page, not portrait scaling')
    p.add_argument('--height',type=float,help='Physical inches; automatic sizing reserves room for all legend entries')
    p.add_argument('--font-size',type=float,default=11,help='Smallest point size at the specified physical dimensions')
    p.add_argument('--ymax',type=float,help='Upper value limit (horizontal-axis upper limit for hbar)')
    p.add_argument('--xscale',choices=['linear','log2'],default='linear')
    p.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf'])
    p.add_argument('--dpi',type=int,default=220)
    p.add_argument('--title',help='Override title')
    p.add_argument('--out',type=Path,help='Output stem, e.g. plots_thesis/custom/wrn128; sidecars share this stem')
    p.add_argument('--config',type=Path,help='JSON option defaults; command-line flags take precedence; underscores replace dashes in keys')
    return p

def parse(p,argv=None):
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if '--config' in argv:
        ix=argv.index('--config');config=json.loads(Path(argv[ix+1]).read_text());valid={a.dest for a in p._actions};unknown=set(config)-valid
        if unknown:p.error('Unknown config keys: '+', '.join(sorted(unknown)))
        p.set_defaults(**config)
    args=p.parse_args(argv)
    if args.backend:args.backend=['nccl' if x=='rccl' else x for x in args.backend]
    if args.font_size<11:print('NOTE: font size below the thesis default; inspect final-size readability.',file=sys.stderr)
    if args.columns<1 or args.width<=0:p.error('columns and width must be positive')
    return args

def slug(v):return re.sub(r'[^A-Za-z0-9._-]+','_',str(v)).strip('_')
def default_stem(runs,args):
    r=runs[0];systems='_vs_'.join(sorted({x['system'] for x in runs}));backends='_vs_'.join(sorted({x['backend'] for x in runs}));batch='GB'+str(r['effective_batch']) if r['mode']=='strong' else 'LB'+str(r['batch_per_rank']);batch=batch if args.x=='ranks' else 'vary_'+args.x
    gpus='gpu'+'-'.join(map(str,sorted({x['ranks'] for x in runs})))
    sub='normal' if args.metric=='time' and args.layout!='lines' else 'lines' if args.metric=='time' else 'hooks' if args.metric=='hook' else 'tails'
    return Path('plots_thesis')/systems/r['family']/r['mode']/backends/sub/(batch+'_'+gpus+'_'+args.layout+'_'+args.normalize)

def execute(args):
    runs,issues,allruns=load_inputs(args.inputs,system=args.assume_system,mode=args.mode,include_partial=args.include_partial)
    if args.lr is not None:runs=[r for r in runs if r['lr']==args.lr]
    runs=select(runs,args)
    if not runs:raise NoData('No matching runs; use parse_results.py to inspect available configurations')
    out=Path(args.out) if args.out else default_stem(runs,args)
    if out.suffix in ('.png','.pdf','.svg'):out=out.with_suffix('')
    result=create(runs,args,out);export_runs(allruns,out.parent/'parsed_runs.csv');out.with_suffix('.issues.json').write_text(json.dumps(issues,indent=2))
    print(f'Wrote {out} ({len(result["points"])} plotted values; {len(result["omitted"])} metrics not recorded)')
    return result

def main(metric='time',layout='bars',description=None):
    p=parser(metric,layout,description);args=parse(p)
    try:execute(args)
    except (ValueError,FileNotFoundError) as exc:p.exit(2,str(exc)+'\n')
