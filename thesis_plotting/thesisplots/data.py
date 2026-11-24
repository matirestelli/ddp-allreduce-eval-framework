"""Parse raw .out/.log/.txt, normalized CSV, or per-epoch training CSV.

No file names are trusted for rank counts. Epoch 1 is discarded. Communication
work is cumulative instrumented work, not necessarily an additive wall-time
component. Missing instrumentation remains missing, never zero.
"""
from __future__ import annotations
import csv, hashlib, json, math, re, statistics
from pathlib import Path

NUMBER=r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
START=re.compile(r'^=== Starting(?: training)?:\s*(.*)$',re.M)
HEADER=re.compile(r'^Distributed Training:\s*(.*?)\s+on\s+(\S+)',re.M)
RANKS=re.compile(r'Ranks:\s*(\d+),\s*Epochs:\s*(\d+),\s*Batch/rank:\s*(\d+),\s*Effective batch:\s*(\d+)')
SKIP_DIRS={'plots','plots_thesis','plots_paper','__pycache__','.git','thesis_plotting','old','archive','oneNode_tuneHyperparameters'}

def num(v, default=None):
    try:
        n=float(v)
        return n if math.isfinite(n) else default
    except (ValueError,TypeError):return default

def med(values):
    values=[v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(values) if values else None

def fields(text):return dict(re.findall(r'(\w+)=([^,\s]+)',text.split(' at ')[0]))
def canonical(hook):
    h=(hook or 'unknown').strip().lower()
    if h in ('none','builtin','built-in (no hook)','built-in','no hook'):return 'none'
    if h.startswith(('hier_zfp','hierarchical_zfp')):return 'hierarchical_zfp_online_coll'
    return h

def context(path):
    s=str(path).lower()
    parts=[x.lower() for x in path.parts[:-1]]
    system=next(('frontier' if x.startswith('experiments_frontier') else 'polaris' for x in parts if x.startswith(('experiments_frontier','experiments_polaris'))),None)
    if system is None:system='frontier' if 'frontier' in s else ('polaris' if 'polaris' in s else 'unknown')
    mode='calibration' if 'tunehyper' in s or 'onenode_tune' in s else ('weak' if 'weakscal' in s else ('strong' if 'strongscal' in s else 'unknown'))
    return system,mode

def make_record(path,text,meta,line=1):
    rank=RANKS.search(text)
    if not rank:return None
    p,expected,batch,rawglobal=map(int,rank.groups())
    header=HEADER.search(text)
    model=(header[1] if header else meta.get('model','unknown')).lower()
    dataset=(header[2] if header else meta.get('dataset','unknown')).lower()
    ba=re.search(r'Backend:\s*(\w+),\s*Algorithm:\s*(.+)',text)
    backend=(ba[1] if ba else meta.get('backend','unknown')).lower()
    rawhook=canonical(ba[2] if ba else meta.get('hook'))
    hook=rawhook
    rate=meta.get('zfp',meta.get('zfp_rate'))
    if rate in ('none','','None'):rate=None
    if rate is None:
        m=re.search(r'(?:_rate|:rate)(\d+)',path.name+' '+text)
        rate=m[1] if m else None
    system,mode=context(path)
    warnings=[]
    quarantined=False
    parent_names=[part.lower() for part in path.parts]
    if any(part in ('experiments_frontier_rccl','experiments_polaris_nccl') for part in parent_names) and backend!='nccl':
        warnings.append('backend disagrees with experiment folder; quarantined from default plots')
        quarantined=True
    if 'HIER_ZFP' in text and 'native fallback' in text and 'online' in rawhook:
        hook='hierarchical_zfp_online_coll'
    epochs={};current=None;labels=set()
    for l in text.splitlines():
        m=re.search(r'\[Epoch\s+(\d+)\]',l)
        if m:current=int(m[1]);epochs.setdefault(current,{'epoch':current})
        if current is None:continue
        row=epochs[current]
        if 't_iter median:' in l:
            m=re.search(r't_iter median:\s*('+NUMBER+')',l)
            if m:row['iter_ms']=num(m[1])
            m=re.search(r'train-only epoch:\s*('+NUMBER+')s',l)
            if m:row['train_s']=num(m[1])
        if '|  Time:' in l or re.search(r'\|\s*Time:',l):row['finished']=True
        if '[HOOK_WORK_SUMMARY]' in l or '[HOOK_TAIL_SUMMARY]' in l:
            kv=fields(l);label=kv.get('label','');labels.add(label)
            parts=label.split(':')
            if len(parts)>=2 and canonical(parts[1])=='hierarchical_zfp_online_coll':hook='hierarchical_zfp_online_coll'
            for key in ['work_total_ms','work_mean_ms','calls','tail_total_ms','tail_mean_ms','steps']:
                if key in kv:row[key]=num(kv[key])
    if len({canonical(l.split(':')[1]) for l in labels if len(l.split(':'))>1})>1:
        warnings.append('multiple internal hook labels in one block; inspect source')
    done=bool(re.search(r'Done in\s*'+NUMBER+r's',text))
    statuses=re.findall(r'^=== Finished(?: training)?:.*?status=(\d+)',text,re.M)
    observed=max([e for e,r in epochs.items() if r.get('finished')],default=0)
    complete=done and observed==expected and (not statuses or statuses[-1]=='0')
    m=re.search(r'Image size:\s*(\d+)',text)
    image=int(m[1]) if m else (224 if re.search('wideresnet.?224',str(path),re.I) else (32 if 'wide_resnet' in model else None))
    m=re.search(r'(?:Batch transform:.*?->|gpu_upscale\s+\d+\s*->)\s*(\d+)',text)
    if m:image=int(m[1])
    m=re.search(r'\[Accum\].*?micro-steps/rank=(\d+).*?global micro-batches=(\d+).*?tokens/iter=([\d,]+)',text)
    micro=int(m[1]) if m else 1
    accum=int(m[2]) if m else int(meta.get('accum','0') or 0)
    tokens=int(m[3].replace(',','')) if m else None
    effective=batch*accum if accum else rawglobal
    block=int(meta['block']) if meta.get('block') else (tokens//effective if tokens and effective else None)
    lr=num(meta.get('lr'))
    if lr is None:
        m=re.search(r'LR:\s*('+NUMBER+')',text);lr=num(m[1]) if m else None
    summary=re.search(r'Train-only epoch time.*?median=('+NUMBER+')s',text)
    fp=re.search(r'_(\d+)gpus?',path.name,re.I)
    if fp and int(fp[1])!=p:warnings.append(f'filename ranks {fp[1]} differ from actual ranks {p}')
    if meta.get('ws') and int(meta['ws'])!=p:warnings.append('launch rank count differs from training header')
    if not complete:warnings.append(f'incomplete: {observed}/{expected} epochs; done={done}; status={statuses[-1] if statuses else "not logged"}')
    family='gpt' if 'gpt' in model else ('wideresnet224' if image==224 else ('wideresnet32' if 'wide_resnet' in model else model))
    r=dict(system=system,mode=mode,model=model,dataset=dataset,image_size=image,family=family,backend=backend,ranks=p,batch_per_rank=batch,global_batch=rawglobal,effective_batch=effective,accum_global=accum or None,micro_steps=micro,block_size=block,tokens_per_iter=tokens,lr=lr,epochs=expected,observed_epochs=observed,complete=complete,raw_hook=rawhook,algorithm=hook,zfp_rate=str(rate) if rate is not None else None,epochs_data=[epochs[k] for k in sorted(epochs)],train_summary_s=num(summary[1]) if summary else None,source=str(path.resolve()),source_line=line,warnings=warnings,internal_labels=sorted(labels),native_fallback='native fallback' in text)
    r['method']=hook+(f':{r["zfp_rate"]}' if r['zfp_rate'] else '')
    r['quarantined']=quarantined
    r['id']=hashlib.sha256(f'{r["source"]}:{line}'.encode()).hexdigest()[:12]
    return r

def parse_text(path):
    text=path.read_text(errors='replace');starts=list(START.finditer(text));runs=[]
    segments=[(m.start(),starts[i+1].start() if i+1<len(starts) else len(text),fields(m[1])) for i,m in enumerate(starts)] if starts else [(0,len(text),{})]
    for begin,end,meta in segments:
        segment=text[begin:end];heads=list(HEADER.finditer(segment))
        # Appended standalone training blocks must not inherit the prior hook.
        pieces=[(0,heads[1].start() if len(heads)>1 else len(segment),meta)]
        if len(heads)>1:pieces += [(h.start(),heads[i+1].start() if i+1<len(heads) else len(segment),{}) for i,h in enumerate(heads) if i>0]
        for start,stop,mm in pieces:
            body=segment[start:stop];r=make_record(path,body,mm,text[:begin+start].count('\n')+1)
            if r:runs.append(r)
    return runs

# Per-epoch CSV aliases written by the training framework.
CSV_ALIASES={'world_size':'ranks','batch_size':'batch_per_rank','global_batch_size':'global_batch','epoch_train_time_s':'train_s','t_iter_median_ms':'iter_ms','hook_work_total_ms':'work_total_ms','hook_work_mean_ms':'work_mean_ms','hook_calls':'calls','tail_steps':'steps'}
def parse_csv(path):
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:return []
    if 'epochs_data' in rows[0]:
        out=[]
        for row in rows:
            r=json.loads(row['record_json']) if row.get('record_json') else row
            if isinstance(r.get('epochs_data'),str):r['epochs_data']=json.loads(r['epochs_data'])
            out.append(r)
        return out
    if 'epoch' not in rows[0]:raise ValueError(f'{path}: CSV must be per-epoch training data or exported parsed_runs.csv; arbitrary summary columns are not inferred')
    grouped={}
    for row in rows:
        d={CSV_ALIASES.get(k,k):v for k,v in row.items()}
        key=tuple(d.get(k,'') for k in ['model','dataset','backend','algorithm','zfp_rate','ranks','batch_per_rank','global_batch','run_id'])
        grouped.setdefault(key,[]).append(d)
    out=[]
    for key,group in grouped.items():
        d=group[0];p=int(num(d.get('ranks'),0));bs=int(num(d.get('batch_per_rank'),0));epmax=max(int(num(x['epoch'],0)) for x in group);expected=int(num(d.get('num_epochs'),epmax))
        if not p or not bs:raise ValueError(f'{path}: CSV requires world_size/ranks and batch_size/batch_per_rank')
        system,mode=context(path);model=d.get('model','unknown').lower();image=int(num(d.get('image_size'),224 if '224' in str(path) else 32));hook=canonical(d.get('algorithm',d.get('hook')));rate=d.get('zfp_rate');rate=None if rate in ('','none',None) else str(rate)
        ed=[]
        for x in group:
            er={k:num(x.get(k)) for k in ['train_s','iter_ms','work_total_ms','work_mean_ms','calls','tail_total_ms','tail_mean_ms','steps']};er.update(epoch=int(num(x['epoch'],0)),finished=True);ed.append(er)
        gb=int(num(d.get('global_batch'),p*bs));accum=int(num(d.get('accum_global'),0))
        r=dict(system=d.get('system') or system,mode=d.get('mode') or mode,model=model,dataset=d.get('dataset','unknown'),image_size=image,family='gpt' if 'gpt' in model else ('wideresnet224' if image==224 else 'wideresnet32'),backend=d.get('backend','unknown').lower(),ranks=p,batch_per_rank=bs,global_batch=gb,effective_batch=int(num(d.get('effective_batch'),bs*accum if accum else gb)),accum_global=accum or None,micro_steps=int(num(d.get('micro_steps'),1)),block_size=num(d.get('block_size')),tokens_per_iter=num(d.get('tokens_per_iter')),lr=num(d.get('lr',d.get('learning_rate'))),epochs=expected,observed_epochs=epmax,complete=epmax==expected,raw_hook=hook,algorithm=hook,zfp_rate=rate,epochs_data=ed,train_summary_s=None,source=str(path.resolve()),source_line=2,warnings=['CSV completion inferred from epoch coverage; provide num_epochs to detect truncation'],internal_labels=[],native_fallback=False)
        r['method']=hook+(f':{rate}' if rate else '');r['id']=hashlib.sha256((r['source']+repr(key)+repr(ed)).encode()).hexdigest()[:12];out.append(r)
    return out

def signature(r,include_method=True):
    keys=['system','mode','model','dataset','image_size','backend','ranks','batch_per_rank','global_batch','effective_batch','accum_global','block_size','lr','epochs']
    if include_method:keys+=['method']
    return tuple(r.get(k) for k in keys)

def load_inputs(inputs,system=None,mode=None,include_partial=False):
    paths=set();issues=[];runs=[]
    for arg in inputs:
        p=Path(arg).expanduser().resolve()
        if not p.exists():raise FileNotFoundError(p)
        if p.is_file():paths.add(p)
        else:
            for f in p.rglob('*'):
                if f.is_file() and f.suffix.lower() in ('.out','.log','.txt','.csv') and not any(part in SKIP_DIRS for part in f.relative_to(p).parts):paths.add(f)
    for p in sorted(paths):
        try:rr=parse_csv(p) if p.suffix.lower()=='.csv' else parse_text(p)
        except (ValueError,KeyError,TypeError) as exc:issues.append({'source':str(p),'warning':str(exc)});continue
        for r in rr:
            if system and r['system']=='unknown':r['system']=system
            if mode and r['mode']=='unknown':r['mode']=mode
            if r['mode']=='unknown':r['warnings'].append('scaling mode unknown; supply --mode')
            for warning in r['warnings']:issues.append({'source':r['source'],'line':r['source_line'],'warning':warning})
            runs.append(r)
    # Exact metric traces identify mirrored .out/.log copies. Keep all genuinely
    # different timings as repeated measurements; no fastest-result selection.
    unique={}
    for r in runs:
        trace=json.dumps(r['epochs_data'],sort_keys=True)
        key=signature(r)+(r['complete'],trace,r.get('train_summary_s'))
        old=unique.get(key)
        if old:
            old.setdefault('duplicate_sources',[]).append({'source':r['source'],'line':r['source_line']})
            if Path(r['source']).suffix=='.log':
                r['duplicate_sources']=old['duplicate_sources']+[{'source':old['source'],'line':old['source_line']}];unique[key]=r
        else:unique[key]=r
    return [r for r in unique.values() if (include_partial or r['complete']) and not r.get('quarantined')],issues,list(unique.values())

def epoch_values(r,metric,unit='auto'):
    rows=[e for e in r['epochs_data'] if e['epoch']>1]
    if metric=='time':return [e.get('train_s' if r['mode']=='strong' else 'iter_ms') for e in rows]
    unit=('epoch' if r['mode']=='strong' else 'iter') if unit=='auto' else unit
    values=[]
    for e in rows:
        if metric=='hook':
            if unit=='call':v=e.get('work_mean_ms') if e.get('work_mean_ms') is not None else (e['work_total_ms']/e['calls'] if e.get('work_total_ms') is not None and e.get('calls') else None)
            elif unit=='iter':v=e.get('work_total_ms')/e['steps'] if e.get('work_total_ms') is not None and e.get('steps') else None
            else:v=e.get('work_total_ms')/1000 if e.get('work_total_ms') is not None else None
        elif metric=='tail':
            if unit=='call':raise ValueError('Tail is per optimizer step, not per hook call; choose iter or epoch')
            v=(e.get('tail_mean_ms') if e.get('tail_mean_ms') is not None else (e['tail_total_ms']/e['steps'] if e.get('tail_total_ms') is not None and e.get('steps') else None)) if unit=='iter' else (e.get('tail_total_ms')/1000 if e.get('tail_total_ms') is not None else None)
        elif metric=='tail_pct':v=100*e['tail_total_ms']/(e['train_s']*1000) if e.get('tail_total_ms') is not None and e.get('train_s') else None
        else:raise ValueError(f'Unknown metric: {metric}')
        values.append(v)
    return [v for v in values if v is not None and math.isfinite(v)]

def value(r,metric,unit='auto'):
    if metric=='time' and r['mode']=='strong' and r.get('train_summary_s') is not None:return r['train_summary_s']
    return med(epoch_values(r,metric,unit))

def export_runs(runs,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not runs:return
    with path.open('w',newline='') as f:
        cols=['id','system','family','mode','backend','ranks','batch_per_rank','effective_batch','method','complete','source','source_line','epochs_data','record_json']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in runs:
            row={k:r.get(k) for k in cols};row['epochs_data']=json.dumps(r['epochs_data']);row['record_json']=json.dumps(r);w.writerow(row)
