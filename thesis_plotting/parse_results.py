#!/usr/bin/env python3
"""Audit inputs without plotting; include all partial runs in the exported CSV.

  python parse_results.py /path/to/code_oct26 --out plots_thesis/audit
  python parse_results.py a.out b.log --mode strong --system frontier --out audit
  python parse_results.py normalized.csv --out roundtrip

Outputs parsed_runs.csv (usable by every plot script), issues.json and summary.
"""
import argparse,json,collections
from pathlib import Path
from thesisplots.data import load_inputs,export_runs
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inputs',nargs='+');p.add_argument('--out',type=Path,default=Path('plots_thesis/audit'));p.add_argument('--system',choices=['frontier','polaris']);p.add_argument('--mode',choices=['strong','weak']);a=p.parse_args()
    runs,issues,allruns=load_inputs(a.inputs,system=a.system,mode=a.mode);a.out.mkdir(parents=True,exist_ok=True);export_runs(allruns,a.out/'parsed_runs.csv');(a.out/'issues.json').write_text(json.dumps(issues,indent=2));summary=collections.Counter((r['system'],r['family'],r['mode'],r['backend'],r['complete']) for r in allruns);(a.out/'summary.txt').write_text('\n'.join(f'{k}: {v}' for k,v in sorted(summary.items())));print((a.out/'summary.txt').read_text());print(f'{len(issues)} audit notices; {len(runs)} completed runs')
