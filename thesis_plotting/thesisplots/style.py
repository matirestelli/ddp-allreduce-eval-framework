"""Shared shaded families, pattern keys and contrast checks; no hue-only keys."""
import re
PALETTES={'Baseline':['#575757'],'DDP diagnostics':['#626262','#7B6254','#655E7C','#4C706C'],'Ring':['#145C32','#2E7846','#478B57','#61966F'],'Recursive doubling':['#174F87','#2E6C9E','#4784AD','#5C92B6'],'Hierarchical':['#68408C','#9770AD'],'Direct':['#99530B','#A45F20','#8B5100','#A16C30','#7C4510'],'k-ring':['#75433E'],'Other':['#464646']}
PATTERNS={'Baseline':[''],'DDP diagnostics':['//','xx','++','oo'],'Ring':['','///','\\\\','xx'],'Recursive doubling':['..','oo','++','--'],'Hierarchical':['*','||'],'Direct':['OO','///','xx','++','||'],'k-ring':['--/'],'Other':['xx']}
ORDER=list(PALETTES)

def family(method):
    m=method.split(':')[0]
    if m=='none':return 'Baseline'
    if m.startswith('default'):return 'DDP diagnostics'
    if m.startswith('recursive_doubling'):return 'Recursive doubling'
    if m.startswith('ring'):return 'Ring'
    if m.startswith('hier'):return 'Hierarchical'
    if m.startswith('direct'):return 'Direct'
    if m.startswith('kring'):return 'k-ring'
    return 'Other'

def label(method):
    m,_,rate=method.partition(':')
    if m=='none':return 'Built-in baseline'
    if m.startswith('default'):return m.replace('_',' ')
    if m in ('ring','recursive_doubling'):return 'Uncompressed'
    if m.startswith('hier'):return f'ZFP online · rate{rate}'
    if m=='direct_p':return 't = GPU count'
    if m.startswith('direct_t'):return 't = '+m.split('_t')[-1]
    if m.startswith('kring'):return 'k = '+m.rsplit('k',1)[-1]
    if 'zfp_naive' in m:return f'ZFP naive · rate{rate}'
    if 'zfp_online' in m:return f'ZFP online · rate{rate}'
    return method

def variant(method):
    m,_,rate=method.partition(':')
    if m in ('none','ring','recursive_doubling','direct_p'):return 0
    if 'naive' in m:return 1
    if m.startswith('hier'):return 0 if rate=='16' else 1
    if 'online' in m:return 2 if rate=='16' else 3
    if m.startswith('direct_t'):
        t=int(m.split('_t')[-1]);return max(0,t.bit_length()-3)
    if m.startswith('default'):return ['default','default_sync','default_clone','default_cpu_stage'].index(m) if m in ['default','default_sync','default_clone','default_cpu_stage'] else 0
    return 0

def sort_key(method):return (ORDER.index(family(method)),variant(method),method)
def abbreviation(f):return {'Baseline':'B','DDP diagnostics':'DDP','Ring':'Ring','Recursive doubling':'RD','Hierarchical':'H','Direct':'D','k-ring':'K','Other':'O'}[f]
def full_label(method):return label(method) if family(method)=='Baseline' else family(method)+' · '+label(method)

def luminance(c):
    v=[int(c[i:i+2],16)/255 for i in [1,3,5]];v=[x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in v];return sum(x*w for x,w in zip(v,[.2126,.7152,.0722]))
def contrast(a,b):
    x,y=sorted([luminance(a),luminance(b)]);return (y+.05)/(x+.05)

def style(method,shades=True):
    f=family(method);idx=variant(method);color=PALETTES[f][idx%len(PALETTES[f])] if shades else PALETTES[f][0];hatch=PATTERNS[f][idx%len(PATTERNS[f])]
    if method.endswith(':10'):hatch='//..'  # Rate10 never impersonates rate8.
    markers={'Baseline':['o'],'DDP diagnostics':['1','2','3','4'],'Ring':['s','^','v','>'],'Recursive doubling':['D','p','h','<'],'Hierarchical':['P','X'],'Direct':['*','d','H','+','X'],'k-ring':['8'],'Other':['x']}
    marker=markers[f][idx%len(markers[f])]
    return dict(color=color,hatch=hatch,ink='white',marker=marker,linestyle=['-','--','-.',':'][idx%4])

def contrast_report(methods,shades=True):
    rows=[]
    for m in methods:
        s=style(m,shades);r=contrast(s['color'],'#FFFFFF')
        if r<3:raise ValueError(f'Palette contrast below 3:1 for {m}: {r}')
        rows.append(dict(method=m,color=s['color'],hatch=s['hatch'],white_hatch_vs_fill=r,line_vs_white=r,text_vs_white=21,outline_vs_white=21))
    return rows
