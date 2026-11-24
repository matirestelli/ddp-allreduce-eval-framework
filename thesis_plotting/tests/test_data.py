"""Run: python -m unittest discover -s tests -v (from thesis_plotting)."""
import csv,tempfile,unittest
from pathlib import Path
from thesisplots.data import make_record,parse_text,load_inputs,value,export_runs
from thesisplots.cli import parser,parse
from thesisplots.render import select,transform_direct,workload
from thesisplots.style import contrast_report

TEXT='''Distributed Training: WIDE_RESNET50_2 on CIFAR10
Backend: NCCL, Algorithm: ring_zfp_online_coll
Ranks: 16, Epochs: 3, Batch/rank: 8, Effective batch: 128
[Epoch 1] Train Loss: 1 | Time: 10s
[Epoch 1] t_iter median: 999 ms | train-only epoch: 99s
[HOOK_WORK_SUMMARY] label=nccl:hier_zfp_online_coll:rate8 calls=20 work_total_ms=100 work_mean_ms=5
[HOOK_TAIL_SUMMARY] label=nccl:hier_zfp_online_coll:rate8 steps=10 tail_total_ms=20 tail_mean_ms=2
[Epoch 2] Train Loss: 1 | Time: 10s
[Epoch 2] t_iter median: 10 ms | train-only epoch: 1s
[HOOK_WORK_SUMMARY] label=nccl:hier_zfp_online_coll:rate8 calls=20 work_total_ms=100 work_mean_ms=5
[HOOK_TAIL_SUMMARY] label=nccl:hier_zfp_online_coll:rate8 steps=10 tail_total_ms=20 tail_mean_ms=2
[Epoch 3] Train Loss: 1 | Time: 10s
[Epoch 3] t_iter median: 12 ms | train-only epoch: 1.2s
[HOOK_WORK_SUMMARY] label=nccl:hier_zfp_online_coll:rate8 calls=40 work_total_ms=200 work_mean_ms=5
[HOOK_TAIL_SUMMARY] label=nccl:hier_zfp_online_coll:rate8 steps=10 tail_total_ms=40 tail_mean_ms=4
Done in 30s | Best val acc: 90%
Train-only epoch time (epochs 2..N): median=1.11s
=== Finished: backend=nccl, hook=ring_zfp_online_coll, status=0
'''
class DataTests(unittest.TestCase):
    def rec(self,text=TEXT,path=None):return make_record(path or Path('experiments_frontier_rccl/strongScaling/r_16gpus.out'),text,{'zfp':'8'})
    def test_actual_hierarchical(self):
        r=self.rec();self.assertEqual(r['algorithm'],'hierarchical_zfp_online_coll');self.assertEqual(r['raw_hook'],'ring_zfp_online_coll')
    def test_units_and_warmup(self):
        r=self.rec();self.assertEqual(value(r,'hook','call'),5);self.assertEqual(value(r,'hook','iter'),15);self.assertAlmostEqual(value(r,'hook','epoch'),.15);self.assertEqual(value(r,'tail','iter'),3);self.assertAlmostEqual(value(r,'tail','epoch'),.03);self.assertEqual(value(r,'time'),1.11)
    def test_weak_iteration_medians(self):
        r=self.rec(path=Path('experiments_frontier_rccl/weakScaling/r.out'));self.assertEqual(value(r,'time'),11)
    def test_missing_instrumentation_is_none(self):
        r=self.rec('\n'.join(l for l in TEXT.splitlines() if 'HOOK_' not in l));self.assertIsNone(value(r,'hook'));self.assertIsNone(value(r,'tail'))
    def test_partial_not_complete(self):self.assertFalse(self.rec(TEXT[:TEXT.index('[Epoch 3]')])['complete'])
    def test_nonzero_exit(self):self.assertFalse(self.rec(TEXT.replace('status=0','status=1'))['complete'])
    def test_filename_is_not_rank_source(self):
        r=self.rec(path=Path('experiments_frontier_rccl/strongScaling/r_64gpus.out'));self.assertEqual(r['ranks'],16);self.assertTrue(any('filename ranks' in w for w in r['warnings']))
    def test_standalone_blocks_do_not_inherit_hook(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.out';p.write_text('=== Starting: model=x, hook=ring_zfp_online_coll, zfp=8\n'+TEXT+'\n'+TEXT.replace('Algorithm: ring_zfp_online_coll','Algorithm: none'))
            rs=parse_text(p);self.assertEqual(len(rs),2);self.assertEqual(rs[1]['raw_hook'],'none')
    def test_mirrored_log_dedup_and_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'a.out').write_text(TEXT);(p/'b.log').write_text(TEXT);rs,_,_=load_inputs([p],mode='strong',system='frontier');self.assertEqual(len(rs),1);export_runs(rs,p/'normalized.csv');rr,_,_=load_inputs([p/'normalized.csv']);self.assertEqual(value(rr[0],'tail','iter'),3)
    def test_accumulation_workload(self):
        r=self.rec(TEXT.replace('Batch/rank: 8','Batch/rank: 12').replace('Effective batch: 128','Effective batch: 192')+'\n[Accum] micro-steps/rank=4, global micro-batches=64, tokens/iter=786,432\n');self.assertEqual(r['effective_batch'],768);self.assertEqual(r['block_size'],1024)
    def test_native_fallback_alias(self):
        text='\n'.join(l for l in TEXT.splitlines() if 'HOOK_' not in l)+'\n[HIER_ZFP] native fallback: single node\n';r=self.rec(text);self.assertEqual(r['algorithm'],'hierarchical_zfp_online_coll');self.assertIsNone(value(r,'hook'))
    def test_direct_gpu_rule(self):
        rs=[dict(method='direct_t8',ranks=8),dict(method='direct_t8',ranks=16),dict(method='direct_t16',ranks=16)];self.assertEqual([r['method'] for r in transform_direct(rs,'gpu')],['direct_p','direct_p'])
    def test_color_contrast(self):
        report=contrast_report(['none','ring','ring_zfp_naive:16','ring_zfp_online_coll:8','recursive_doubling_zfp_online_coll:8','hierarchical_zfp_online_coll:8']);self.assertTrue(all(r['white_hatch_vs_fill']>=3 for r in report))
    def test_quarantine_wrong_backend_folder(self):
        r=self.rec(TEXT.replace('Backend: NCCL','Backend: MPI'));self.assertTrue(r['quarantined'])
    def test_weak_x_does_not_use_epoch_time(self):
        r=self.rec(path=Path('experiments_frontier_rccl/weakScaling/r.out'));self.assertNotEqual(value(r,'time'),r['train_summary_s'])
    def test_rate_preference_never_relabels(self):
        a=self.rec();a['algorithm']='ring_zfp_online_coll';a['method']='ring_zfp_online_coll:10';a['zfp_rate']='10'
        b=dict(a,method='ring_zfp_online_coll:8',zfp_rate='8',epochs=10)
        args=parse(parser(),['dummy','--prefer-rate8'])
        rr=select([a,b],args);self.assertEqual([r['method'] for r in rr],['ring_zfp_online_coll:8'])
        self.assertEqual(select([a],args)[0]['method'],'ring_zfp_online_coll:10')
    def test_partial_default_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.out';p.write_text(TEXT[:TEXT.index('[Epoch 3]')]);rs,_,allr=load_inputs([p],mode='strong');self.assertEqual(rs,[]);self.assertEqual(len(allr),1)
    def test_direct_fixed_curves_have_distinct_markers(self):
        from thesisplots.style import style
        self.assertEqual(len({style('direct_t'+str(t))['marker'] for t in [4,8,16,32,64]}),5)
if __name__=='__main__':unittest.main()

class TailContextTests(unittest.TestCase):
    def test_matched_epoch_context_excludes_warmup_and_unmatched_epochs(self):
        from thesisplots.render import tail_pair_values
        r=dict(mode='strong',epochs_data=[dict(epoch=1,train_s=999,tail_total_ms=999000),dict(epoch=2,train_s=2,tail_total_ms=500,steps=10),dict(epoch=3,train_s=4,tail_total_ms=1500,steps=20),dict(epoch=4,train_s=100)])
        self.assertEqual(tail_pair_values(r,'epoch'),(3,1))
        self.assertEqual(tail_pair_values(r,'iter'),(200,62.5))
        with self.assertRaises(ValueError):tail_pair_values(r,'call')
    def test_pair_uses_same_repeats_and_does_not_fill_missing_with_zero(self):
        from thesisplots.render import aggregate_tail_pairs
        base=dict(id='a',system='frontier',backend='mpi',ranks=16,method='ring',raw_hook='ring',mode='strong',source='missing.out',source_line=1,epochs=2,observed_epochs=2,complete=True,native_fallback=False)
        a=dict(base,epochs_data=[dict(epoch=2,train_s=4,tail_total_ms=1000)])
        b=dict(base,id='b',epochs_data=[dict(epoch=2,train_s=8,tail_total_ms=3000)])
        missing=dict(base,id='c',epochs_data=[dict(epoch=2,train_s=90)])
        points,omitted=aggregate_tail_pairs([a,b,missing],'epoch','ranks','median')
        self.assertEqual(points[0]['value'],2)
        self.assertEqual(points[0]['training_value'],6)
        self.assertEqual(points[0]['n_runs'],2)
        self.assertEqual(len(omitted),1)
