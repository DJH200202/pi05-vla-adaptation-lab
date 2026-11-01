"""Build presentation data from recorded experiments. Never imports the policy or runs it."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/train500'
GROUPS = ['Base', 'A', 'B', 'C', 'D']
LABELS = {'Base': 'Original base', 'A': 'Action LoRA', 'B': 'Dual LoRA',
          'C': 'Dual LoRA + vision/projections', 'D': 'LLM LoRA + full expert/vision/projections'}

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def ema(values, alpha=.05):
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1-alpha)*out[-1])
    return out

def build():
    data = {'groups': GROUPS, 'labels': LABELS, 'training': {}, 'evaluation': {},
            'paired': {}, 'cases': [], 'schema_version': 1, 'ema_alpha': .05}
    all_rows = {}
    manifests = []
    for g in GROUPS:
        directory = RUN/'eval'/g
        summary = json.loads((directory/'summary.json').read_text())
        records = [r for p in sorted(directory.glob('episodes-*.jsonl')) for r in read_jsonl(p)]
        records.sort(key=lambda r: (r['task_id'],r['initial_state_id']))
        keys = [(r['task_id'],r['initial_state_id']) for r in records]
        assert len(keys) == len(set(keys)) == 500
        assert Counter(r['task_id'] for r in records) == Counter({i:50 for i in range(10)})
        assert all(r['outcome'] in ['success','timeout'] for r in records)
        assert sum(r['success'] for r in records) == summary['successes']
        assert summary['errors'] == 0 and summary['state'] == 'completed'
        assert (directory/'exit-code.txt').read_text().strip() == '0'
        manifest = (directory/'eval_manifest.json').read_bytes()
        digest = hashlib.sha256(manifest).hexdigest()
        assert digest == summary['eval_manifest_sha256']
        manifests.append(digest)
        all_rows[g] = records
        data['evaluation'][g] = summary
    assert len(set(manifests)) == 1
    data['manifest_sha256'] = manifests[0]
    data['tasks'] = [{'id': i, 'prompt': next(r['prompt'] for r in all_rows['C'] if r['task_id']==i)} for i in range(10)]
    reference_samples = None
    for g in 'ABCD':
        directory = RUN/'train'/g
        events = read_jsonl(directory/'metrics.jsonl')
        rows = [e for e in events if e['event']=='train']
        assert [r['step'] for r in rows] == list(range(1,3001))
        assert (directory/'exit-code.txt').read_text().strip() == '0'
        assert not any(e['event'].startswith('val') or 'validation_full' in e['event'] for e in events)
        samples = [r['sample_indices'] for r in rows]
        if reference_samples is None: reference_samples = samples
        assert samples == reference_samples, 'Different sampled batches across groups'
        cfg = json.loads((directory/'config.json').read_text())
        start = next(e for e in events if e['event']=='start')
        assert cfg['validation_mode']=='none' and start['train_frames']==61750
        assert start['train_episodes']==500 and start['global_batch']==32
        assert start['validation_frames']==0
        loss = [r['train_loss'] for r in rows]
        csv_rows = list(csv.DictReader((directory/'training.csv').open()))
        assert len(csv_rows)==3000
        for raw, exported in zip(rows,csv_rows):
            assert raw['step']==int(exported['step']) and raw['train_loss']==float(exported['train_loss'])
        compilation_steps = [int(r['step']) for r in csv_rows if r['includes_compile']=='True']
        steady = [r for r in rows if r['step'] not in compilation_steps]
        data['training'][g] = {'config':cfg,'parameters':start['parameter_counts'],
            'adaptation':start['adaptation'],'final_loss':loss[-1],
            'last100_mean':statistics.mean(loss[-100:]), 'last300_mean':statistics.mean(loss[-300:]),
            'compile_steps':compilation_steps,
            'median_update_seconds':statistics.median(r['update_seconds'] for r in steady),
            'checkpoints':[e['step'] for e in events if e['event']=='checkpoint'],
            'step':[r['step'] for r in rows], 'loss':loss, 'loss_ema':ema(loss),
            'lr':[r['lr'] for r in rows], 'grad_norm':[r['grad_norm'] for r in rows],
            'update_seconds':[r['update_seconds'] for r in rows]}
    data['same_training_sample_indices'] = True
    data['episodes'] = [{ 'task_id':r['task_id'], 'initial_state_id':r['initial_state_id'],
                          **{g:all_rows[g][i]['success'] for g in GROUPS}} for i,r in enumerate(all_rows['C'])]
    for a,b in [('B','C'),('C','D')]:
        c = Counter((bool(x['success']),bool(y['success'])) for x,y in zip(all_rows[a],all_rows[b]))
        data['paired'][a+'_'+b] = {'both_success':c[(True,True)],'left_only':c[(True,False)],
                                  'right_only':c[(False,True)],'both_failure':c[(False,False)]}
    selections = [
        ('adaptation','B 失败，C / D 成功',lambda x:not x['B'] and x['C'] and x['D']),
        ('c_only','C 成功，D 失败',lambda x:x['C'] and not x['D']),
        ('d_only','任务 5：D 成功，C 失败',lambda x:x['task_id']==5 and not x['C'] and x['D']),
        ('both_fail','C / D 均失败',lambda x:not x['C'] and not x['D'])]
    for case_id,title,condition in selections:
        row = next(x for x in data['episodes'] if condition(x))
        key = (row['task_id'],row['initial_state_id'])
        groups = GROUPS if case_id=='adaptation' else ['B','C','D']
        records = {g:next(r for r in all_rows[g] if (r['task_id'],r['initial_state_id'])==key) for g in groups}
        assert len({r['initial_state_sha256'] for r in records.values()})==1
        data['cases'].append({'id':case_id,'title':title,'task_id':key[0],'initial_state_id':key[1],
                             'prompt':records['C']['prompt'],'records':records,
                             'selection':'First matching episode in (task_id, initial_state_id) order; selected examples, not a random sample.'})
    data['historical'] = list(csv.DictReader((ROOT/'results/train450/summary.csv').open()))
    out = ROOT/'demo/assets'
    out.mkdir(parents=True,exist_ok=True)
    (out/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
    (out/'data.js').write_text('window.DEMO_DATA='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    compact = {g:{k:v for k,v in d.items() if k in ['final_loss','last100_mean','last300_mean','parameters','median_update_seconds','compile_steps']} for g,d in data['training'].items()}
    (ROOT/'results/derived.json').write_text(json.dumps({'training':compact,'paired':data['paired'],'same_training_sample_indices':True},ensure_ascii=False,indent=2))
    print(json.dumps({'training':compact,'paired':data['paired'],'selected':[(x['id'],x['task_id'],x['initial_state_id']) for x in data['cases']]},indent=2))
    return data

if __name__=='__main__': build()
