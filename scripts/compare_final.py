
import zipfile
import os
import json
import pandas as pd

results_dir = r'd:\GitHub\my_repo\Brain_Fingerprinting\results'

files_to_extract = {
    'Baseline (CVAE+SDL)': 'results (11).zip',
    'Original (ConvAE+SDL)': 'target_results.zip',
    'MetricBolT (SOTA)': 'results (10).zip'
}

comparison_data = []

for cat, z_name in files_to_extract.items():
    z_path = os.path.join(results_dir, z_name)
    if not os.path.exists(z_path): continue
    
    with zipfile.ZipFile(z_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('results.json'):
                try:
                    data = json.loads(zf.read(name))
                    task = data.get('task', 'unknown')
                    if task == 'unknown':
                        for t in ['motor', 'wm', 'emotion', 'gambling', 'language', 'relational', 'social']:
                            if t in name.lower(): task = t; break
                    
                    m = None
                    if 'cvae_sdl' in name:
                        m = data.get('cvae_sdl_metrics')
                    elif 'ablation_results' in data or 'ablation_accuracies' in data:
                        abl = data.get('ablation_results') or data.get('ablation_accuracies')
                        val = abl.get('convae_sdl')
                        if isinstance(val, (int, float)):
                            m = {'top_1_accuracy': val}
                        elif isinstance(val, dict):
                            m = val.get('metrics') or val
                    
                    if not m:
                        m = data.get('metricbolt_metrics') or data.get('metrics')
                    
                    if not m and 'top_1_accuracy' in data: m = data
                    
                    if m:
                        comparison_data.append({
                            'Method': cat,
                            'Task': task,
                            'Top-1 Accuracy': m.get('top_1_accuracy') or m.get('accuracy'),
                            'Top-5 Accuracy': m.get('top_5_accuracy'),
                            'MRR': m.get('mrr'),
                            'Diff-ID': m.get('differential_id')
                        })
                except:
                    pass

df = pd.DataFrame(comparison_data)
tasks = ['motor', 'wm', 'emotion', 'gambling', 'language', 'relational', 'social']
df = df[df['Task'].isin(tasks)]

print("\n" + "="*80)
print("BRAIN FINGERPRINTING: CROSS-PIPELINE COMPARISON")
print("="*80)

if not df.empty:
    pivoted = df.pivot_table(index='Task', columns='Method', values='Top-1 Accuracy')
    print(pivoted.to_string())
    print("\n" + "-"*80)
    print("AVERAGE PERFORMANCE METRICS (AVG OVER TASKS)")
    print("-"*80)
    summary = df.groupby('Method')[['Top-1 Accuracy', 'Top-5 Accuracy', 'MRR', 'Diff-ID']].mean()
    print(summary.to_string())
else:
    print("No data collected.")
print("="*80 + "\n")
