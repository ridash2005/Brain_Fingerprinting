import os
import re
import pandas as pd
import glob
from datetime import datetime

def parse_section(content, header):
    """Extracts text between a specific header and the next header or end."""
    pattern = re.compile(rf"{header}\n-+\n(.*?)(?=\n\d+\. |\n=+\n|$)", re.DOTALL)
    match = pattern.search(content)
    return match.group(1).strip() if match else ""

def parse_key_value_lines(text):
    data = {}
    for line in text.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            data[key.strip()] = val.strip()
    return data

def parse_ablation_table(text):
    lines = text.split('\n')
    data = []
    # Skip header lines
    start_idx = 0
    for i, line in enumerate(lines):
        if 'raw_fc' in line:
            start_idx = i
            break
    
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) >= 4:
            data.append({
                'Method': parts[0],
                'Acc': parts[1],
                'Top-5': parts[2],
                'MRR': parts[3]
            })
    return data

def parse_sota_table(text):
    data = {}
    for line in text.split('\n'):
        parts = line.split()
        if len(parts) >= 2:
            # Key is everything except last, value is last
            val = parts[-1]
            key = " ".join(parts[:-1])
            data[key] = val
    return data

def parse_robustness(text):
    noise_data = {}
    sample_data = {}
    
    current_section = None
    for line in text.split('\n'):
        line = line.strip()
        if "Noise Robustness:" in line:
            current_section = 'noise'
            continue
        elif "Sample Size Robustness:" in line:
            current_section = 'sample'
            continue
        
        if current_section == 'noise' and 'sigma=' in line:
            parts = line.split(':')
            key = parts[0].strip()
            val = parts[1].strip()
            noise_data[key] = val
        elif current_section == 'sample' and 'N=' in line:
            parts = line.split(':')
            key = parts[0].strip()
            val = parts[1].strip()
            sample_data[key] = val
            
    return noise_data, sample_data

def parse_report(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    data = {}
    
    # 1. Dataset Info
    info_text = parse_section(content, r"1\. DATASET INFORMATION")
    data['info'] = parse_key_value_lines(info_text)
    
    # 2. Ablation Results
    ablation_text = parse_section(content, r"2\. ABLATION STUDY RESULTS \(Table 1\)")
    data['ablation'] = parse_ablation_table(ablation_text)
    
    # 3. SOTA Comparison
    sota_text = parse_section(content, r"3\. STATE-OF-THE-ART COMPARISON \(Table 2\)")
    data['sota'] = parse_sota_table(sota_text)
    
    # 4. Statistical Validation
    stat_text = parse_section(content, r"4\. STATISTICAL VALIDATION")
    data['stats'] = parse_key_value_lines(stat_text)
    
    # 5. Cross Validation
    cv_text = parse_section(content, r"5\. CROSS-VALIDATION RESULTS")
    data['cv'] = parse_key_value_lines(cv_text)
    
    # 6. Comprehensive Metrics
    comp_text = parse_section(content, r"6\. COMPREHENSIVE METRICS \(Proposed Method\)")
    data['metrics'] = parse_key_value_lines(comp_text)
    
    # 7. Robustness
    rob_text = parse_section(content, r"7\. ROBUSTNESS ANALYSIS")
    data['robustness_noise'], data['robustness_sample'] = parse_robustness(rob_text)
    
    # 8. Model Architecture
    model_text = parse_section(content, r"8\. MODEL ARCHITECTURE DETAILS")
    data['model_raw'] = model_text
    
    return data

def generate_markdown(all_results, output_file):
    with open(output_file, 'w') as f:
        # Title
        f.write("# Comprehensive Brain Fingerprinting Analysis Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Methodology Explanation
        f.write("## 1. Methodology & Metrics Explanation\n\n")
        f.write("### Evaluation Metrics\n")
        f.write("- **Identification Accuracy (Top-1):** The percentage of subjects correctly identified from the database (Rank-1).\n")
        f.write("- **Top-5 Accuracy:** The percentage of times the correct subject is present within the top 5 predicted matches.\n")
        f.write("- **Mean Reciprocal Rank (MRR):** A statistic measure for evaluating the return of a ranked list of answers. MRR is 1 if the first result is correct, 0.5 if second, etc.\n")
        f.write("- **Differential Identifiability:** The gap between the mean intra-subject similarity and mean inter-subject similarity. Higher is better.\n")
        
        f.write("\n### Statistical Tests\n")
        f.write("- **Permutation Test:** Evaluates if the performance is significantly better than random chance by shuffling labels (1000 iterations).\n")
        f.write("- **McNemar Test:** A paired non-parametric test to compare the proposed method against the baseline on a per-subject basis.\n")
        f.write("- **Bootstrap Confidence Intervals:** 95% CI calculated using 1000 bootstrap samples to estimate uncertainty.\n")

        # 2. Aggregate Summary
        f.write("\n## 2. Aggregate Performance Summary\n\n")
        f.write("| Task | Proposed Acc | Baseline Acc | Improvement | MRR | Diff. Ident. |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        summary_stats = []
        for task_name, r in all_results.items():
            prop_acc = float(r['data']['metrics'].get('Top-1 Accuracy', 0))
            finn_acc = float(r['data']['sota'].get('finn_2015', 0))
            mrr = float(r['data']['metrics'].get('Mean Reciprocal Rank', 0))
            diff_id = float(r['data']['metrics'].get('Differential Identifiability', 0))
            imp = (prop_acc - finn_acc) / finn_acc * 100 if finn_acc > 0 else 0
            
            summary_stats.append({
                'Task': task_name,
                'Proposed': prop_acc,
                'Baseline': finn_acc,
                'Imp': imp,
                'MRR': mrr,
                'DI': diff_id
            })
            
            f.write(f"| {task_name} | {prop_acc:.4f} | {finn_acc:.4f} | +{imp:.2f}% | {mrr:.4f} | {diff_id:.4f} |\n")
        
        # Averages
        df_sum = pd.DataFrame(summary_stats)
        avgs = df_sum.drop(columns=['Task']).mean()
        f.write(f"| **AVERAGE** | **{avgs['Proposed']:.4f}** | **{avgs['Baseline']:.4f}** | **+{avgs['Imp']:.2f}%** | **{avgs['MRR']:.4f}** | **{avgs['DI']:.4f}** |\n")

        # 3. Detailed Task Reports
        f.write("\n## 3. Detailed Task Analysis\n")
        
        for task_name, r in all_results.items():
            data = r['data']
            rel_path = r['rel_path'] # relative path from report root to result dir
            
            f.write(f"\n### 3.{list(all_results.keys()).index(task_name)+1} Task: {task_name}\n")
            f.write(f"**Source Directory:** `{rel_path}`\n\n")
            
            # 3.1 Metrics
            f.write("#### A. Comprehensive Metrics\n")
            f.write("| Metric | Value | Description |\n")
            f.write("|---|---|---|\n")
            metrics_def = {
                'Top-1 Accuracy': "Strict identification accuracy",
                'Top-3 Accuracy': "Correct match in top 3",
                'Top-5 Accuracy': "Correct match in top 5",
                'Mean Rank': "Average rank of correct subject",
                'Mean Reciprocal Rank': "Harmonic mean of ranks",
                'Differential Identifiability': "Separation between self/other"
            }
            for k, v in data['metrics'].items():
                desc = metrics_def.get(k, "")
                f.write(f"| {k} | {v} | {desc} |\n")
                
            # 3.2 Ablation
            f.write("\n#### B. Ablation Study (Component Analysis)\n")
            f.write("Comparison of different architectural choices:\n")
            f.write("| Method | Accuracy | Top-5 | MRR |\n")
            f.write("|---|---|---|---|\n")
            for row in data['ablation']:
                f.write(f"| {row['Method']} | {row['Acc']} | {row['Top-5']} | {row['MRR']} |\n")
                
            # 3.3 Statistical Validation
            f.write("\n#### C. Statistical Validation\n")
            f.write("Significance testing results:\n")
            f.write("| Test | Result | Interpretation |\n")
            f.write("|---|---|---|\n")
            stats = data['stats']
            f.write(f"| Bootstrap Mean | {stats.get('Bootstrap Mean Accuracy', 'N/A')} | Stability of the mean |\n")
            f.write(f"| 95% CI | {stats.get('95% Confidence Interval', 'N/A')} | Reliability range |\n")
            f.write(f"| Permutation Test | {stats.get('Permutation Test (vs Chance) p-value', 'N/A')} | P-value < 0.05 indicates significance over random |\n")
            f.write(f"| McNemar Test | {stats.get('McNemar Test p-value', 'N/A')} | P-value < 0.05 indicates significance over baseline |\n")
            
            # 3.4 Robustness
            f.write("\n#### D. Robustness Analysis\n")
            f.write("**Noise Robustness (Accuracy vs Sigma):**\n")
            f.write("- Evaluation of model performance when Gaussian noise is added to the input time series.\n\n")
            f.write("| Noise Level (Sigma) | Accuracy |\n")
            f.write("|---|---|\n")
            for k, v in data['robustness_noise'].items():
                f.write(f"| {k} | {v} |\n")
                
            f.write("\n**Sample Size Robustness (Accuracy vs N):**\n")
            f.write("- Evaluation of model performance with varying number of subjects in the database.\n\n")
            f.write("| Sample Size (N) | Accuracy |\n")
            f.write("|---|---|\n")
            for k, v in data['robustness_sample'].items():
                f.write(f"| {k} | {v} |\n")

            # 3.5 Images
            f.write("\n#### E. Visualizations\n")
            
            # Define images to look for and their descriptions
            image_map = {
                'heatmap_convae_sdl.png': "**Reconstruction Similarity Matrix (Proposed):**\nShows the similarity scores between all pairs of subjects. A strong diagonal indicates high self-similarity (correct identification) and low cross-similarity.",
                'ablation_results.png': "**Ablation Study:**\nBar chart comparing the accuracy of the proposed method against baselines and partial implementations.",
                'robustness.png': "**Robustness Analysis:**\nCurves showing how accuracy changes with increased noise and reduced sample sizes.",
                'dictionary_atoms.png': "**Learned Dictionary Atoms:**\nVisualization of the sparse components (atoms) learned by the K-SVD Dictionary Learning module, representing fundamental connectivity motifs.",
                'similarity_dist.png': "**Similarity Distributions:**\nHistograms of intra-subject (self) vs. inter-subject (others) similarity scores. Less overlap indicates better identifiability.",
                'full_correlation_matrix.png': "**Full Correlation Matrix:**\nRaw Functional Connectivity matrix visualization.",
                'heatmap_group_avg.png': "**Group Average Heatmap:**\nSimilarity matrix using simple group averaging."
            }
            
            for img_name, description in image_map.items():
                img_path = f"{rel_path}/{img_name}"
                # Check if image actually exists is hard without full walk, assuming yes if folder exists
                # In markdown, using relative path
                f.write(f"{description}\n\n")
                f.write(f"![{img_name}]({img_path})\n\n")
                
            f.write("---\n")

def main():
    base_dir = r'd:\GitHub\my_repo\Brain_Fingerprinting\results\hcp_fingerprinting_results_20260124_233031'
    if not os.path.exists(base_dir):
        # Fallback search
        base_dir = r'results/hcp_fingerprinting_results_20260124_233031'
        
    all_results = {}
    
    # Walk through directories
    # Only look 1 level deep
    root = base_dir
    items = sorted(os.listdir(root))
    
    for item in items:
        full_path = os.path.join(root, item)
        if os.path.isdir(full_path) and item.startswith('run_'):
            report_path = os.path.join(full_path, 'MANUSCRIPT_REPORT.txt')
            if os.path.exists(report_path):
                print(f"Processing {item}...")
                data = parse_report(report_path)
                
                # Determine task name cleanly
                task_name = data['info'].get('Current Analysis Task', 'UNKNOWN')
                
                # Relative path for Images (assuming report is in project root)
                # D:\GitHub\my_repo\Brain_Fingerprinting\COMPREHENSIVE_REPORT.md
                # D:\GitHub\my_repo\Brain_Fingerprinting\results\hcp_fingerprinting_results_...\run_...
                # Relative: results/hcp.../run...
                
                rel_path = f"results/hcp_fingerprinting_results_20260124_233031/{item}"
                
                all_results[task_name] = {
                    'data': data,
                    'full_path': full_path,
                    'rel_path': rel_path
                }
    
    if all_results:
        generate_markdown(all_results, 'COMPREHENSIVE_REPORT.md')
        print("Successfully generated COMPREHENSIVE_REPORT.md")
    else:
        print("No results found to process.")

if __name__ == "__main__":
    main()
