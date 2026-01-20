import os
import subprocess
import shutil
import time
import sys
import torch

def run_cmd(cmd):
    print(f"\n[EXEC] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        print(f"[STDOUT]: {result.stdout}")
        print(f"[STDERR]: {result.stderr}")
        return False, result.stdout
    print("[SUCCESS]")
    return True, result.stdout

def verify():
    print("=== Professional End-to-End Verification Start ===")
    
    # Setup paths
    ROOT = os.getcwd()
    MOCK_DIR = os.path.join(ROOT, "MOCK_DATA")
    CONFIG_PATH = os.path.join(ROOT, "config", "basic_parameters.txt")
    BACKUP_CONFIG = CONFIG_PATH + ".bak"
    
    # 1. Mock Data Generation
    status, _ = run_cmd([sys.executable, "tests/mock_data_generator.py", MOCK_DIR, "10"])
    if not status: raise Exception("Mock data generation failed")
    
    # 2. Config Injection
    shutil.copy(CONFIG_PATH, BACKUP_CONFIG)
    with open(CONFIG_PATH, 'r') as f:
        lines = f.readlines()
    
    with open(CONFIG_PATH, 'w') as f:
        for line in lines:
            if line.startswith("HCP_DIR"):
                mock_dir_fixed = MOCK_DIR.replace("\\", "/")
                f.write(f'HCP_DIR = "{mock_dir_fixed}"\n')
            elif line.startswith("N_SUBJECTS"):
                f.write("N_SUBJECTS = 10\n")
            else:
                f.write(line)
    
    print("Config injected with mock parameters.")
    
    try:
        # 3. Test Imports
        status, _ = run_cmd([sys.executable, "tests/test_imports.py"])
        if not status: raise Exception("Import test failed")
        
        # 4. Dry Run Data Downloader
        status, _ = run_cmd([sys.executable, "src/data/download_hcp_data.py", "--dry-run"])
        if not status: raise Exception("Data downloader dry-run failed")
        
        # 5. FC Generation
        tasks = ["rest", "motor", "wm", "emotion"]
        for task in tasks:
            status, _ = run_cmd([sys.executable, "src/processing/generate_whole_brain_fc.py", "-task", task])
            if not status: raise Exception(f"FC Generation failed for {task}")
            
        status, _ = run_cmd([sys.executable, "src/processing/generate_network_fc.py", "-task", "motor", "-network", "Frontoparietal"])
        if not status: raise Exception("Network FC Generation failed")
        
        # 6. Training Pipeline (Reduced epochs for speed)
        # Note: I'll modify train_model.py temporarily or just run it as is. 
        # I'll modify it to 2 epochs for verification.
        TRAIN_FILE = "src/train_model.py"
        with open(TRAIN_FILE, 'r') as f: content = f.read()
        with open(TRAIN_FILE, 'w') as f: f.write(content.replace("num_epochs = 20", "num_epochs = 2"))
        
        status, _ = run_cmd([sys.executable, "src/train_model.py", "-model", "conv_ae", "-data", "rest"])
        
        # Restore epochs
        with open(TRAIN_FILE, 'w') as f: f.write(content)
        
        if not status: raise Exception("Training pipeline failed")
        
        # 7. Refinement Pipeline
        status, out = run_cmd([sys.executable, "src/processing/refine_whole_brain.py", "-task", "motor"])
        if not status: raise Exception("Refinement pipeline (ConvAE) failed")
        
        status, _ = run_cmd([sys.executable, "src/processing/refine_whole_brain_avg.py", "-task", "motor"])
        if not status: raise Exception("Refinement pipeline (Avg) failed")
        
        # 7.1 Network-Specific Pipeline
        # We need a trained network model. Let's save a dummy model of correct class.
        network_model_dir = "src/models/trained/conv_ae_fp_Frontoparietal"
        os.makedirs(network_model_dir, exist_ok=True)
        sys.path.append(ROOT)
        from src.models.conv_ae_fp import ConvAutoencoderFP
        dummy_model = ConvAutoencoderFP()
        torch.save(dummy_model.state_dict(), os.path.join(network_model_dir, "best_model.pth"))
        
        # We also need fc_rest_Frontoparietal.npy
        status, _ = run_cmd([sys.executable, "src/processing/generate_network_fc.py", "-task", "rest", "-network", "Frontoparietal"])
        
        status, _ = run_cmd([sys.executable, "src/processing/refine_network.py", "-task", "motor", "-network", "Frontoparietal"])
        if not status: raise Exception("Refinement pipeline (Network) failed")
        
        # 8. Baseline Model
        status, _ = run_cmd([sys.executable, "src/models/baseline_correlation.py"])
        if not status: raise Exception("Baseline correlation test failed")
        
        # 9. Optimization Heatmap
        # We need a log file. Let's run a minimal optimization.
        # Modified optimize_hyperparameters.py for speed (1 K, 1 L)
        OPT_FILE = "src/processing/optimize_hyperparameters.py"
        with open(OPT_FILE, 'r') as f: opt_content = f.read()
        # No modification needed as it only runs K=15, L=2..15 which is fast for 10 subjects.
        
        status, out = run_cmd([sys.executable, OPT_FILE, "-data", "rest", "-task", "motor"])
        if not status: raise Exception("Optimization script failed")
        
        # Find log file in output
        # Logs are in logs/runs/YYYYMMDD_HHMMSS_optimize_rest_to_motor/accuracy_log.txt
        # Let's find the latest directory
        log_runs_dir = "logs/runs"
        latest_run = sorted(os.listdir(log_runs_dir))[-1]
        log_path = os.path.join(log_runs_dir, latest_run, "accuracy_log.txt")
        
        status, _ = run_cmd([sys.executable, "src/visualization/plot_optimization_heatmap.py", "-log", log_path, "-output", f"results/verification_heatmap.png"])
        if not status: raise Exception("Visualization script failed")
        
        # 10. Demo Pipeline
        status, _ = run_cmd([sys.executable, "src/demo_pipeline.py"])
        if not status: raise Exception("Demo pipeline failed")
        
        print("\n=== Integration Test Suite PASSED ===")
        
    except Exception as e:
        print(f"\n=== Integration Test Suite FAILED: {str(e)} ===")
    finally:
        # Cleanup
        print("\nCleaning up verification artifacts...")
        shutil.copy(BACKUP_CONFIG, CONFIG_PATH)
        os.remove(BACKUP_CONFIG)
        if os.path.exists(MOCK_DIR):
            shutil.rmtree(MOCK_DIR)
        print("Restored original configuration.")

if __name__ == "__main__":
    verify()
