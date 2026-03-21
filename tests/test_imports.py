import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

print("Testing updated imports...")

try:
    from src.models.conv_ae import ConvAutoencoder
    print("Maske src.models.conv_ae: SUCCESS")
except ImportError as e:
    print(f"Make src.models.conv_ae: FAILED - {e}")

try:
    from src.models.conv_ae_fp import ConvAutoencoderFP
    print("Make src.models.conv_ae_fp: SUCCESS")
except ImportError as e:
    print(f"Make src.models.conv_ae_fp: FAILED - {e}")

try:
    from src.models.sparse_dictionary_learning import k_svd
    print("Make src.models.sparse_dictionary_learning: SUCCESS")
except ImportError as e:
    print(f"Make src.models.sparse_dictionary_learning: FAILED - {e}")

try:
    from src.utils.config_parser import parse_basic_params
    print("Make src.utils.config_parser: SUCCESS")
except ImportError as e:
    print(f"Make src.utils.config_parser: FAILED - {e}")

try:
    from src.utils.hcp_io import load_timeseries
    print("Make src.utils.hcp_io: SUCCESS")
except ImportError as e:
    print(f"Make src.utils.hcp_io: FAILED - {e}")

try:
    from src.utils.matrix_ops import calculate_accuracy
    print("Make src.utils.matrix_ops: SUCCESS")
except ImportError as e:
    print(f"Make src.utils.matrix_ops: FAILED - {e}")

# Analysis Module Tests
analysis_modules = [
    'statistical_validation', 'ablation_studies', 'interpretability',
    'robustness_analysis', 'state_of_art_comparison', 'evaluation_metrics',
    'cross_validation', 'dataset_description', 'run_complete_analysis'
]

for mod in analysis_modules:
    try:
        exec(f"from src.analysis.{mod} import *")
        print(f"Make src.analysis.{mod}: SUCCESS")
    except ImportError as e:
        print(f"Make src.analysis.{mod}: FAILED - {e}")
    except Exception as e:
        print(f"Make src.analysis.{mod}: LOAD ERROR - {e}")

print("Import test validation complete.")
