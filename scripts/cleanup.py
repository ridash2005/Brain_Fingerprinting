import os
import shutil

def cleanup():
    # Directories to completely remove
    dirs_to_remove = [
        'FC_DATA',
        'MOCK_DATA',
    ]
    
    # Directories to clear but keep
    dirs_to_clear = [
        'results',
        'logs',
        'src/models/trained'
    ]
    
    print("Starting professional repository cleanup...")
    
    # Remove specific directories
    for d in dirs_to_remove:
        path = os.path.abspath(d)
        if os.path.exists(path):
            print(f"Removing {d}...")
            shutil.rmtree(path)
            
    # Clear directory contents but keep the folder
    for d in dirs_to_clear:
        path = os.path.abspath(d)
        if os.path.exists(path):
            print(f"Clearing contents of {d}...")
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            os.makedirs(path, exist_ok=True)

    # Clean up __pycache__
    print("Cleaning up __pycache__ directories...")
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                path = os.path.join(root, d)
                print(f"Removing {path}...")
                shutil.rmtree(path)
                
    # Remove backup files
    print("Removing backup files...")
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.bak'):
                path = os.path.join(root, f)
                print(f"Removing {path}...")
                os.unlink(path)

    print("\nCleanup complete. Repository is now in a pristine state.")

if __name__ == "__main__":
    cleanup()
