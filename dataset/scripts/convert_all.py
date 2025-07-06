import os
import glob
import subprocess
import sys

def main():
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Converters are in: ../converters (relative to this script)
    converter_dir = os.path.join(current_dir, '..', 'converters')
    # The root directory (parent of dataset)
    root_dir = os.path.dirname(os.path.dirname(current_dir))

    # List all .py files in converter_dir, excluding __init__.py
    converter_scripts = glob.glob(os.path.join(converter_dir, '*.py'))
    module_names = []
    for script in converter_scripts:
        base = os.path.basename(script)
        if base == '__init__.py':
            continue
        module_name = os.path.splitext(base)[0]
        module_names.append(module_name)

    # Sort to run in consistent order
    module_names.sort()

    for module_name in module_names:
        full_module_path = f"dataset.converters.{module_name}"
        print(f"Running {full_module_path}...")
        cmd = [sys.executable, "-m", full_module_path]
        try:
            subprocess.run(cmd, cwd=root_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error while running {full_module_path}: {e}")
            sys.exit(1)

    print("All converters finished successfully.")

if __name__ == "__main__":
    main()
