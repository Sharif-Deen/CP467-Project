import subprocess
import sys

## Run each file in order so prerequisite files are generated.
def main():
    subprocess.run([sys.executable, "Code/Preprocessing.py"], check=True)
    subprocess.run([sys.executable, "Code/feature_extraction.py"], check=True)
    subprocess.run([sys.executable, "Code/rule_based_classifier.py"], check=True)
    subprocess.run([sys.executable, "Code/machine_learning.py"], check=True)

if __name__ == "__main__":
    main()
