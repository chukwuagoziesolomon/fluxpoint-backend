# CELL 1: SETUP & DEPENDENCIES
# This cell clones the repo and installs all required packages

# Step 1: Clone the fluxpoint repository (if not already cloned)
import os
if not os.path.exists('/content/fluxpoint'):
    !git clone https://github.com/chukwuagoziesolomon/fluxpoint-backend.git /content/fluxpoint
    print("Repository cloned successfully")
else:
    print("Repository already exists")

# Step 2: PULL LATEST CHANGES (includes validation fixes!)
import subprocess
os.chdir('/content/fluxpoint')
result = subprocess.run(['git', 'pull', 'origin', 'main'], capture_output=True, text=True)
print("Git pull output:")
print(result.stdout)
if result.stderr:
    print("Git errors:")
    print(result.stderr)

# Step 3: Add to Python path
import sys
sys.path.insert(0, '/content/fluxpoint')

# Step 4: Install dependencies
!pip install -q pandas numpy scikit-learn torch yfinance

print("\n" + "="*80)
print("CELL 1 COMPLETE: Environment ready with latest code fixes")
print("="*80)
