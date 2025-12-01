# fix_deploy.py - Run this to fix deployment issues
import os

print("🔧 Fixing deployment issues...")

# 1. Update requirements.txt
reqs = """Flask==2.3.3
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
setuptools==65.0.0
wheel==0.38.4"""

with open('requirements.txt', 'w') as f:
    f.write(reqs)
print("✅ Updated requirements.txt")

# 2. Create runtime.txt
with open('runtime.txt', 'w') as f:
    f.write('python-3.9.18\n')
print("✅ Created runtime.txt (Python 3.9)")

# 3. Create Procfile
with open('Procfile', 'w') as f:
    f.write('web: python app.py\n')
print("✅ Created Procfile")

# 4. Update .gitignore
gitignore = """__pycache__/
*.pyc
*.pyo
*.pyd
.env
venv/
env/
.vscode/
.idea/
*.pkl
*.csv
*.txt
eda_plots/
best_model.pkl
model_comparison.csv
features.txt
__pycache__/
*.pyc
*.pyo
*.pyd
.DS_Store
"""

with open('.gitignore', 'w') as f:
    f.write(gitignore)
print("✅ Updated .gitignore")

print("\n🎯 Now run these commands:")
print("1. git add requirements.txt runtime.txt Procfile .gitignore")
print("2. git commit -m 'Fixed deployment dependencies'")
print("3. git push")
print("\nThen wait for Render to redeploy automatically!")