cd C:\Users\louan\Desktop\fragment_hit_triage

python -m py_compile app.py

git init
git add .
git commit -m "Deploy Streamlit Cloud app"
git branch -M main

# Replace YOUR_USERNAME with your GitHub username before running this line:
git remote add origin https://github.com/YOUR_USERNAME/fragment_hit_triage.git

git push -u origin main
