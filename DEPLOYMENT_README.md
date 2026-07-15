# Streamlit Cloud Deployment Files

Put these files in the root of your `fragment_hit_triage` project:

```text
fragment_hit_triage/
├── app.py
├── requirements.txt
├── packages.txt
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Files

### `.gitignore`
Prevents GitHub from uploading local virtual environments, cache files, large outputs, and large full-data CSV files.

### `requirements.txt`
Python packages that Streamlit Cloud should install.

If your deployed app fails on RDKit, try removing this line:

```txt
rdkit
```

Then redeploy. Your app can still work if your code has a non-RDKit fallback.

### `packages.txt`
Optional Linux system packages. These help with some chemistry/plotting dependencies.

### `.streamlit/config.toml`
Basic Streamlit settings. It also raises upload size to 200 MB.

## Recommended deployment workflow

1. Copy these files into your project root.
2. Do not upload full raw data to GitHub.
3. Commit and push the project to GitHub.
4. In Streamlit Community Cloud:
   - Repository: your GitHub repo
   - Branch: `main`
   - Main file path: `app.py`
   - Python version: choose the same version you used locally if possible

## Important

For the online app, prefer uploading a small result file such as:

```text
outputs/top10000_for_app.csv
```

Do not run the 1.34M-row full scoring job inside the free hosted app. Run full scoring locally first, then upload the reduced result file to the web app.
