# Social Media Engagement Predictor

A Streamlit app that estimates engagement for a planned Facebook post. It predicts the chance of landing in the dataset's top engagement quartile and estimates total interactions. The post type is selectable from a dropdown, and a short summary updates with the selected publish date, time, and post details. The app also supports editable post idea comparisons, suggested weekday/hour combinations, model comparisons, feature importance, and a filterable, downloadable data explorer.

## Run locally on Windows

Open PowerShell in this folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py -3` is unavailable, install Python 3.10 or newer from [python.org](https://www.python.org/downloads/) and make sure the Python launcher is enabled. The app opens at `http://localhost:8501`.

The app downloads the UCI Facebook Metrics dataset on first run. If the download is unavailable, it uses generated sample data and displays a warning. To run offline with the real data, place the UCI `dataset_Facebook.csv` file in this folder.

## Deploy on Streamlit Community Cloud

1. Upload the deployable app files: `app.py`, `social_data.py`, `requirements.txt`, `README.md`, `.gitignore`, and `.streamlit/config.toml`. Do not upload `.venv` or `social_media_models.joblib`.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and choose **Create app**.
3. Select your repository and branch, set the app file path to `app.py`, optionally choose an app URL, then select **Deploy**.
4. Streamlit installs the packages from `requirements.txt`. The app will be available at its `*.streamlit.app` address. Future commits to the selected branch update the deployment.

If your project is not yet in GitHub, create a repository there, then in PowerShell from this folder:

```powershell
git init
git add app.py social_data.py requirements.txt README.md .gitignore .streamlit/config.toml
git commit -m "Prepare Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the final remote URL with your repository's URL. The original practical notebooks are optional for running the app; select `app.py` as the Streamlit entrypoint.

## Project files

- `app.py` — Streamlit user interface and prediction flow.
- `social_data.py` — data loading, cleaning, target creation, and feature preparation.
- The original practical notebooks are optional; the Streamlit app runs from the Python files above.

The app trains cached Random Forest models on first start. If `social_media_models.joblib` exists beside `app.py` and its feature columns match, the app loads those saved models instead. Only load model files you trust.
