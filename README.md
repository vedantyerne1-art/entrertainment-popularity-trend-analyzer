# Entertainment Popularity Trend Analyzer

Mobile-friendly Streamlit dashboard for historical United States Top 50 playlist analysis.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `Atlantic_United_States.csv` beside `app.py` before launching.

## Deployment

This is a Streamlit application. Vercel does not natively host Streamlit's long-running Python server, so the supported one-click deployment target is [Streamlit Community Cloud](https://share.streamlit.io/). Select this repository, choose `app.py` as the main file, and deploy.

For Vercel, the app would need to be rebuilt as a separate frontend/API application rather than deployed with a `vercel.json` wrapper around Streamlit.