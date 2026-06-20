# Deployment Guide — Streamlit Community Cloud

I can't deploy this for you directly (deploying needs your own GitHub and
Streamlit Cloud accounts, which I don't have access to), but everything in
this folder is now deployment-ready. This should take about 10 minutes.

## 0. Check your folder structure first

`app.py` and `run_pipeline.py` both expect `scraper.py`, `clusterer.py`,
and `synthesizer.py` to live in a `src/` subfolder next to them (that's
what `sys.path.insert(0, "src")` in both files is doing). Before you push
to GitHub, your project folder should look like:

```
your-repo/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
└── src/
    ├── scraper.py
    ├── clusterer.py
    └── synthesizer.py
```

If your files are currently all flat in one folder, create a `src/`
subfolder and move `scraper.py`, `clusterer.py`, and `synthesizer.py`
into it. (Locally, the imports also work if everything is flat in the
same folder, since Python checks the script's own directory too — but
match the structure above so it's unambiguous and matches what the code
was written expecting.)

## 1. Push to GitHub

```bash
cd your-repo
git init                          # skip if already a git repo
git add app.py run_pipeline.py requirements.txt .env.example .gitignore src/
git commit -m "Spotify Discovery Review Intelligence Engine"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Do **not** commit your real `.env` file or anything in `data/` — the
`.gitignore` in this folder already excludes both.

## 2. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **"New app"**.
3. Pick your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **"Advanced settings"** before deploying:
   - Under **Secrets**, paste:
     ```toml
     GROQ_API_KEY = "your-actual-groq-key-here"
     ```
   - (Optional) add `TWITTER_BEARER_TOKEN` the same way if you're using the live Twitter toggle.
5. Click **Deploy**. First build takes 3-5 minutes (installing `sentence-transformers`/`torch` is the slow part).

You'll get a public URL like `https://your-app-name.streamlit.app` — that's
the link for deliverable #1 and #3 in your fellowship brief.

## 3. Before you record or share the live link

- Open the deployed app yourself first. Toggle on App Store + Play Store +
  Community Forums (all free, no credentials), leave Seed Reviews **off**,
  and click **Run Engine**. Confirm:
  - The semantic cluster map actually renders (this was broken before the
    `umap_x`/`umap_y` fix — if it's still blank, something didn't deploy
    correctly).
  - No card shows an amber "⚠ FALLBACK CONTENT" banner. If one does, your
    Groq free-tier rate limit was hit — wait a minute and re-run, or check
    the key is set correctly in Secrets.
- A reviewer testing your live link is exactly the person who might hit a
  Groq rate limit mid-demo. The fallback banner means that's now visible
  and honest instead of silently fabricated — but it's still better to
  hand over a link where the first run already succeeded cleanly.

## 4. Updating after this deploy

Streamlit Cloud auto-redeploys on every push to your connected branch.
For any further changes: edit locally, commit, push — the live link
updates itself within a minute or two, no redeploy step needed.

## If you'd rather not use Streamlit Community Cloud

Railway and Render both support Streamlit apps with a similar free-tier
flow (connect GitHub repo → set `GROQ_API_KEY` as an environment variable
→ set the start command to `streamlit run app.py --server.port $PORT
--server.address 0.0.0.0`). Streamlit Community Cloud is the path of
least resistance since it's purpose-built for this exact use case, but
either alternative works if you've already got an account on one of them.
