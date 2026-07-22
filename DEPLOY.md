# Deploying the live demo (free)

The app runs anywhere Streamlit runs. Two free options; **Option A is the chosen path**.

---

## Option A — Streamlit Community Cloud (chosen) · one manual step

The repo is already on GitHub: `https://github.com/Nadercr7/rag-pdf-qa`.
Streamlit Community Cloud deploys straight from it — the only part that requires a human is
connecting your GitHub account and pasting the secret.

**Click-by-click:**

1. Open **https://share.streamlit.io** and **sign in with GitHub** (the `Nadercr7` account).
2. Click **Create app** → *"Deploy a public app from GitHub"*.
3. Fill in:
   - **Repository:** `Nadercr7/rag-pdf-qa`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick a subdomain, e.g. `rag-pdf-qa` → the app will live at
     `https://rag-pdf-qa.streamlit.app`
4. Open **Advanced settings…** before deploying:
   - **Python version:** `3.11` (matches development).
   - **Secrets** — paste the line below, **copying the value from your local `.env`**
     (never commit these keys anywhere):

     ```toml
     GEMINI_API_KEYS = "PASTE,YOUR,COMMA,SEPARATED,KEYS,HERE"
     ```

     ⚠️ The secrets box is **TOML, not dotenv**: copying the raw `KEY=value` line from
     `.env` fails with *"Invalid format: please enter valid TOML"*. The value must be
     wrapped in double quotes (`KEY = "value"`), one `KEY = "..."` per line.

5. Click **Deploy**. First build takes ~2–5 minutes. On first load the app auto-ingests the
   sample corpus (a few seconds), then it's ready.
6. Smoke-test it: ask *"How many vacation days do full-time employees get?"* → expect a
   grounded answer citing `meridian_employee_handbook.pdf — p.1`. Then ask
   *"Does the company offer pet insurance?"* → expect exactly
   *"I couldn't find this in the documents."*

**Troubleshooting**

| Symptom | Fix |
|---|---|
| `RuntimeError: unsupported version of sqlite3` (chromadb) | Uncomment the `pysqlite3-binary` line in `requirements.txt` and push — the code auto-swaps it in when present. |
| `No Gemini API key configured` | The secret name must be exactly `GEMINI_API_KEYS` (or `GEMINI_API_KEY`), set in app **Settings → Secrets**. |
| Rate-limit errors under load | Free tier is ~10 req/min **per key** — add more comma-separated keys to the secret. |

Notes: the container filesystem is **ephemeral** — restarts wipe `chroma_db/` and the app
re-seeds the sample corpus automatically. Uploaded PDFs disappear on restart (fine for a demo).

---

## Option B — Hugging Face Spaces (alternative, also free)

1. Create a Space: **huggingface.co → New Space** → SDK **Streamlit**, name e.g. `rag-pdf-qa`.
2. Add this YAML **frontmatter at the very top of `README.md`** (HF requires it):

   ```yaml
   ---
   title: Grounded RAG PDF QA
   emoji: 📄
   colorFrom: blue
   colorTo: green
   sdk: streamlit
   sdk_version: "1.60.0"
   app_file: app.py
   ---
   ```

3. Push the repo to the Space (username = your HF username, password = a **write** token from
   *Settings → Access Tokens*):

   ```bash
   git remote add space https://huggingface.co/spaces/<HF_USER>/rag-pdf-qa
   git push space main
   ```

4. In the Space: **Settings → Variables and secrets → New secret** →
   name `GEMINI_API_KEYS`, value = your comma-separated keys (they arrive as env vars —
   the app reads them automatically).
5. The Space rebuilds on every push; the app serves on the Space URL.

---

## Local "deployment" (for the client)

```bash
pip install -r requirements.txt
copy .env.example .env      # add GEMINI_API_KEY(S) or OPENAI_API_KEY
streamlit run app.py
```

Switching the demo to the client's OpenAI account is two lines in the host secrets:
`LLM_PROVIDER = "openai"` and `OPENAI_API_KEY = "sk-..."` (re-ingest happens automatically in
the namespaced `pdfs_openai_1536` collection on first use).
