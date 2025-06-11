# TV-Data-Quality&nbsp;📺🔍

A reproducible, end-to-end toolkit for **cleaning, enriching and de-duplicating
TV-programme metadata**.  
Built for DS/DE interviews but ready for real-world catalogues.

---

## 1 · What’s inside?

| Folder / file | Purpose |
|---------------|---------|
| `notebooks/tv_data_quality.ipynb` | **Colab-friendly notebook** – run the whole pipeline in minutes. |
| `tvdq/` | Re-usable Python package<br>• `cleaning.py` – text fixes, type inference, year parsing<br>• `enrichment.py` – SBERT embeddings + KeyBERT keywords<br>• `linkage.py` – blocking & agglomerative clustering |
| `app/main.py` | **FastAPI** micro-service exposing `/enrich` and `/dedupe` endpoints. |
| `Dockerfile`  | 1-command build for the API. |
| `requirements.txt` | Python deps frozen for Python 3.11. |
| `Makefile` | Common helpers (`make lint test api`). |
| `README.md` | You’re here. |

---

## 2 · Quick start in Google Colab

1. **Open the notebook**  
   <https://colab.research.google.com/github/aurimas13/tv-data-quality/blob/main/notebooks/tv_data_quality.ipynb>

2. **Runtime ▶︎ Change runtime type ▶︎ GPU** (recommended).

3. **Upload the sample** when prompted (`TV_sample.zip`).

4. *(Optional but faster)* Add a free [Hugging Face token](https://huggingface.co/settings/tokens)  
   * Colab menu **Tools ▸ Secrets** → key `HF_TOKEN`, value `hf_…`.

5. **Run-all** – the notebook will  
   * install deps,  
   * profile a 10 k sample,  
   * clean & enrich records,  
   * cluster duplicates,  
   * save `tv_data_cleaned.parquet`,  
   * emit quick charts for your slide deck.

Typical wall-clock: **~3 min** on a T4 GPU for the full 200 k-row sample.

---

## 3 · Local reproduction

```bash
# clone & enter
git clone https://github.com/aurimas13/tv-data-quality.git
cd tv-data-quality

# create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the notebook locally or as a script
jupyter lab
# OR
python -m tvdq.cli --in TV_sample.csv --out tv_data_cleaned.parquet
