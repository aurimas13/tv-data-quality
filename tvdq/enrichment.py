# ⚡ 📝 3.1  Fast, batched keyword extraction (GPU-aware & HF-authenticated)

# ① Authenticate once (expects HF_TOKEN saved in Colab “Secrets” or env var)
login(token=os.getenv("HF_TOKEN"))      # Tools ▸ Secrets ▸ add HF_TOKEN

# ② Pick device and load models (first pull is ~90 MB ⇒ 1-2 min with token)
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
kw_model    = KeyBERT(model=embed_model)

# ③ Batched extraction to avoid per-row overhead
def extract_keywords_batched(texts, batch_size=512, top_n=5):
    """
    texts: list[str]  →  returns list[list[str]] of keywords per doc.
    Uses MMR for diversity and processes up to `batch_size` docs at once.
    """
    results = []
    for start in tqdm(range(0, len(texts), batch_size)):
        chunk = texts[start:start + batch_size]
        kw_chunk = kw_model.extract_keywords(
            chunk,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
            use_mmr=True           # diversify phrases
        )
        # kw_chunk is list[list[(kw, score)]] – keep only kw strings
        results.extend([[kw for kw, _ in doc] for doc in kw_chunk])
    return results

# ④ Run on the whole column (2-3 × faster on GPU; ~5 × faster than row-loop)
descriptions = df["description"].fillna("").tolist()
df["keywords"] = extract_keywords_batched(descriptions, batch_size=512, top_n=5)
