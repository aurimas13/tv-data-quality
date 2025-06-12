# ← clean_text, infer_type, …

# 🧹 2.1  Basic cleaning helpers
def clean_text(s):
    if pd.isna(s): return np.nan
    s = re.sub(r"\s+", " ", str(s)).strip()
    s = re.sub(r"[“”\"']", "", s)
    return s

for col in ["name", "description", "actors", "director", "genre", "country"]:
    df[col] = df[col].apply(clean_text)

# Standardise content_type (movie/series/event/other)
def infer_type(row):
    if pd.notna(row["content_type"]): return row["content_type"]
    if re.search(r"(s\d+e\d+|episode|season)", str(row["name"]), re.I):
        return "series"
    if re.search(r"( vs |basket|football|euroleague|world cup)", str(row["name"]), re.I):
        return "event"
    return "movie"
df["content_type"] = df.apply(infer_type, axis=1)

# Impute year from name "(1999)" pattern
year_rx = re.compile(r"\((19|20)\d{2}\)")
df["year"] = df.apply(
    lambda r: r["year"] if pd.notna(r["year"]) else
    int(year_rx.search(r["name"]).group(0)[1:-1]) if year_rx.search(r["name"]) else np.nan,
    axis=1
)
