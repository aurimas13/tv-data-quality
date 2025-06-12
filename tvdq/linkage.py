# ← clustering logic

# 🕸️ 4.1  Record linkage – title blocking + SBERT clustering
def normalise_title(t): return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()

df["title_norm"] = df["name"].apply(normalise_title)
blocks = {}
for idx, title in enumerate(df["title_norm"]):
    key = title[:25]          # crude first-N char block
    blocks.setdefault(key, []).append(idx)

cluster_ids = np.full(len(df), -1)
current_cluster = 0

for idxs in blocks.values():
    if len(idxs) == 1:
        cluster_ids[idxs[0]] = current_cluster; current_cluster += 1
        continue
    emb = model.encode(df.loc[idxs, "title_norm"].tolist(), show_progress_bar=False)
    clust = AgglomerativeClustering(n_clusters=None, distance_threshold=0.45, metric='cosine').fit(emb)
    for local, global_idx in enumerate(idxs):
        cluster_ids[global_idx] = current_cluster + clust.labels_[local]
    current_cluster += clust.labels_.max() + 1

df["cluster_id"] = cluster_ids
print("Created", df["cluster_id"].nunique(), "content groupings")
