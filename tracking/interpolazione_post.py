import pandas as pd

df = pd.read_csv("tracking/outputs/tracking_data.csv")

# Per ogni track_id
results = []
for tid, group in df.groupby("track_id"):
    group = group.sort_values("frame")
    group["frame_num"] = group["frame"].str.extract(r"(\d+)").astype(int)

    # Reindicizza con tutti i frame tra il primo e l'ultimo
    full_index = pd.RangeIndex(start=group["frame_num"].min(), stop=group["frame_num"].max() + 1)
    group = group.set_index("frame_num").reindex(full_index)

    # Interpola bbox linearmente
    for col in ["x1", "y1", "x2", "y2"]:
        group[col] = group[col].interpolate()

    group["track_id"] = tid
    group["class"] = group["class"].fillna(method="ffill")  # opzionale
    results.append(group.reset_index().rename(columns={"index": "frame_num"}))

# Unisci e salva
df_interp = pd.concat(results)
df_interp["frame"] = df_interp["frame_num"].apply(lambda x: f"frame_{x:04d}.jpg")
df_interp = df_interp[["frame", "track_id", "x1", "y1", "x2", "y2", "class"]]
df_interp.to_csv("tracking/outputs/tracking_data_interp.csv", index=False)
