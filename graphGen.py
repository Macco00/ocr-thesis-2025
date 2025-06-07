import pandas as pd
import matplotlib.pyplot as plt
import os

# === Inställningar ===
base_dir = "results"
csv_path = os.path.join(base_dir, "metrics_log.csv")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# === Läs CER-resultat ===
df = pd.read_csv(csv_path)
df["Preprocessed"] = df["Preprocessed"].astype(bool)
engines = df["Engine"].unique()

# === CER per bild: stapeldiagram med och utan preprocessing, sorterat ===
for engine in engines:
    df_engine = df[df["Engine"] == engine].copy()
    df_orig = df_engine[df_engine["Preprocessed"] == False]
    df_pre = df_engine[df_engine["Preprocessed"] == True]

    # Identifiera gemensamma bilder
    common_images = set(df_orig["Image"]) & set(df_pre["Image"])
    df_orig = df_orig[df_orig["Image"].isin(common_images)].sort_values("CER (%)")
    df_pre = df_pre[df_pre["Image"].isin(common_images)].set_index("Image").loc[df_orig["Image"]].reset_index()

    if df_orig.empty or df_pre.empty:
        print(f"[INFO] Hoppar över {engine}, saknar kompletta data.")
        continue

    labels = df_orig["Image"].tolist()
    x = range(len(labels))

    plt.figure(figsize=(12, 5))
    plt.bar([i - 0.2 for i in x], df_orig["CER (%)"], width=0.4, label="Original")
    plt.bar([i + 0.2 for i in x], df_pre["CER (%)"], width=0.4, label="Preprocessed (CLAHE + OTSU)")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(f"CER per Image – {engine} (sorted by Original CER)")
    plt.xlabel("Image")
    plt.ylabel("CER (%)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(plots_dir, f"cer_CLAHEnOTSU_sorted_{engine.lower()}.png")
    plt.savefig(out_path)
    plt.close()
