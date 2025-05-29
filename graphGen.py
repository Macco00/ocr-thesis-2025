import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# === Inställningar ===
base_dir = "results"
csv_path = os.path.join(base_dir, "metrics_log.csv")
params_path = os.path.join(base_dir, "params_log.csv")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# === Läs CER-resultat ===
df = pd.read_csv(csv_path)
df["Preprocessed"] = df["Preprocessed"].astype(bool)

# === Extrahera motornamn ===
engines = df["Engine"].unique()

# === Stapelgrafer per bild (sorterat efter original CER) ===
for engine in engines:
    subset = df[df["Engine"] == engine].copy()

    if engine in ["Google", "Amazon"]:
        subset = subset[subset["Preprocessed"] == False]

    orig = subset[subset["Preprocessed"] == False]
    pre = subset[subset["Preprocessed"] == True]

    # Sortera bilder efter original CER
    orig = orig.sort_values("CER (%)")
    if not pre.empty:
        pre = pre.set_index("Image").reindex(orig["Image"]).reset_index()

    labels = orig["Image"].tolist()
    x = range(len(labels))

    plt.figure(figsize=(12, 5))
    plt.bar([i - 0.2 for i in x], orig["CER (%)"], width=0.4, label="Original")

    if engine == "Tesseract" and not pre.empty:
        plt.bar([i + 0.2 for i in x], pre["CER (%)"], width=0.4, label="Preprocessed")

    plt.xticks(x, labels, rotation=45)
    plt.title(f"CER per Image - {engine}" + (" (Only Original)" if engine in ["Google", "Amazon"] else ""))
    plt.xlabel("Image (sorted by original CER)")
    plt.ylabel("CER (%)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(plots_dir, f"cer_bars_{engine.lower()}_sorted.png")
    plt.savefig(out_path)
    plt.close()

# === Trösklingsanalys för Tesseract: CER vs Block Size per C ===
if os.path.exists(params_path):
    param_df = pd.read_csv(params_path)
    param_df["C"] = param_df["C"].astype(int)
    param_df["Block Size"] = param_df["Block Size"].astype(int)
    param_df["CER (%)"] = param_df["CER (%)"].astype(float)

    # === 1. Linjeplot (per C) ===
    plt.figure(figsize=(10, 6))
    for c_val in sorted(param_df["C"].unique()):
        sub = param_df[param_df["C"] == c_val].sort_values("Block Size")
        plt.plot(sub["Block Size"], sub["CER (%)"], marker='o', label=f"C = {c_val}")

    plt.title("Tesseract: CER vs Block Size (per C)")
    plt.xlabel("Block Size")
    plt.ylabel("CER (%)")
    plt.legend(title="C Value", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tesseract_cer_vs_blocksize_per_c.png"))
    plt.close()

    # === 2. Heatmap ===
    pivot = param_df.pivot_table(index="Block Size", columns="C", values="CER (%)")
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        cbar_kws={"label": "CER (%)"},
        linewidths=0.5,
        linecolor="gray"
    )
    plt.title("Tesseract CER (%) – Heatmap (Block Size × C)")
    plt.xlabel("C")
    plt.ylabel("Block Size")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tesseract_heatmap_blocksize_c.png"))
    plt.close()

    # === 3. Scatter plot ===
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        param_df["Block Size"],
        param_df["C"],
        c=param_df["CER (%)"],
        s=param_df["CER (%)"] * 5 + 10,
        cmap="coolwarm",
        edgecolor="black"
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("CER (%)")
    plt.title("Tesseract CER (%) – Scatter (Block Size vs C)")
    plt.xlabel("Block Size")
    plt.ylabel("C")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tesseract_scatter_blocksize_c.png"))
    plt.close()
else:
    print("[INFO] Hittar inte params_log.csv – hoppar över tröskelgraf.")

    # === Analys: Tröskelval beroende på original CER ===
if os.path.exists(csv_path) and os.path.exists(params_path):
    print("[INFO] Bygger scatter plots över valda tröskelparametrar beroende på original CER...")
    
    df_metrics = pd.read_csv(csv_path)
    df_params = pd.read_csv(params_path)

    # Endast Tesseract-rader
    df_tess_orig = df_metrics[(df_metrics["Engine"] == "Tesseract") & (df_metrics["Preprocessed"] == False)]
    df_tess_pre = df_metrics[(df_metrics["Engine"] == "Tesseract") & (df_metrics["Preprocessed"] == True)]

    # Slå ihop original CER med params
    merged = df_params.merge(
        df_tess_orig[["Image", "CER (%)"]],
        on="Image",
        how="inner",
        suffixes=('', '_Original')
    )

    # Lägg till preprocessed CER för att räkna förbättring
    merged = merged.merge(
        df_tess_pre[["Image", "CER (%)"]],
        on="Image",
        how="inner",
        suffixes=('', '_Preprocessed')
    )

    # Räkna förbättring
    merged["Improvement"] = merged["CER (%)_Original"] - merged["CER (%)_Preprocessed"]

    # === Plot 1: Original CER vs C ===
    plt.figure(figsize=(10, 6))
    scatter1 = plt.scatter(
        merged["CER (%)_Original"],
        merged["C"],
        c=merged["Improvement"],
        s=merged["Improvement"].clip(lower=0)*10 + 20,
        cmap="RdYlGn",
        edgecolor="black"
    )
    cbar = plt.colorbar(scatter1)
    cbar.set_label("CER Improvement (%)")
    plt.title("Vald C beroende på original CER (Tesseract)")
    plt.xlabel("Original CER (%)")
    plt.ylabel("Vald C")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scatter_c_vs_original_cer.png"))
    plt.close()

    # === Plot 2: Original CER vs Block Size ===
    plt.figure(figsize=(10, 6))
    scatter2 = plt.scatter(
        merged["CER (%)_Original"],
        merged["Block Size"],
        c=merged["Improvement"],
        s=merged["Improvement"].clip(lower=0)*10 + 20,
        cmap="RdYlGn",
        edgecolor="black"
    )
    cbar = plt.colorbar(scatter2)
    cbar.set_label("CER Improvement (%)")
    plt.title("Vald Block Size beroende på original CER (Tesseract)")
    plt.xlabel("Original CER (%)")
    plt.ylabel("Vald Block Size")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scatter_blocksize_vs_original_cer.png"))
    plt.close()
else:
    print("[INFO] Saknar en av loggfilerna – hoppar över scatter-analyser.")

