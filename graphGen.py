import pandas as pd
import matplotlib.pyplot as plt
import os

# Läs in CSV-filen
csv_path = os.path.join("results", "metrics_log.csv")
df = pd.read_csv(csv_path)

# Skapa undermapp för grafer om den inte finns
plots_dir = os.path.join("results", "plots")
os.makedirs(plots_dir, exist_ok=True)

# Konvertera till rätt datatyper
df["Preprocessed"] = df["Preprocessed"].astype(bool)

# Plot: CER per engine (original vs preprocessed)
engines = df["Engine"].unique()

for engine in engines:
    subset = df[df["Engine"] == engine]
    pre = subset[subset["Preprocessed"] == True]
    orig = subset[subset["Preprocessed"] == False]

    plt.figure()
    plt.plot(orig["Image"], orig["CER (%)"], label="Original", marker="o")
    plt.plot(pre["Image"], pre["CER (%)"], label="Preprocessed", marker="x")
    plt.title(f"CER per Image - {engine}")
    plt.xlabel("Image")
    plt.ylabel("CER (%)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"cer_plot_{engine.lower()}.png"))
    plt.close()

# Plot: Genomsnittlig CER per motor
avg_cer = df.groupby(["Engine", "Preprocessed"])["CER (%)"].mean().reset_index()

plt.figure()
for processed in [False, True]:
    data = avg_cer[avg_cer["Preprocessed"] == processed]
    label = "Preprocessed" if processed else "Original"
    plt.bar(data["Engine"] + (" (P)" if processed else ""), data["CER (%)"], label=label)

plt.title("Average CER per Engine")
plt.ylabel("CER (%)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "avg_cer_comparison.png"))
plt.close()
