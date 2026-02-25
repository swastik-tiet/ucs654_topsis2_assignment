import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

models = [
    "facebook/bart-large-cnn",
    "google/pegasus-xsum",
    "t5-large",
    "google/pegasus-cnn_dailymail",
    "philschmid/distilbart-cnn-12-6-samsum",
    "sshleifer/distilbart-cnn-12-6",
]

criteria = [
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "BERTScore",
    "Inference Time (s)",
    "Model Size (GB)",
]

impacts = [1, 1, 1, 1, -1, -1]

weights = np.array([0.25, 0.25, 0.20, 0.15, 0.10, 0.05])

data = np.array([
    [44.16, 21.28, 40.90, 0.894, 3.20, 1.63],
    [47.21, 24.56, 39.25, 0.901, 4.50, 2.28],
    [42.50, 20.69, 40.09, 0.880, 3.80, 2.75],
    [44.17, 21.47, 41.11, 0.893, 4.10, 2.28],
    [40.30, 18.20, 37.50, 0.871, 1.50, 0.52],
    [42.10, 19.80, 39.20, 0.878, 1.80, 0.82],
])


def run_topsis(data, weights, impacts):
    norm_matrix = data / np.sqrt((data ** 2).sum(axis=0))
    weighted_matrix = norm_matrix * weights
    impacts_arr = np.array(impacts)

    ideal_best = np.where(
        impacts_arr == 1,
        weighted_matrix.max(axis=0),
        weighted_matrix.min(axis=0)
    )
    ideal_worst = np.where(
        impacts_arr == 1,
        weighted_matrix.min(axis=0),
        weighted_matrix.max(axis=0)
    )

    dist_to_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_to_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    topsis_score = dist_to_worst / (dist_to_best + dist_to_worst)

    ranks = np.empty(len(topsis_score), dtype=int)
    ranks[topsis_score.argsort()[::-1]] = np.arange(1, len(topsis_score) + 1)

    return norm_matrix, weighted_matrix, ideal_best, ideal_worst, dist_to_best, dist_to_worst, topsis_score, ranks


norm, weighted, ideal_best, ideal_worst, d_best, d_worst, scores, ranks = run_topsis(data, weights, impacts)

results_df = pd.DataFrame({
    "Model": models,
    "ROUGE-1": data[:, 0],
    "ROUGE-2": data[:, 1],
    "ROUGE-L": data[:, 2],
    "BERTScore": data[:, 3],
    "Inf. Time (s)": data[:, 4],
    "Size (GB)": data[:, 5],
    "D+ (best)": d_best.round(4),
    "D- (worst)": d_worst.round(4),
    "TOPSIS Score": scores.round(4),
    "Rank": ranks,
}).sort_values("Rank")

print("=" * 90)
print("   TOPSIS ANALYSIS - BEST PRE-TRAINED TEXT SUMMARIZATION MODEL")
print("   Roll No: 102316020")
print("=" * 90)
print(results_df.to_string(index=False))
print("=" * 90)
print("Best Model:", results_df.iloc[0]["Model"])
print("TOPSIS Score:", results_df.iloc[0]["TOPSIS Score"])
print("=" * 90)

short_names = [m.split("/")[-1] for m in models]
ranked_idx = np.argsort(scores)[::-1]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

fig = plt.figure(figsize=(20, 16))
fig.suptitle(
    "TOPSIS Analysis - Best Pre-trained Text Summarization Model\nRoll No: 102316020",
    fontsize=16, fontweight="bold", y=0.98
)
gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

ax1 = fig.add_subplot(gs[0, :2])
bar_colors = ["#2ecc71" if i == ranked_idx[0] else "#3498db" for i in range(len(models))]
bars = ax1.bar(
    [short_names[i] for i in ranked_idx],
    [scores[i] for i in ranked_idx],
    color=[bar_colors[i] for i in ranked_idx],
    edgecolor="black", linewidth=0.5
)
ax1.set_title("TOPSIS Scores (Higher = Better)", fontweight="bold")
ax1.set_ylabel("Closeness Coefficient")
ax1.set_ylim(0, 1)
ax1.tick_params(axis="x", rotation=20)
for bar, idx in zip(bars, ranked_idx):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{scores[idx]:.3f}",
        ha="center", va="bottom", fontsize=9
    )
ax1.legend(handles=[mpatches.Patch(color="#2ecc71", label="Best Model")], loc="upper right")

ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")
table_rows = [[str(ranks[i]), short_names[i], f"{scores[i]:.4f}"] for i in ranked_idx]
tbl = ax2.table(
    cellText=table_rows,
    colLabels=["Rank", "Model", "Score"],
    cellLoc="center", loc="center",
    colWidths=[0.15, 0.55, 0.30]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for col in range(3):
    tbl[(1, col)].set_facecolor("#2ecc71")
ax2.set_title("Final Rankings", fontweight="bold")

ax3 = fig.add_subplot(gs[1, :2])
x = np.arange(len(models))
w = 0.25
ax3.bar(x - w, data[:, 0], w, label="ROUGE-1", color="#e74c3c")
ax3.bar(x, data[:, 1], w, label="ROUGE-2", color="#e67e22")
ax3.bar(x + w, data[:, 2], w, label="ROUGE-L", color="#f1c40f")
ax3.set_xticks(x)
ax3.set_xticklabels(short_names, rotation=20, ha="right")
ax3.set_title("ROUGE Scores Comparison", fontweight="bold")
ax3.set_ylabel("Score")
ax3.legend()

ax4 = fig.add_subplot(gs[1, 2], polar=True)
radar_categories = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]
N = len(radar_categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
radar_raw = data[:, :4].copy()
radar_norm = (radar_raw - radar_raw.min(0)) / (radar_raw.max(0) - radar_raw.min(0))
for ci, model_idx in enumerate(ranked_idx[:2]):
    vals = radar_norm[model_idx].tolist() + [radar_norm[model_idx][0]]
    color = ["#2ecc71", "#3498db"][ci]
    ax4.plot(angles, vals, "o-", linewidth=2, color=color, label=short_names[model_idx])
    ax4.fill(angles, vals, alpha=0.1, color=color)
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(radar_categories, size=8)
ax4.set_title("Top-2 Models Radar", fontweight="bold", pad=15)
ax4.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=7)

ax5 = fig.add_subplot(gs[2, 0])
sc = ax5.scatter(
    data[:, 4], data[:, 5],
    c=scores, cmap="RdYlGn",
    s=150, edgecolors="black", linewidth=0.5, zorder=3
)
for i, name in enumerate(short_names):
    ax5.annotate(name, (data[i, 4], data[i, 5]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)
plt.colorbar(sc, ax=ax5, label="TOPSIS Score")
ax5.set_xlabel("Inference Time (s)")
ax5.set_ylabel("Model Size (GB)")
ax5.set_title("Speed vs Size", fontweight="bold")
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(short_names, d_best, "o--", color="#e74c3c", label="D+ (to ideal best)")
ax6.plot(short_names, d_worst, "s--", color="#2ecc71", label="D- (to ideal worst)")
ax6.set_title("Separation Distances", fontweight="bold")
ax6.set_ylabel("Euclidean Distance")
ax6.tick_params(axis="x", rotation=25)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(gs[2, 2])
heat_df = pd.DataFrame(data, index=short_names, columns=criteria)
heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min())
sns.heatmap(heat_norm, ax=ax7, cmap="YlGn", annot=heat_df.round(2),
            fmt=".2f", linewidths=0.5, annot_kws={"size": 7})
ax7.set_title("Criteria Heatmap (normalized)", fontweight="bold")
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=30, ha="right", fontsize=7)
ax7.tick_params(axis="y", labelsize=7)

plt.savefig("topsis_results.png", dpi=150, bbox_inches="tight")
results_df.to_csv("topsis_results.csv", index=False)
print("Saved topsis_results.png and topsis_results.csv")
