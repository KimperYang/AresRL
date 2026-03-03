import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# Data
# -----------------------
data = pd.DataFrame({
    "Model": ["gpt-4o", "gpt-5", "gemini-3-pro-preview"] * 2,
    "Dataset": ["Old Data"] * 3 + ["New Data"] * 3,
    "Accuracy": [49.8, 77.9, 84.7, 39.9, 57.0, 62.7]
})

# -----------------------
# Style
# -----------------------
sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=1.0
)

palette = {
    "Old Data": "#4C72B0",   # muted blue
    "New Data": "#DD8452"    # muted orange
}

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=data,
    x="Model",
    y="Accuracy",
    hue="Dataset",
    palette=palette,
    width=0.55   # 柱子更细
)

# -----------------------
# Labels & formatting
# -----------------------
ax.set_title("Accuracy: Old vs New Data", pad=12)
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("")

ax.set_ylim(0, 100)
ax.legend(title="", frameon=True)

# 在柱子上标数值
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=11)

sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
