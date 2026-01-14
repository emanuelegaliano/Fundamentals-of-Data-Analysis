from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

PALETTE = {
    "red": "#902F1A",
    "green": "#564F13",
    "dark_gray": "#2B2B2B",
    "light_bg": "#F4EBDC",
    "accent": "#D1832F",
}

def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r, g, b)

COL_RED  = _hex_to_rgb01(PALETTE["red"])
COL_DARK = _hex_to_rgb01(PALETTE["dark_gray"])
COL_BG   = _hex_to_rgb01(PALETTE["light_bg"])
PIZZA_SCALE = [PALETTE["light_bg"], "#E0B085", "#C97A4A", PALETTE["red"]]

def set_matplotlib_defaults():
    """One place for global matplotlib defaults (optional)."""
    plt.rcParams["axes.titlepad"] = 10
    plt.rcParams["font.size"] = 11

def _add_diagonal_pattern(ax, color, alpha=0.18, lw=1.2, spacing=0.12):
    """
    Adds diagonal lines in AXES coordinates (0..1), independent of data limits.
    Useful for your 'slide-like' background.
    """
    lines = []
    bs = np.arange(-1.0, 2.0, spacing)
    for b in bs:
        x0, x1 = -0.2, 1.2
        y0, y1 = x0 + b, x1 + b
        lines.append([(x0, y0), (x1, y1)])

    lc = LineCollection(
        lines,
        colors=[color],
        linewidths=lw,
        alpha=alpha,
        transform=ax.transAxes,
        zorder=0,
        clip_on=True
    )
    ax.add_collection(lc)

def style_axes(ax, transparent_bg: bool = False, pattern: bool = False, tint_alpha: float = 0.0):
    """
    Unified axes styling:
    - transparent or light background
    - subtle grid + spines
    - optional tint + diagonal pattern
    """
    fig = ax.figure

    if transparent_bg:
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(PALETTE["light_bg"])
        fig.patch.set_alpha(1)
        ax.set_facecolor(PALETTE["light_bg"])

        if tint_alpha and tint_alpha > 0:
            ax.axhspan(0, 1, transform=ax.transAxes, color=PALETTE["accent"], alpha=tint_alpha, zorder=0)

        if pattern:
            _add_diagonal_pattern(ax, color=PALETTE["red"], alpha=0.12, lw=1.0, spacing=0.11)

    ax.grid(True, alpha=0.10)
    for sp in ax.spines.values():
        sp.set_color((0, 0, 0, 0.25))
        sp.set_linewidth(1.0)

    ax.tick_params(colors=PALETTE["dark_gray"])
    ax.xaxis.label.set_color(PALETTE["dark_gray"])
    ax.yaxis.label.set_color(PALETTE["dark_gray"])
    ax.title.set_color(PALETTE["dark_gray"])

# -----------------------------
# Duplicates helper (reporting)
# -----------------------------
def check_row_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return all duplicate rows (keep=False) sorted for inspection."""
    return df[df.duplicated(keep=False)].sort_values(list(df.columns))

# -----------------------------
# Safe sum helper (feature engineering)
# -----------------------------
def safe_sum(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Row-wise sum that won't crash if cols is empty; coerces to numeric and fills NaN."""
    if not cols:
        return pd.Series(0, index=frame.index, dtype="float64")
    return frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

# -----------------------------
# Geo helpers: haversine + coastal check
# -----------------------------
EARTH_RADIUS_KM = 6371.0

def haversine_distance_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points (km)."""
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c

def is_coastal(lat, lon, coastline_points, max_distance_km=25) -> bool:
    """
    True if (lat,lon) is within max_distance_km from any point in coastline_points.
    coastline_points: iterable of (lat, lon)
    """
    min_dist = float("inf")
    for clat, clon in coastline_points:
        d = haversine_distance_km(lat, lon, clat, clon)
        if d < min_dist:
            min_dist = d
    return min_dist <= max_distance_km

# -----------------------------
# Model evaluation visuals (ROC + confusion)
# -----------------------------
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def plot_roc_story(y_true, pred_prob, transparent_bg=False, title_extra=""):
    fpr, tpr, _ = roc_curve(y_true, pred_prob)
    auc_val = roc_auc_score(y_true, pred_prob)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(fpr, tpr, linewidth=2.8, color=PALETTE["red"], label=f"ROC (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color=PALETTE["dark_gray"], alpha=0.35)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve {title_extra}".strip())

    ax.legend(frameon=False, labelcolor=PALETTE["dark_gray"])
    style_axes(ax, transparent_bg=transparent_bg)
    return fig, ax, auc_val

def plot_confusion_story(y_true, pred, labels=("Interna", "Costiera"), transparent_bg=False):
    cm = confusion_matrix(y_true, pred)
    cm_max = cm.max() if cm.max() > 0 else 1
    norm = cm / cm_max

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    # manual RGBA "colormap": diagonal -> red, off-diagonal -> dark gray
    img = np.zeros((2, 2, 4))
    for i in range(2):
        for j in range(2):
            alpha = 0.15 + 0.70 * norm[i, j]
            if i == j:
                r, g, b = _hex_to_rgb01(PALETTE["red"])
                img[i, j] = (r, g, b, alpha)
            else:
                r, g, b = _hex_to_rgb01(PALETTE["dark_gray"])
                img[i, j] = (r, g, b, 0.10 + 0.40 * norm[i, j])

    ax.imshow(img, interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, color=PALETTE["light_bg"])

    style_axes(ax, transparent_bg=transparent_bg)
    ax.grid(False)
    return fig, ax

def plot_smooth_cdf(
    df,
    price_col,
    bw_adjust=1.2,
    x_max_clip=30,
    transparent_bg=False
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # --- colori progetto ---
    COL_COAST = PALETTE["red"]  # rosso pizza
    COL_INLAND = PALETTE["dark_gray"]   # grigio scuro
    BG = PALETTE["light_bg"]  # beige chiaro

    def smooth_cdf_from_kde(x, grid, bw_adjust):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        kde = gaussian_kde(x)
        kde.set_bandwidth(kde.factor * bw_adjust)
        pdf = kde(grid)
        cdf = np.cumsum((pdf[:-1] + pdf[1:]) / 2 * np.diff(grid))
        cdf = np.insert(cdf, 0, 0.0)
        return cdf / cdf[-1]

    coast = df.loc[df["coastal_city"], price_col].dropna()
    inland = df.loc[df["internal_city"], price_col].dropna()

    xmax = min(x_max_clip, np.nanpercentile(pd.concat([coast, inland]), 99.5))
    grid = np.linspace(0, xmax, 450)

    cdf_coast = smooth_cdf_from_kde(coast, grid, bw_adjust)
    cdf_inland = smooth_cdf_from_kde(inland, grid, bw_adjust)

    # punto di massima separazione
    diff = np.abs(cdf_coast - cdf_inland)
    idx = np.argmax(diff)
    x_star = grid[idx]
    y1, y2 = cdf_coast[idx], cdf_inland[idx]

    fig, ax = plt.subplots(figsize=(11, 6))

    # --- sfondo ---
    if transparent_bg:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
    else:
        ax.set_facecolor(BG)

    # curve
    ax.plot(grid, cdf_coast, lw=3, color=COL_COAST, label="Città costiere")
    ax.plot(grid, cdf_inland, lw=3, color=COL_INLAND, label="Città interne")

    # area di separazione
    ax.fill_between(grid, cdf_coast, cdf_inland, color=COL_COAST, alpha=0.12)

    # linea verticale chiave (unica rimasta)
    ax.axvline(x_star, color="gray", lw=1.5, ls="--")

    # punti evidenziati
    ax.scatter([x_star], [y1], color=COL_COAST, zorder=5)
    ax.scatter([x_star], [y2], color=COL_INLAND, zorder=5)

    # annotazione
    ax.annotate(
        f"Differenza massima ≈ {abs(y2 - y1):.2f}\n(intorno a ${x_star:.1f})",
        xy=(x_star, (y1 + y2) / 2),
        xytext=(x_star + 2, 0.35),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=12,
        color=COL_INLAND
    )

    # --- pulizia assi ---
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)

    # rimuove tutti i tick x tranne quello “narrativo”
    ax.set_xticks([round(x_star, 1)])
    ax.set_xticklabels([f"${x_star:.1f}"])

    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0%", "50%", "100%"])

    ax.set_xlabel("Prezzo (USD)")
    ax.set_ylabel("Quota cumulata")

    ax.grid(False)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    plt.show()

def clean_strings(s: pd.Series) -> pd.Series:
    '''
    Clean string data by normalizing unicode characters, stripping whitespace,
    replacing multiple spaces with a single space, and converting common
    missing value indicators to pandas NA.
    '''
    s = s.str.normalize('NFKC').str.strip().str.replace(r'\s+', ' ', regex=True)
    s = s.replace({r'(?i)^(na|n/a|none|null|nan)$': pd.NA, 
                   r'^$': pd.NA}, regex=True)
    return s

def returnTopN(series: pd.Series, n: int) -> pd.Series:
    '''
    Return a Series containing the top N most frequent values from the input Series.
    '''
    return series.value_counts().head(n)

def plotCategorialVsNumeric(
    df: pd.DataFrame,
    categorialCol: str,
    numericCol: str,
    top_n: int
) -> None:
    '''
    Plot density plots of a numeric column for the top N categories of a categorial column.
    '''

    top_categories = returnTopN(df[categorialCol], top_n).index
    plt.figure(figsize=(12, 6))
    for category in top_categories:
        subset = df[df[categorialCol] == category]
        sns.kdeplot(
            subset[numericCol].dropna(),
            label=str(category),
            fill=True,
            alpha=0.5
        )
    plt.title(f"Density Plot of {numericCol} by Top {top_n} Categories of {categorialCol}")
    plt.xlabel(numericCol)
    plt.ylabel("Density")
    plt.legend(title=categorialCol)
    plt.grid(True)
    plt.show()