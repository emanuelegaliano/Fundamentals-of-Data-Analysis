from __future__ import annotations

import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from scipy import stats

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

def check_anova_assumptions(groups, labels=None, max_normaltest_n=5000):
    """
    Check basic assumptions for one-way ANOVA on a list of groups.
    
    Parameters
    ----------
    groups : list of array-like
        Each element is a 1D array/Series with the data for one group.
    labels : list of str, optional
        Names of the groups (same length as groups). If None, generic labels are used.
    max_normaltest_n : int, default 5000
        Maximum sample size used for the D'Agostino-Pearson normality test
        (if a group is larger, a random subsample of this size is used).
        
    Returns
    -------
    results : dict
        Dictionary with:
        - "levene": (stat, p)
        - "groups": list of dicts with per-group summary (label, n, std, skew, kurtosis, normaltest_stat, normaltest_p)
        - "variance_ratio": max(std^2) / min(std^2)
    """
    # Convert to numpy arrays and drop NaN
    clean_groups = []
    for g in groups:
        g = np.asarray(g)
        g = g[~np.isnan(g)]
        clean_groups.append(g)
    
    if labels is None:
        labels = [f"group_{i}" for i in range(len(clean_groups))]
    
    # Basic sizes
    print("=== Group sizes ===")
    for lab, g in zip(labels, clean_groups):
        print(f"{lab}: n = {len(g)}")
    
    # 1) Levene test for homogeneity of variances
    levene_stat, levene_p = stats.levene(*clean_groups)
    print("\n=== Levene test for equality of variances ===")
    print(f"Statistic = {levene_stat:.3f}, p-value = {levene_p:.3e}")
    
    # 2) Per-group shape: skewness, kurtosis, normality (D'Agostino K^2)
    group_results = []
    print("\n=== Per-group shape and normality (D'Agostino-Pearson) ===")
    for lab, g in zip(labels, clean_groups):
        std = np.std(g, ddof=1)
        skew = stats.skew(g, bias=False)
        kurt = stats.kurtosis(g, bias=False)  # excess kurtosis (0 for normal)
        
        # Normality test: use subsample if very large
        if len(g) >= 8:  # minimum recommended for normaltest
            if len(g) > max_normaltest_n:
                sample = np.random.default_rng(0).choice(g, size=max_normaltest_n, replace=False)
            else:
                sample = g
            k2_stat, k2_p = stats.normaltest(sample)
        else:
            k2_stat, k2_p = np.nan, np.nan
        
        print(f"{lab}: std = {std:.3f}, skew = {skew:.3f}, excess kurtosis = {kurt:.3f}, "
              f"normality p = {k2_p:.3e}")
        
        group_results.append({
            "label": lab,
            "n": len(g),
            "std": std,
            "skew": skew,
            "kurtosis": kurt,
            "normaltest_stat": k2_stat,
            "normaltest_p": k2_p
        })
    
    # 3) Simple numeric measure of heteroscedasticity: ratio of max/min variance
    variances = [gr["std"]**2 for gr in group_results if not np.isnan(gr["std"])]
    if len(variances) > 1:
        variance_ratio = max(variances) / min(variances)
    else:
        variance_ratio = np.nan
    
    print("\n=== Variance ratio (max variance - min variance) ===")
    print(f"Variance ratio = {variance_ratio:.3f}")
    
    if variance_ratio > 4:
        print("Warning: large variance ratio (> 4) suggests strong heteroscedasticity.")
    
    results = {
        "levene": (levene_stat, levene_p),
        "groups": group_results,
        "variance_ratio": variance_ratio
    }
    return results

CMAP_PIZZA = LinearSegmentedColormap.from_list(
    "pizza",
    [PALETTE["light_bg"], "#E0B085", "#C97A4A", PALETTE["red"]],
    N=256,
)

def plot_decision_boundary(
    X_2d: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    proba: np.ndarray,
    *,
    title: str,
    transparent_bg: bool = False,
    label0: str = "Interna (0)",
    label1: str = "Costiera (1)",
    show_colorbar: bool = True,
    levels: int = 13,
):
    """
    Plot a 2D decision surface (probability for class 1) plus scatter points.

    Parameters
    ----------
    X_2d : np.ndarray
        2D coordinates of samples, shape (n_samples, 2) (e.g., PCA projection).
    y : np.ndarray
        Binary labels (0/1), shape (n_samples,).
    xx, yy : np.ndarray
        Meshgrid arrays from np.meshgrid, shape (n_grid, n_grid).
    proba : np.ndarray
        Probability for class 1 on the mesh, shape like xx/yy (n_grid, n_grid).
    title : str
        Plot title.
    transparent_bg : bool
        If True, transparent background (useful for slides).
    label0, label1 : str
        Legend labels for classes 0 and 1.
    show_colorbar : bool
        Whether to draw a colorbar.
    levels : int
        Number of filled contour levels between 0 and 1.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    # --- defensive conversions ---
    X_2d = np.asarray(X_2d)
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba)

    fig, ax = plt.subplots(figsize=(10, 7))
    style_axes(ax, transparent_bg=transparent_bg)

    # Filled probability surface (class 1)
    cf = ax.contourf(
        xx,
        yy,
        proba,
        levels=np.linspace(0, 1, levels),
        cmap=CMAP_PIZZA,
        alpha=0.55,
        vmin=0,
        vmax=1,
    )

    # Decision threshold at 0.5
    ax.contour(
        xx,
        yy,
        proba,
        levels=[0.5],
        colors=[PALETTE["dark_gray"]],
        linewidths=2.2,
        alpha=0.85,
    )

    # Scatter points
    ax.scatter(
        X_2d[y == 0, 0],
        X_2d[y == 0, 1],
        s=14,
        alpha=0.55,
        c=[PALETTE["dark_gray"]],
        label=label0,
        edgecolors="none",
    )
    ax.scatter(
        X_2d[y == 1, 0],
        X_2d[y == 1, 1],
        s=14,
        alpha=0.55,
        c=[PALETTE["red"]],
        label=label1,
        edgecolors="none",
    )

    # Minimal colorbar
    if show_colorbar:
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.outline.set_alpha(0.25) # type: ignore
        cb.set_label("P(Coast)")

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    return fig, ax

def plot_pca_cumvar(
    cum_var: np.ndarray,
    explained_2d: float,
    *,
    transparent_bg: bool = False,
):
    """
    Plot cumulative explained variance from PCA and highlight 2-PC cutoff.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    style_axes(ax, transparent_bg=transparent_bg)

    ax.plot(
        np.arange(1, len(cum_var) + 1),
        cum_var,
        marker="o",
        lw=2.5,
        color=PALETTE["dark_gray"],
    )

    ax.axhline(
        explained_2d,
        linestyle="--",
        lw=2,
        color=PALETTE["red"],
        label=f"2 PCs = {explained_2d:.2%}",
    )

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA — cumulative explained variance")
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    return fig, ax


def search_kmeans_k(
    X: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 20,
    silhouette_sample: int | None = 2500,
    random_state: int = 42,
    n_init: int = 20,
) -> tuple[int, pd.DataFrame]:
    """
    Grid-search K for KMeans using inertia + silhouette.
    Optionally compute silhouette on a random subsample for speed.

    Returns
    -------
    best_k : int
        K with the highest silhouette score (on the chosen sample).
    k_result : pd.DataFrame
        Table with columns: k, inertia, silhouette
    """
    rng = np.random.RandomState(random_state)
    n = len(X)

    # Build silhouette sample (positional indices, consistent with numpy arrays)
    if silhouette_sample is not None and silhouette_sample < n:
        sample_idx = rng.choice(n, size=silhouette_sample, replace=False)
        X_sil = X.iloc[sample_idx]
    else:
        sample_idx = None
        X_sil = X

    rows: list[dict] = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        km.fit(X)

        # Compute silhouette on sample using predicted labels on that sample
        labels_sil = km.predict(X_sil)
        sil = silhouette_score(X_sil, labels_sil)

        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})

    k_result = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_k = int(k_result.loc[k_result["silhouette"].idxmax(), "k"]) # type: ignore
    return best_k, k_result


def explain_clusters_lift(
    df: pd.DataFrame,
    cluster_col: str,
    binary_cols: list[str],
    top_n_features: int = 10,
    min_presence: float = 0.30,
    lift_threshold: float = 0.10,
) -> dict[int, pd.DataFrame]:
    """
    Explain clusters using "lift" over global prevalence for binary (0/1) features.

    For each cluster:
      lift(feature) = mean_in_cluster(feature) - global_mean(feature)

    Keeps only features with:
      - cluster_mean >= min_presence
      - lift >= lift_threshold

    Returns
    -------
    out : dict[int, pd.DataFrame]
        cluster_id -> table with cluster_mean, global_mean, lift
    """
    if cluster_col not in df.columns:
        raise KeyError(f"cluster_col '{cluster_col}' not found in df.")

    # Keep only valid binary columns that exist and are truly 0/1
    bin_cols = [
        c for c in binary_cols
        if c in df.columns and df[c].dropna().isin([0, 1]).all()
    ]
    if not bin_cols:
        raise ValueError("No valid binary 0/1 columns found in binary_cols.")

    global_mean = df[bin_cols].mean(numeric_only=True)
    out: dict[int, pd.DataFrame] = {}

    for cl in sorted(df[cluster_col].dropna().unique()):
        cl = int(cl)
        cl_mean = df.loc[df[cluster_col] == cl, bin_cols].mean(numeric_only=True)
        lift = (cl_mean - global_mean).sort_values(ascending=False)

        tbl = pd.DataFrame(
            {
                "cluster_mean": cl_mean.loc[lift.index],
                "global_mean": global_mean.loc[lift.index],
                "lift": lift,
            }
        )

        tbl = tbl[
            (tbl["cluster_mean"] >= min_presence) &
            (tbl["lift"] >= lift_threshold)
        ].head(top_n_features)

        out[cl] = tbl

    return out

def plot_kmeans_k_diagnostics(
    k_result: pd.DataFrame,
    best_k: int,
    *,
    transparent_bg: bool = False,
    title_prefix: str = "KMeans hyperparameter search",
):
    """
    Plot silhouette score and inertia (elbow) vs k using project palette + styling.

    Parameters
    ----------
    k_result : pd.DataFrame
        Output from `search_kmeans_k`, with columns: k, inertia, silhouette.
    best_k : int
        Selected k to highlight.
    transparent_bg : bool
        If True, transparent background (slides).
    title_prefix : str
        Prefix used in plot titles.

    Returns
    -------
    (fig1, ax1), (fig2, ax2)
    """
    if not {"k", "silhouette", "inertia"}.issubset(set(k_result.columns)):
        raise ValueError("k_result must contain columns: ['k','silhouette','inertia'].")

    # --- Silhouette plot ---
    fig1, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(
        k_result["k"],
        k_result["silhouette"],
        marker="o",
        linewidth=2.5,
        color=PALETTE["dark_gray"],
    )
    ax1.axvline(
        best_k,
        linestyle="--",
        linewidth=2.2,
        color=PALETTE["red"],
        alpha=0.9,
        label=f"best_k = {best_k}",
    )
    ax1.set_title(f"{title_prefix} — Silhouette")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Silhouette score")
    ax1.legend(frameon=False, loc="best")
    style_axes(ax1, transparent_bg=transparent_bg)
    fig1.tight_layout()

    # --- Elbow (inertia) plot ---
    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    ax2.plot(
        k_result["k"],
        k_result["inertia"],
        marker="o",
        linewidth=2.5,
        color=PALETTE["dark_gray"],
    )
    ax2.axvline(
        best_k,
        linestyle="--",
        linewidth=2.2,
        color=PALETTE["red"],
        alpha=0.9,
        label=f"best_k = {best_k}",
    )
    ax2.set_title(f"{title_prefix} — Elbow (inertia)")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Inertia (within-cluster SSE)")
    ax2.legend(frameon=False, loc="best")
    style_axes(ax2, transparent_bg=transparent_bg)
    fig2.tight_layout()

    return (fig1, ax1), (fig2, ax2)

def plot_cluster_sizes(
    df: pd.DataFrame,
    cluster_col: str,
    *,
    title: str = "Cluster sizes",
    transparent_bg: bool = False,
    pattern: bool = False,
    tint_alpha: float = 0.12,
):
    counts = df[cluster_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        counts.index.astype(str),
        counts.values, # type: ignore
        color=PALETTE["red"],
        edgecolor="none",
        zorder=2,
    )

    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of items")

    style_axes(ax, transparent_bg=transparent_bg, pattern=pattern, tint_alpha=tint_alpha)
    fig.tight_layout()
    return fig, ax, counts

def plot_box_by_cluster(
    df: pd.DataFrame,
    cluster_col: str,
    value_col: str,
    *,
    title: str | None = None,
    ylabel: str | None = None,
    showfliers: bool = False,
    transparent_bg: bool = False,
    pattern: bool = False,
    tint_alpha: float = 0.12,
):
    clusters = sorted(df[cluster_col].dropna().unique())
    data = [df.loc[df[cluster_col] == c, value_col].dropna().values for c in clusters] # type: ignore

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data,
        labels=[str(c) for c in clusters], # type: ignore
        patch_artist=True,
        showfliers=showfliers,
    )

    # project styling (data = red, reference/median = green)
    for box in bp["boxes"]:
        box.set_facecolor(PALETTE["red"])
        box.set_edgecolor(PALETTE["dark_gray"])
    for whisker in bp["whiskers"]:
        whisker.set_color(PALETTE["dark_gray"])
    for cap in bp["caps"]:
        cap.set_color(PALETTE["dark_gray"])
    for median in bp["medians"]:
        median.set_color(PALETTE["green"])

    ax.set_title(title or f"{value_col} by cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel(ylabel or value_col)

    style_axes(ax, transparent_bg=transparent_bg, pattern=pattern, tint_alpha=tint_alpha)
    fig.tight_layout()
    return fig, ax
