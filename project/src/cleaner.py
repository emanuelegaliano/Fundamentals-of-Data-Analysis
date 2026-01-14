from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import Birch


# -----------------------------
# Normalization helpers
# -----------------------------

_HTML_ENTITY_RE = re.compile(r"&[a-z]+;")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")

def _basic_normalize(text: str) -> str:
    """Lowercase + very lightweight cleanup suitable for menu-item strings."""
    if text is None:
        return ""
    s = str(text).lower().strip()
    # remove common html entities
    s = _HTML_ENTITY_RE.sub(" ", s)
    # normalize separators
    s = s.replace("/", " ").replace("|", " ").replace("-", " ")
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


# -----------------------------
# Attribute extraction patterns
# -----------------------------

# Sizes (categorical: keep first match as "main" size, but also create dummies)
_SIZE_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(xl|x\s*large|xlarge|extra\s*large)\b", "xlarge"),
    (r"\b(large|lg)\b", "large"),
    (r"\b(medium|md)\b", "medium"),
    (r"\b(small|sm)\b", "small"),
    (r"\b(personal|individual)\b", "personal"),
    (r"\b(family)\b", "family"),
    (r"\b(kids?|child)\b", "kids"),
]

# Diet / dietary flags
_DIET_PATTERNS: List[Tuple[str, str]] = [
    (r"\bgluten\s*[- ]?\s*free\b|\bgf\b", "glutenfree"),
    (r"\bvegan\b", "vegan"),
    (r"\bvegetarian\b|\bveggie\b", "vegetarian"),
    (r"\b(keto)\b", "keto"),
    (r"\b(halal)\b", "halal"),
]

# Product / format flags
_PRODUCT_FLAG_PATTERNS: Dict[str, List[str]] = {
    "is_slice": [
        r"\b(slice|by\s+the\s+slice|per\s+slice)\b",
    ],
    "is_calzone": [
        r"\bcalzone(s)?\b",
    ],
    "is_stromboli": [
        r"\bstromboli(s)?\b",
    ],
    "is_garlicbread": [
        r"\bgarlic\s*bread\b",
    ],
    "is_breadsticks": [
        r"\bbread\s*sticks?\b",
    ],
    "is_pasta": [
        r"\b(pasta|spaghetti|penne|rigatoni|lasagna)\b",
    ],
    # "is_pizza" is intentionally NOT a strict flag; most items are pizza-ish.
}

# Crust / base style flags (boolean dummies)
_CRUST_FLAG_PATTERNS: Dict[str, List[str]] = {
    "crust_thin": [
        r"\bthin\s*crust\b",
        r"\bthin\b",
    ],
    "crust_thick": [
        r"\bthick\s*crust\b",
        r"\bthick\b",
    ],
    "crust_pan": [
        r"\bpan\b",
        r"\bpan\s*pizza\b",
    ],
    "crust_deepdish": [
        r"\bdeep\s*dish\b",
        r"\bdeepdish\b",
    ],
    "crust_sicilian": [
        r"\bsicilian\b",
    ],
    "crust_handtossed": [
        r"\bhand\s*tossed\b",
        r"\bhandtossed\b",
    ],
    "crust_newyork": [
        r"\bnew\s*york\b",
        r"\bny\s*style\b",
    ],
    "crust_chicago": [
        r"\bchicago\b",
        r"\bchicago\s*style\b",
    ],
    "crust_stuffed": [
        r"\bstuffed\s*crust\b",
        r"\bstuffed\b",
    ],
    "crust_glutenfree": [
        r"\bgluten\s*[- ]?\s*free\s*crust\b",
        r"\bgf\s*crust\b",
    ],
    "crust_cauliflower": [
        r"\bcauliflower\s*crust\b",
        r"\bcauli\s*crust\b",
    ],
    "crust_wholewheat": [
        r"\bwhole\s*wheat\b",
        r"\bwholewheat\b",
        r"\bwheat\s*crust\b",
    ],
}

# Other "style" tags (these are useful for clustering text even if not exported as dummies)
_STYLE_TAG_PATTERNS: List[Tuple[str, str]] = [
    (r"\bwhite\b", "white"),
    (r"\bmargherita\b", "margherita"),
    (r"\b(bbq|barbecue)\b", "bbq"),
    (r"\b(buffalo)\b", "buffalo"),
]

# marketing/noise phrases that often add little semantic value
_NOISE_PATTERNS: List[str] = [
    r"\b(best|famous|award\s*winning|signature|special|classic|original)\b",
    r"\b(chef\s*special|house\s*special)\b",
    r"\b(combo|combination)\b",
    r"\b(pizza)\b",  # optional: remove the word "pizza" because it's in almost all values
]


# -----------------------------
# Extraction logic
# -----------------------------

@dataclass
class Extracted:
    enriched_text: str          # tags + cleaned residual
    residual_text: str          # cleaned residual only
    attrs: Dict[str, int]       # dummy-like features (0/1 and a few categorical->dummies)


def _apply_flag_patterns(s: str, patterns: Dict[str, List[str]]) -> Tuple[Dict[str, int], str]:
    """Detect boolean flags with regex patterns, optionally remove matched terms from text."""
    flags: Dict[str, int] = {}
    for flag, pats in patterns.items():
        hit = 0
        for pat in pats:
            if re.search(pat, s):
                hit = 1
                s = re.sub(pat, " ", s)
        flags[flag] = hit
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return flags, s


def _extract_attributes(text: str) -> Extracted:
    """
    Extract explainable signals and build an enriched representation for clustering.

    Output:
    - attrs: dummy features (ints 0/1)
    - enriched_text: "tag=..." tokens + residual tokens
    """
    s = _basic_normalize(text)

    # remove marketing/noise
    for pat in _NOISE_PATTERNS:
        s = re.sub(pat, " ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()

    attrs: Dict[str, int] = {}

    # 1) Size (categorical but exported as dummies)
    size_value: Optional[str] = None
    for pat, lbl in _SIZE_PATTERNS:
        if re.search(pat, s):
            size_value = lbl
            s = re.sub(pat, " ", s)
            break
    for _, lbl in _SIZE_PATTERNS:
        attrs[f"size_{lbl}"] = 1 if (size_value == lbl) else 0

    # 2) Diet flags
    diet_hits: List[str] = []
    for pat, lbl in _DIET_PATTERNS:
        if re.search(pat, s):
            diet_hits.append(lbl)
            s = re.sub(pat, " ", s)
    for _, lbl in _DIET_PATTERNS:
        attrs[f"diet_{lbl}"] = 1 if (lbl in diet_hits) else 0

    # 3) Product flags (slice/calzone/stromboli/...)
    prod_flags, s = _apply_flag_patterns(s, _PRODUCT_FLAG_PATTERNS)
    attrs.update(prod_flags)

    # 4) Crust flags
    crust_flags, s = _apply_flag_patterns(s, _CRUST_FLAG_PATTERNS)
    attrs.update(crust_flags)

    # 5) Topping count (1 topping, 2 toppings...)
    topping_count = None
    m = re.search(r"\b(\d+)\s*topping(s)?\b", s)
    if m:
        try:
            topping_count = int(m.group(1))
            s = re.sub(r"\b\d+\s*topping(s)?\b", " ", s)
        except ValueError:
            topping_count = None
    # Export coarse bins as dummies (helps for modeling later)
    for k in ["topping_1", "topping_2", "topping_3plus"]:
        attrs[k] = 0
    if topping_count == 1:
        attrs["topping_1"] = 1
    elif topping_count == 2:
        attrs["topping_2"] = 1
    elif isinstance(topping_count, int) and topping_count >= 3:
        attrs["topping_3plus"] = 1

    # 6) Extra style tags (not all become dummy cols; but we include them in enriched text)
    style_tags: List[str] = []
    for pat, lbl in _STYLE_TAG_PATTERNS:
        if re.search(pat, s):
            style_tags.append(lbl)
            s = re.sub(pat, " ", s)

    # final residual cleanup
    s = _MULTI_SPACE_RE.sub(" ", s).strip()

    # Build enriched tokens (these strongly guide TF-IDF clustering)
    tags: List[str] = []
    if size_value:
        tags.append(f"size={size_value}")
    for d in diet_hits:
        tags.append(f"diet={d}")

    # crust tags for clustering (even if dummies already exist)
    for flag, hit in crust_flags.items():
        if hit:
            tags.append(f"crust={flag.replace('crust_','')}")

    # product tags for clustering
    for flag, hit in prod_flags.items():
        if hit:
            tags.append(f"kind={flag.replace('is_','')}")

    # extra style tags
    for t in style_tags:
        tags.append(f"style={t}")

    enriched = " ".join(tags + ([s] if s else []))
    return Extracted(enriched_text=enriched.strip(), residual_text=s, attrs=attrs)


# -----------------------------
# Main API
# -----------------------------

def clean_categorical_column(
    df: pd.DataFrame,
    column: str,
    new_column: str,
    cluster_id_column: Optional[str] = None,
    *,
    birch_threshold: float = 0.6,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
    extra_stopwords: Optional[Iterable[str]] = None,
    add_dummy_features: bool = True,
    dummy_prefix: Optional[str] = None,
    keep_residual_text: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster a categorical-text column quickly and (optionally) add dummy feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Source text column (e.g., "menus.name").
    new_column : str
        Output column containing a representative/cleaned label per cluster.
    cluster_id_column : Optional[str]
        If provided, write the integer cluster id for each row.
    birch_threshold : float
        Main clustering knob. Lower => more clusters; higher => fewer clusters.
    min_df : int
        TF-IDF min_df (applied on unique strings). Increase to speed up and reduce noise.
    ngram_range : (int, int)
        N-gram range for TF-IDF. (1,2) usually works well for menu names.
    extra_stopwords : Optional[Iterable[str]]
        Extra stopwords (lowercased) to remove.
    add_dummy_features : bool
        If True, create dummy columns extracted from text (0/1) and append to df.
    dummy_prefix : Optional[str]
        Prefix for dummy columns. If None, uses f"{column}__".
    keep_residual_text : bool
        If True, also create a "{new_column}__residual" column with residual text (no tags).

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with `new_column` plus optional cluster id and dummy columns.
    cluster_summary : pd.DataFrame
        One row per cluster with counts, examples, and (if dummies enabled) feature prevalence.

    Notes
    -----
    - Clusters only UNIQUE values (fast).
    - Uses sparse TF-IDF + BIRCH (scales well).
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in df.")

    s = df[column].astype("string")

    # Unique values (excluding NA)
    unique_vals = s.dropna().unique()
    unique_vals = pd.Series(unique_vals, dtype="string")

    # Extract enriched text + attributes for each unique value
    extracted = unique_vals.map(_extract_attributes)

    enriched_texts = extracted.map(lambda x: x.enriched_text).astype("string").tolist()
    residual_texts = extracted.map(lambda x: x.residual_text).astype("string").tolist()
    attrs_list = extracted.map(lambda x: x.attrs).tolist()

    # Vectorize
    stop_words = None
    if extra_stopwords is not None:
        stop_words = list({str(w).lower() for w in extra_stopwords})

    vectorizer = TfidfVectorizer(
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words=stop_words,
        norm="l2",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(enriched_texts)  # sparse CSR

    # Cluster (fast)
    birch = Birch(threshold=birch_threshold, n_clusters=None)
    labels = birch.fit_predict(X)

    # Build mapping from original unique value -> cluster id
    mapping = pd.DataFrame({
        "value": unique_vals.values,
        "cluster_id": labels,
        "enriched": enriched_texts,
        "residual": residual_texts,
    })

    # Choose a representative label for each cluster:
    # pick the most frequent original value (by dataset frequency) as the "clean" label.
    freq = s.value_counts(dropna=True)
    mapping["freq"] = mapping["value"].map(freq).fillna(1).astype(int)

    # cluster representative = value with max frequency in that cluster
    rep = (
        mapping.sort_values(["cluster_id", "freq"], ascending=[True, False])
        .groupby("cluster_id", as_index=False)
        .first()[["cluster_id", "value"]]
        .rename(columns={"value": "cluster_rep"})
    )
    mapping = mapping.merge(rep, on="cluster_id", how="left")

    # Apply mapping back to full df
    value_to_rep = dict(zip(mapping["value"], mapping["cluster_rep"]))
    df_out = df.copy()
    df_out[new_column] = s.map(value_to_rep).astype("string")

    if keep_residual_text:
        value_to_residual = dict(zip(mapping["value"], mapping["residual"]))
        df_out[f"{new_column}__residual"] = s.map(value_to_residual).astype("string")

    if cluster_id_column:
        value_to_cluster = dict(zip(mapping["value"], mapping["cluster_id"]))
        df_out[cluster_id_column] = s.map(value_to_cluster).astype("Int64")

    # Dummy features (fast mapping from unique values)
    if add_dummy_features:
        pref = dummy_prefix if dummy_prefix is not None else f"{column}__"

        # Determine feature columns from union of keys (stable order)
        all_feat_keys = sorted({k for d in attrs_list for k in d.keys()})
        attrs_df = pd.DataFrame(attrs_list, columns=all_feat_keys).fillna(0).astype(np.int8)
        attrs_df.insert(0, "value", unique_vals.values) # type: ignore

        # map each feature via dict (avoid row-by-row apply on full df)
        for k in all_feat_keys:
            dmap = dict(zip(attrs_df["value"], attrs_df[k]))
            df_out[f"{pref}{k}"] = s.map(dmap).fillna(0).astype(np.int8)

    # Cluster summary for manual review
    # Weight by frequency to describe the dataset rather than only unique values
    mapping["n_rows"] = mapping["freq"]
    cluster_sizes = mapping.groupby("cluster_id")["n_rows"].sum().rename("n_rows").reset_index()

    # Examples: top 5 values by frequency
    top_examples = (
        mapping.sort_values(["cluster_id", "freq"], ascending=[True, False])
        .groupby("cluster_id")["value"]
        .apply(lambda x: list(x.head(5)))
        .rename("top_examples")
        .reset_index()
    )

    # Keywords: take top TF-IDF terms by average within cluster (cheap + interpretable).
    # We compute a simple centroid by mean TF-IDF over the rows in the cluster.
    cluster_keywords: List[Tuple[int, List[str]]] = []
    feature_names = np.array(vectorizer.get_feature_names_out())

    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            cluster_keywords.append((int(cid), []))
            continue

        centroid = X[idx].mean(axis=0)  # type: ignore 1 x n_features 
        arr = np.asarray(centroid).ravel()

        # take up to 10 non-zero top terms
        order = np.argsort(arr)[::-1]
        kw: List[str] = []
        for j in order:
            if arr[j] <= 0:
                break
            kw.append(str(feature_names[j]))
            if len(kw) >= 10:
                break

        cluster_keywords.append((int(cid), kw))

    kw_df = pd.DataFrame(cluster_keywords, columns=["cluster_id", "top_keywords"])

    cluster_summary = (
        cluster_sizes.merge(rep, on="cluster_id", how="left")
        .merge(top_examples, on="cluster_id", how="left")
        .merge(kw_df, on="cluster_id", how="left")
        .sort_values("n_rows", ascending=False)
        .reset_index(drop=True)
    )

    # If dummy features enabled, add prevalence per cluster (weighted by row frequency)
    if add_dummy_features:
        pref = dummy_prefix if dummy_prefix is not None else f"{column}__"
        # Rebuild attrs_df for unique values with frequency
        all_feat_keys = sorted({k for d in attrs_list for k in d.keys()})
        attrs_df = pd.DataFrame(attrs_list, columns=all_feat_keys).fillna(0).astype(np.int8)
        attrs_df["cluster_id"] = labels
        attrs_df["freq"] = mapping["freq"].values

        # Weighted mean prevalence
        prevalences = {}
        for k in all_feat_keys:
            wmean = (
                attrs_df.groupby("cluster_id")
                .apply(lambda g: float(np.average(g[k].astype(float), weights=g["freq"])))
                .rename(f"p_{k}")
            )
            prevalences[f"p_{k}"] = wmean
        prev_df = pd.DataFrame(prevalences).reset_index()

        cluster_summary = cluster_summary.merge(prev_df, on="cluster_id", how="left")

    return df_out, cluster_summary