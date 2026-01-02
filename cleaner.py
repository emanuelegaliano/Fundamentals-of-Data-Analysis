# categorical_cleaner.py
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


def clean_categorical_column(
    df: pd.DataFrame,
    column: str,
    new_column: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    distance_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Clean and unify a categorical text column by clustering similar values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Name of the categorical column to clean.
    new_column : str, default "menus.cleanedNams"
        Name of the column where the cleaned values will be stored.
    model_name : str
        Sentence transformer model to use for embeddings.
    distance_threshold : float
        Maximum distance for clustering. Higher means fewer clusters.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with an additional column `new_column`.
    """

    # Work on a copy of the column as strings
    original_series = df[column].astype(str)

    # Get unique values to embed
    unique_values = original_series.unique().tolist()
    if len(unique_values) == 0:
        # nothing to do
        df[new_column] = original_series
        return df

    # 1) Encode unique values with a sentence-transformer model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(unique_values, show_progress_bar=True)

    # 2) Cluster them based on their embeddings
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="average",
        metric="cosine"
    )
    labels = clustering.fit_predict(embeddings)

    # 3) Build clusters: label -> list of values
    clusters = {}
    for val, lab in zip(unique_values, labels):
        clusters.setdefault(lab, []).append(val)

    # 4) Choose a canonical value per cluster.
    value_counts = original_series.value_counts()

    cluster_canonical = {}
    for lab, values in clusters.items():
        values_sorted = sorted(
            values,
            key=lambda v: (-value_counts.get(v, 0), len(v)),
        )
        canonical = values_sorted[0]
        for v in values:
            cluster_canonical[v] = canonical    

    df[new_column] = original_series.map(lambda x: cluster_canonical.get(x, x))
    return df

def exists_dir(path: str) -> bool:
    return os.path.exists(path)

def create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def dir_contains_files(path: str) -> bool:
    return any(os.scandir(path))