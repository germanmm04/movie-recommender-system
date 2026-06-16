import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_clean_features(path: str = os.path.join("Dataset", "clean_user_genre_features.csv")) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se ha encontrado el fichero de datos limpios en {path}. "
            f"Primero ejecuta eda.py para generarlo."
        )
    return pd.read_csv(path)


def scale_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve:
    - X_scaled: matriz normalizada
    - user_ids: vector con los userId
    - feature_names: nombres de las columnas de características
    """
    user_ids = df["userId"].values
    feature_names = [c for c in df.columns if c != "userId"]
    X = df[feature_names].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, user_ids, np.array(feature_names)


def run_kmeans(X_scaled: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(X_scaled)
    return kmeans


def run_dbscan(
    X_scaled: np.ndarray,
    eps: float = 0.8,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> DBSCAN:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(X_scaled)
    return dbscan


def describe_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    method_name: str,
    feature_names: List[str],
    top_n_genres: int = 3,
) -> None:
    """
    Imprime una descripción textual de cada cluster basada en las medias
    de las características (géneros). Sirve para explicar "de qué va" cada grupo.

    - df: dataframe con columnas [userId] + géneros.
    - labels: etiquetas de cluster (un entero por usuario).
    - method_name: nombre del algoritmo (K-Means, DBSCAN, DPC).
    - feature_names: lista de columnas de características (géneros).
    """
    print(f"\nDescripción de clusters para {method_name}:")

    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        # En DBSCAN/DPC el -1 suele ser ruido: lo ignoramos en la descripción
        if lab == -1:
            continue

        mask = labels == lab
        n_users = mask.sum()
        if n_users == 0:
            continue

        cluster_features = df.loc[mask, feature_names]
        mean_by_genre = cluster_features.mean().sort_values(ascending=False)
        top_genres = list(mean_by_genre.head(top_n_genres).index)

        print(
            f"- {method_name} - Cluster {lab}: {n_users} usuarios, "
            f"géneros principales: {', '.join(top_genres)}"
        )


def compute_dpc(
    X_scaled: np.ndarray,
    dc_percentile: float = 2.0,
    max_points: int = 2000,
) -> np.ndarray:
    """
    Implementación sencilla de Density Peaks Clustering (DPC) basada en:
    Rodriguez & Laio (Science, 2014).

    NOTA: Complejidad O(n^2). Para evitar problemas con muchos usuarios
    limitamos el número de puntos a `max_points`.

    Devuelve un vector de etiquetas de clusters para los puntos usados.
    """
    n = X_scaled.shape[0]
    if n > max_points:
        print(
            f"Advertencia: hay {n} usuarios, se usará solo una muestra de {max_points} "
            f"para DPC por cuestiones de tiempo/memoria."
        )
        idx = np.random.RandomState(42).choice(n, size=max_points, replace=False)
        X = X_scaled[idx]
        index_map = idx
    else:
        X = X_scaled
        index_map = np.arange(n)

    m = X.shape[0]

    # Matriz de distancias (euclídea)
    from sklearn.metrics import pairwise_distances

    print("Calculando matriz de distancias para DPC (puede tardar un poco)...")
    dist = pairwise_distances(X, metric="euclidean")

    # Distancia de corte dc como percentil de todas las distancias
    tri = dist[np.triu_indices(m, k=1)]
    dc = np.percentile(tri, dc_percentile)
    print(f"dc (percentil {dc_percentile}) = {dc:.4f}")

    # 1) Densidad local rho_i = número de puntos a una distancia < dc
    rho = np.sum(dist < dc, axis=1) - 1  # restamos 1 para no contarse a sí mismo

    # 2) Delta_i = distancia al punto más cercano con densidad mayor
    delta = np.zeros(m)
    nearest_higher = np.full(m, -1, dtype=int)

    # Ordenar puntos de mayor a menor densidad
    sort_idx = np.argsort(-rho)
    max_dist = dist.max()

    # El punto con mayor densidad se define con delta máxima
    delta[sort_idx[0]] = max_dist
    nearest_higher[sort_idx[0]] = -1

    for i in range(1, m):
        idx_i = sort_idx[i]
        # Candidatos con densidad mayor
        higher = sort_idx[:i]
        dists_to_higher = dist[idx_i, higher]
        j = np.argmin(dists_to_higher)
        delta[idx_i] = dists_to_higher[j]
        nearest_higher[idx_i] = higher[j]

    # Selección simple de centros: puntos con rho y delta altos
    rho_norm = (rho - rho.min()) / (rho.max() - rho.min() + 1e-9)
    delta_norm = (delta - delta.min()) / (delta.max() - delta.min() + 1e-9)
    gamma = rho_norm * delta_norm

    # Número de clusters aproximado: 3 (puedes ajustar este valor)
    n_centers = 3
    center_indices = np.argsort(-gamma)[:n_centers]
    print(f"Centros DPC seleccionados (índices locales): {center_indices}")

    labels_local = np.full(m, -1, dtype=int)
    for c_idx, center in enumerate(center_indices):
        labels_local[center] = c_idx

    # Propagar etiquetas desde centros a puntos de menor densidad
    for i in range(m):
        idx_i = sort_idx[i]
        if labels_local[idx_i] == -1:
            labels_local[idx_i] = labels_local[nearest_higher[idx_i]]

    # Reconstruir vector de etiquetas al tamaño original (n), -1 si no se usó
    labels_global = np.full(n, -1, dtype=int)
    labels_global[index_map] = labels_local
    return labels_global


def plot_clusters_2d(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    title: str,
    max_points: int = 3000,
):
    """
    Reduce dimensionalidad con PCA a 2D y pinta un scatter coloreado por cluster.
    """
    n = X_scaled.shape[0]
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, size=max_points, replace=False)
        X_plot = X_scaled[idx]
        labels_plot = labels[idx]
    else:
        X_plot = X_scaled
        labels_plot = labels

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_plot)

    unique_labels = np.unique(labels_plot)
    # Asignar un color por cluster, ruido (-1) en gris
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(8, 6))
    for lab in unique_labels:
        mask = labels_plot == lab
        color = "lightgray" if lab == -1 else cmap(int(lab) % 10)
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=10,
            c=color,
            label=f"Cluster {lab}",
            alpha=0.7,
        )

    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def main():
    print("Cargando dataset limpio...")
    df = load_clean_features()
    X_scaled, user_ids, feature_names = scale_features(df)

    # --- K-MEANS ---
    print("Entrenando K-Means...")
    kmeans = run_kmeans(X_scaled, n_clusters=5)
    labels_kmeans = kmeans.labels_
    print(f"K-Means: número de clusters = {len(np.unique(labels_kmeans))}")
    describe_clusters(
        df=df,
        labels=labels_kmeans,
        method_name="K-Means",
        feature_names=list(feature_names),
    )

    # --- DBSCAN ---
    print("Entrenando DBSCAN...")
    dbscan = run_dbscan(X_scaled, eps=0.8, min_samples=5)
    labels_dbscan = dbscan.labels_
    print(
        f"DBSCAN: clusters = {len(set(labels_dbscan) - {-1})}, ruido = {(labels_dbscan == -1).sum()}"
    )

    # --- DPC ---
    print("Ejecutando Density Peaks Clustering (DPC)...")
    labels_dpc = compute_dpc(X_scaled)
    print(f"DPC: clusters (excluyendo -1) = {len(set(labels_dpc) - {-1})}")

    # Descripción de clusters para DBSCAN y DPC
    describe_clusters(
        df=df,
        labels=labels_dbscan,
        method_name="DBSCAN",
        feature_names=list(feature_names),
    )
    describe_clusters(
        df=df,
        labels=labels_dpc,
        method_name="DPC",
        feature_names=list(feature_names),
    )

    # Guardar asignaciones de cluster para recomendaciones posteriores
    clusters_df = pd.DataFrame(
        {
            "userId": user_ids,
            "cluster_kmeans": labels_kmeans,
            "cluster_dbscan": labels_dbscan,
            "cluster_dpc": labels_dpc,
        }
    )
    output_path = os.path.join("Dataset", "user_clusters.csv")
    clusters_df.to_csv(output_path, index=False)
    print(f"Asignaciones de clusters guardadas en: {output_path}")

    # Visualizaciones 2D
    print("Mostrando visualizaciones de clusters (PCA 2D)...")
    plot_clusters_2d(X_scaled, labels_kmeans, "Clusters de usuarios - K-Means")
    plot_clusters_2d(X_scaled, labels_dbscan, "Clusters de usuarios - DBSCAN")
    plot_clusters_2d(X_scaled, labels_dpc, "Clusters de usuarios - DPC")


if __name__ == "__main__":
    main()

