import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATASET_DIR = "Dataset"


def load_base_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies_path = os.path.join(DATASET_DIR, "movies.csv")
    ratings_path = os.path.join(DATASET_DIR, "ratings.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


def load_clean_features(path: str = os.path.join(DATASET_DIR, "clean_user_genre_features.csv")) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se ha encontrado el fichero de datos limpios en {path}. "
            f"Primero ejecuta eda.py para generarlo."
        )
    return pd.read_csv(path)


def scale_and_train_kmeans(
    n_clusters: int = 5, random_state: int = 42
) -> Tuple[KMeans, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Entrena K-Means sobre las características usuario-género.

    Devuelve:
    - modelo KMeans
    - X_scaled: matriz normalizada
    - user_ids
    - feature_names
    - scaler (para poder transformar nuevos usuarios)
    """
    df_features = load_clean_features()
    user_ids = df_features["userId"].values
    feature_names = np.array([c for c in df_features.columns if c != "userId"])
    X = df_features[feature_names].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(X_scaled)

    # Devolvemos también df_features para poder describir los clusters
    return kmeans, X_scaled, user_ids, feature_names, scaler, df_features


def prepare_movies_with_genres(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies_expanded = movies.assign(genres=movies["genres"].str.split("|")).explode(
        "genres"
    )
    return movies_expanded


def build_new_user_genre_vector(
    new_user_ratings: pd.DataFrame,
    movies_expanded: pd.DataFrame,
    genre_columns: List[str],
) -> np.ndarray:
    """
    Construye el vector de características (medias por género) para un usuario nuevo.

    new_user_ratings: DataFrame con columnas [movieId, rating].
    """
    df = new_user_ratings.merge(
        movies_expanded[["movieId", "genres"]], on="movieId", how="left"
    )
    user_genre_mean = df.groupby("genres")["rating"].mean()

    # Crear vector ordenado según genre_columns
    vec = np.zeros(len(genre_columns), dtype=float)
    genre_to_idx = {g: i for i, g in enumerate(genre_columns)}
    for genre, value in user_genre_mean.items():
        if genre in genre_to_idx:
            vec[genre_to_idx[genre]] = value
    return vec.reshape(1, -1)


def describe_kmeans_cluster(
    df_features: pd.DataFrame,
    kmeans: KMeans,
    cluster_id: int,
    feature_names: np.ndarray,
    top_n_genres: int = 3,
) -> str:
    """
    Devuelve una descripción textual sencilla del cluster de K-Means,
    indicando los géneros predominantes en ese grupo de usuarios.
    """
    labels = kmeans.labels_
    mask = labels == cluster_id
    n_users = mask.sum()
    if n_users == 0:
        return f"Cluster {cluster_id}: sin usuarios asignados."

    cluster_features = df_features.loc[mask, feature_names]
    mean_by_genre = cluster_features.mean().sort_values(ascending=False)
    top_genres = list(mean_by_genre.head(top_n_genres).index)

    return (
        f"Cluster {cluster_id}: {n_users} usuarios, con especial preferencia por "
        f"películas de géneros: {', '.join(top_genres)}."
    )


def recommend_for_existing_user(
    user_id: int,
    kmeans: KMeans,
    X_scaled: np.ndarray,
    user_ids: np.ndarray,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    min_ratings_per_movie: int = 10,
) -> pd.DataFrame:
    """
    Genera recomendaciones para un usuario existente usando el cluster de K-Means:
    - Se identifica el cluster del usuario.
    - Se buscan las películas mejor valoradas dentro de ese cluster.
    - Se devuelven las películas que el usuario aún no ha visto.
    """
    if user_id not in user_ids:
        raise ValueError(
            f"El usuario {user_id} no está en el dataset limpio. "
            f"Puedes probar con otro userId."
        )

    # Índice del usuario en X_scaled
    idx = np.where(user_ids == user_id)[0][0]
    user_cluster = kmeans.labels_[idx]

    # Usuarios del mismo cluster
    cluster_user_ids = user_ids[kmeans.labels_ == user_cluster]

    # Valoraciones de ese cluster
    cluster_ratings = ratings[ratings["userId"].isin(cluster_user_ids)].copy()

    # Películas que el usuario ya ha visto
    user_movies = ratings[ratings["userId"] == user_id]["movieId"].unique()
    cluster_ratings = cluster_ratings[~cluster_ratings["movieId"].isin(user_movies)]

    # Calcular nota media y número de valoraciones por película dentro del cluster
    movie_stats = (
        cluster_ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_rating", "count": "num_ratings"})
        .reset_index()
    )

    # Filtrar películas con pocas valoraciones (ruido)
    movie_stats = movie_stats[movie_stats["num_ratings"] >= min_ratings_per_movie]

    # Ordenar por nota media y número de valoraciones
    movie_stats = movie_stats.sort_values(
        by=["mean_rating", "num_ratings"], ascending=[False, False]
    )

    # Añadir títulos
    recs = movie_stats.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    return recs.head(top_n)


def recommend_for_new_user(
    new_user_ratings: pd.DataFrame,
    kmeans: KMeans,
    scaler: StandardScaler,
    feature_names: np.ndarray,
    X_scaled: np.ndarray,
    user_ids: np.ndarray,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    min_ratings_per_movie: int = 10,
) -> Tuple[int, pd.DataFrame]:
    """
    Recomendaciones para un usuario nuevo (sin userId en el dataset):
    - Se construye su vector de características usuario-género.
    - Se asigna al cluster más cercano (K-Means.predict).
    - Se recomiendan las películas populares dentro de ese cluster.

    new_user_ratings: DataFrame con columnas [movieId, rating].
    """
    movies_expanded = prepare_movies_with_genres(movies)

    genre_columns = list(feature_names)
    new_vec = build_new_user_genre_vector(new_user_ratings, movies_expanded, genre_columns)
    new_vec_scaled = scaler.transform(new_vec)

    # Asignar cluster
    user_cluster = int(kmeans.predict(new_vec_scaled)[0])

    # Usuarios del mismo cluster
    cluster_user_ids = user_ids[kmeans.labels_ == user_cluster]
    cluster_ratings = ratings[ratings["userId"].isin(cluster_user_ids)].copy()

    # El usuario nuevo aún no tiene películas en ratings, así que no hay que excluir nada

    movie_stats = (
        cluster_ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_rating", "count": "num_ratings"})
        .reset_index()
    )

    movie_stats = movie_stats[movie_stats["num_ratings"] >= min_ratings_per_movie]
    movie_stats = movie_stats.sort_values(
        by=["mean_rating", "num_ratings"], ascending=[False, False]
    )

    recs = movie_stats.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    return user_cluster, recs.head(top_n)


def main():
    print("Cargando datos base...")
    movies, ratings = load_base_data()

    print("Entrenando K-Means para recomendaciones...")
    kmeans, X_scaled, user_ids, feature_names, scaler, df_features = scale_and_train_kmeans(
        n_clusters=5
    )

    # Ejemplo 1: usuario existente
    # Elegimos un usuario aleatorio distinto en cada ejecución para ver ejemplos variados
    example_user = int(np.random.choice(user_ids))
    # Identificamos el cluster del usuario para poder describirlo
    idx_example = np.where(user_ids == example_user)[0][0]
    example_cluster = int(kmeans.labels_[idx_example])
    cluster_desc_existing = describe_kmeans_cluster(
        df_features=df_features,
        kmeans=kmeans,
        cluster_id=example_cluster,
        feature_names=feature_names,
    )

    print("\n" + "=" * 80)
    print(f"RECOMENDACIONES PARA USUARIO EXISTENTE (userId = {example_user})".center(80))
    print(cluster_desc_existing)
    recs_existing = recommend_for_existing_user(
        user_id=example_user,
        kmeans=kmeans,
        X_scaled=X_scaled,
        user_ids=user_ids,
        ratings=ratings,
        movies=movies,
        top_n=10,
    )
    # Mostrar solo el nombre de la película (y algo de info extra) para que la salida sea más clara
    if recs_existing.empty:
        print("No se han encontrado películas suficientes en este cluster para recomendar.")
    else:
        print("-" * 80)
        print(
            recs_existing[["title", "genres", "mean_rating", "num_ratings"]]
            .to_string(index=False)
        )

    # Ejemplo 2: usuario nuevo (simulado)
    # Supongamos que ha visto y valorado positivamente algunas películas concretas.
    # Elegimos películas distintas en cada ejecución (sin random_state fijo).
    print("\n" + "=" * 80)
    print("RECOMENDACIONES PARA USUARIO NUEVO (SIMULADO)".center(80))
    sample_movies = ratings["movieId"].drop_duplicates().sample(10)
    new_user_ratings = pd.DataFrame(
        {
            "movieId": sample_movies.values,
            "rating": np.random.uniform(3.5, 5.0, size=len(sample_movies)),
        }
    )

    new_cluster, recs_new = recommend_for_new_user(
        new_user_ratings=new_user_ratings,
        kmeans=kmeans,
        scaler=scaler,
        feature_names=feature_names,
        X_scaled=X_scaled,
        user_ids=user_ids,
        ratings=ratings,
        movies=movies,
        top_n=10,
    )

    cluster_desc_new = describe_kmeans_cluster(
        df_features=df_features,
        kmeans=kmeans,
        cluster_id=new_cluster,
        feature_names=feature_names,
    )

    print(f"Usuario nuevo asignado al cluster: {new_cluster}")
    print(cluster_desc_new)
    if recs_new.empty:
        print("No se han encontrado películas suficientes en este cluster para recomendar.")
    else:
        print("-" * 80)
        print(
            recs_new[["title", "genres", "mean_rating", "num_ratings"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()

