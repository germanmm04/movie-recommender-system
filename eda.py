import os

import numpy as np
import pandas as pd


def load_raw_data(dataset_dir: str = "Dataset") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los ficheros originales de MovieLens (o similar):
    - movies.csv
    - ratings.csv
    """
    movies_path = os.path.join(dataset_dir, "movies.csv")
    ratings_path = os.path.join(dataset_dir, "ratings.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    return movies, ratings


def clean_and_build_user_genre_features(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    min_ratings_per_user: int = 20,
) -> pd.DataFrame:
    """
    EDA + limpieza básica + extracción de características:

    - Elimina valores nulos básicos.
    - Explota los géneros de las películas.
    - Calcula la calificación media por usuario y género.
    - Devuelve un dataframe donde:
        * Cada fila es un usuario.
        * Cada columna (excepto userId) es un género.
        * El valor es la nota media que da ese usuario a ese género.
    """
    # Eliminar valores nulos obvios
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    movies = movies.dropna(subset=["movieId", "title", "genres"])

    # Asegurar tipos
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    # Filtrar usuarios con pocas valoraciones (ruido)
    user_counts = ratings["userId"].value_counts()
    active_users = user_counts[user_counts >= min_ratings_per_user].index
    ratings = ratings[ratings["userId"].isin(active_users)].copy()

    # Separar géneros (están separados por "|")
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies_expanded = movies.assign(genres=movies["genres"].str.split("|")).explode(
        "genres"
    )

    # Unir ratings con géneros
    ratings_genres = ratings.merge(
        movies_expanded[["movieId", "genres"]], on="movieId", how="left"
    )

    # Calcular nota media por usuario y género
    user_genre_mean = (
        ratings_genres.groupby(["userId", "genres"])["rating"].mean().reset_index()
    )

    # Pivotar: filas = usuarios, columnas = géneros
    user_genre_pivot = user_genre_mean.pivot_table(
        index="userId", columns="genres", values="rating", fill_value=0.0
    )

    # Resetear índice para tener userId como columna
    user_genre_pivot = user_genre_pivot.reset_index()

    # Ordenar columnas para consistencia (userId primero)
    genre_cols = sorted([c for c in user_genre_pivot.columns if c != "userId"])
    user_genre_pivot = user_genre_pivot[["userId"] + genre_cols]

    return user_genre_pivot


def main():
    print("Cargando datos crudos...")
    movies, ratings = load_raw_data()

    print("Construyendo matriz de características usuario-género...")
    user_features = clean_and_build_user_genre_features(movies, ratings)

    output_path = os.path.join("Dataset", "clean_user_genre_features.csv")
    user_features.to_csv(output_path, index=False)

    print(f"Dataset limpio guardado en: {output_path}")
    print(f"Número de usuarios: {user_features.shape[0]}")
    print(f"Número de géneros (características): {user_features.shape[1] - 1}")


if __name__ == "__main__":
    main()

