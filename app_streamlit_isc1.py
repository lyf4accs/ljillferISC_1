import ast
import os
import re
from collections import Counter
from urllib.parse import quote

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="ISC1 - Letras y discografía", layout="wide")

DATA_FILE = "tracks_features_web.csv"

STUDIO_REFERENCE = pd.DataFrame(
    {
        "album_name": [
            "The Smiths",
            "Meat Is Murder",
            "The Queen Is Dead",
            "Strangeways, Here We Come",
        ],
        "release_year": [1984, 1985, 1986, 1987],
        "album_type": ["studio", "studio", "studio", "studio"],
    }
)

PREFERRED_ARTISTS = [
    "Beyoncé",
    "The Smiths",
    "Radiohead",
    "Michael Jackson",
    "Coldplay",
    "Ed Sheeran",
    "Bad Bunny",
    "Drake",
    "Nirvana",
    "Bruno Mars",
    "Kanye West",
    "Adele",
    "Eminem",
    "John Lennon",
    "Ariana Grande",
    "Queen",
    "Luis Fonsi",
    "Bon Jovi",
]

STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","so","of","to","in","on","for","from","with","without",
    "at","by","as","is","it","its","be","been","being","am","are","was","were","this","that","these","those",
    "i","me","my","mine","you","your","yours","he","him","his","she","her","hers","we","our","ours","they","them","their","theirs",
    "do","does","did","done","have","has","had","having","not","no","yes","all","any","some","can","could","will","would","shall","should",
    "just","too","very","into","out","up","down","over","under","again","once","here","there","when","where","why","how",
    "oh","ah","la","na","ooh","yeah","hey","ha","uh","mmm","im","ive","dont","didnt","cant","wont","youre","theyre","thats"
}


def parse_artists_list(artists_text):
    try:
        artists = ast.literal_eval(str(artists_text))
        if isinstance(artists, list):
            return [str(a).strip() for a in artists]
        return []
    except Exception:
        return []


def normalize_album_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["artists_list"] = df["artists"].apply(parse_artists_list)
    return df


@st.cache_data(show_spinner=False)
def build_artist_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.explode("artists_list")
        .rename(columns={"artists_list": "artist"})
        .dropna(subset=["artist"])
        .groupby("artist")
        .agg(num_songs=("name", "count"), num_albums=("album_id", "nunique"))
        .sort_values("num_songs", ascending=False)
        .reset_index()
    )


@st.cache_data(show_spinner=False)
def get_smiths_coverage(df: pd.DataFrame):
    mask = df["artists"].astype(str).str.contains(r"'The Smiths'|\"The Smiths\"", regex=True, na=False)
    smiths_df = df[mask].copy()
    studio_album_set = {name.lower() for name in STUDIO_REFERENCE["album_name"]}
    if not smiths_df.empty:
        smiths_df["album_clean"] = smiths_df["album"].apply(normalize_album_name)
        smiths_studio_df = smiths_df[smiths_df["album_clean"].isin(studio_album_set)].copy()
    else:
        smiths_studio_df = pd.DataFrame()
    return smiths_df, smiths_studio_df


@st.cache_data(show_spinner=False)
def get_artist_dataset(df: pd.DataFrame, selected_artist: str) -> pd.DataFrame:
    artist_data = df[df["artists_list"].apply(lambda lst: selected_artist in lst)].copy()
    artist_data["album_clean"] = artist_data["album"].apply(normalize_album_name)
    sort_cols = [c for c in ["year", "album_clean", "track_number", "name"] if c in artist_data.columns]
    artist_data = (
        artist_data.sort_values(sort_cols)
        .drop_duplicates(subset=["name", "album_clean"], keep="first")
        .reset_index(drop=True)
    )
    return artist_data


@st.cache_data(show_spinner=False)
def fetch_lyrics_simple(title: str, artist: str):
    title = str(title).strip().lower()
    artist = str(artist).strip().lower()
    title = re.sub(r"\(.*?\)", "", title)
    title = re.sub(r"\[.*?\]", "", title)
    title = title.split(" - ")[0].strip()
    try:
        url = f"https://api.lyrics.ovh/v1/{quote(artist)}/{quote(title)}"
        response = requests.get(url, timeout=6)
        if response.status_code == 200:
            data = response.json()
            lyrics = data.get("lyrics", "")
            if isinstance(lyrics, str) and lyrics.strip():
                return True, lyrics.strip()
    except Exception:
        pass
    return False, None


def tokenize_lyrics(text):
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t.strip("'") for t in tokens]
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def build_word_plot(top_words_df: pd.DataFrame, artist: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(top_words_df["word"], top_words_df["freq"])
    ax.set_title(f"Top 15 palabras más frecuentes — {artist}")
    ax.set_xlabel("Palabra")
    ax.set_ylabel("Frecuencia")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def build_length_plot(length_df: pd.DataFrame, artist: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(length_df["track_name"], length_df["num_words"])
    ax.set_title(f"Número de palabras por canción — {artist}")
    ax.set_xlabel("Número de palabras")
    ax.set_ylabel("Canción")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def main():
    st.title("ISC1 · Discografía y análisis de letras")

    st.markdown(
        """
        Esta aplicación traslada a web el notebook final del trabajo.
        El apartado 1 usa **The Smiths** como referencia de álbumes de estudio.
        El apartado 2 analiza además, letras de otros artistas que están presentes en el CSV. Como Github no permite subir archivos grandes, esta versión web solo incluye un subconjunto de los datos originales, principalmente artistas que se conoce que no son de música clásica o bandas sonoras, aseugrando así que sí tendrán letras. La cantidad de artistas y canciones disponibles es limitada. Sin embargo, se han seleccionado algunos artistas populares para asegurar una experiencia interactiva interesante. Si quieres analizar un artista específico, asegúrate de que esté presente en el CSV o considera ejecutar el notebook localmente con el dataset completo.
        """
    )

    if not os.path.exists(DATA_FILE):
        st.error(f"No se encuentra {DATA_FILE}. Colócalo en la misma carpeta que la app.")
        st.stop()

    df = load_data(DATA_FILE)
    artist_song_counts = build_artist_counts(df)
    smiths_df, smiths_studio_df = get_smiths_coverage(df)

    tab1, tab2 = st.tabs(["Apartado 1 · The Smiths", "Apartado 2 · Letras"])

    with tab1:
        st.subheader("Álbumes de estudio de referencia de The Smiths")
        st.dataframe(STUDIO_REFERENCE, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Filas de The Smiths en el CSV", int(len(smiths_df)))
        with c2:
            st.metric("Canciones de álbumes de estudio encontradas", int(len(smiths_studio_df)))

    with tab2:
        st.subheader("Exploración interactiva de letras")

        available_artists = [a for a in PREFERRED_ARTISTS if a in artist_song_counts["artist"].tolist()]
        if not available_artists:
            available_artists = artist_song_counts.head(20)["artist"].tolist()

        col1, col2 = st.columns([2, 1])
        with col1:
            selected_artist = st.selectbox("Artista", available_artists, index=0)
        with col2:
            max_songs = st.slider("Canciones a analizar", min_value=2, max_value=5, value=3)

        st.caption("Artistas disponibles en esta versión web")
        st.dataframe(
            artist_song_counts[artist_song_counts["artist"].isin(available_artists)].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Analizar letras", type="primary"):
            artist_data = get_artist_dataset(df, selected_artist)
            if artist_data.empty:
                st.warning("No hay canciones para este artista en el CSV.")
                st.stop()

            songs_df = artist_data[["name", "album"]].copy()
            songs_df = songs_df.rename(columns={"name": "track_name", "album": "album_name"})
            songs_df = songs_df.drop_duplicates(subset=["track_name"]).head(max_songs).reset_index(drop=True)

            lyrics_rows = []
            progress = st.progress(0, text="Buscando letras...")
            for idx, (_, row) in enumerate(songs_df.iterrows(), start=1):
                found, lyrics = fetch_lyrics_simple(row["track_name"], selected_artist)
                lyrics_rows.append(
                    {
                        "track_name": row["track_name"],
                        "album_name": row["album_name"],
                        "lyrics_found": found,
                        "lyrics": lyrics,
                    }
                )
                progress.progress(idx / len(songs_df), text=f"Buscando letras... {idx}/{len(songs_df)}")

            lyrics_df = pd.DataFrame(lyrics_rows)
            st.write(f"**Artista seleccionado:** {selected_artist}")
            st.write(f"**Canciones analizadas:** {len(songs_df)}")
            st.write(f"**Letras encontradas:** {int(lyrics_df['lyrics_found'].sum())}")
            st.write(f"**Letras no encontradas:** {int((~lyrics_df['lyrics_found']).sum())}")
            st.dataframe(lyrics_df[["track_name", "album_name", "lyrics_found"]], use_container_width=True, hide_index=True)

            lyrics_clean_df = lyrics_df[lyrics_df["lyrics_found"]].copy()
            if lyrics_clean_df.empty:
                st.warning("No se han encontrado letras para este artista con el número de canciones seleccionado.")
                st.stop()

            lyrics_clean_df["tokens"] = lyrics_clean_df["lyrics"].apply(tokenize_lyrics)
            lyrics_clean_df["num_words"] = lyrics_clean_df["tokens"].apply(len)

            all_tokens = [token for tokens in lyrics_clean_df["tokens"] for token in tokens]
            top_words_df = pd.DataFrame(Counter(all_tokens).most_common(15), columns=["word", "freq"])
            length_df = (
                lyrics_clean_df[["track_name", "album_name", "num_words"]]
                .sort_values("num_words", ascending=False)
                .reset_index(drop=True)
            )

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(build_word_plot(top_words_df, selected_artist))
            with c2:
                st.pyplot(build_length_plot(length_df, selected_artist))

    st.divider()

if __name__ == "__main__":
    main()
