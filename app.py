from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_recommender import build_content_item_sims, collaborative_scores
from utils import build_item_knn_model, build_ui_matrix, knn_item_scores


def _ensure_streamlit_import():
    try:
        import streamlit as st  # noqa: F401
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit==1.37.1"])
    import streamlit as st  # type: ignore
    return st


st = _ensure_streamlit_import()


def _should_bootstrap():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is None
    except Exception:
        return False


def _launch_streamlit():
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--global.developmentMode=false"]
    raise SystemExit(stcli.main())


if __name__ == "__main__" and _should_bootstrap():
    _launch_streamlit()


st.set_page_config(
    page_title="Movie Recommendation Studio",
    page_icon="Movie",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(59, 130, 246, 0.18), transparent 26%),
            radial-gradient(circle at 90% 8%, rgba(14, 165, 233, 0.16), transparent 24%),
            linear-gradient(180deg, #0b1120 0%, #111827 55%, #0f172a 100%);
        color: #f8fafc;
    }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div {
        color: #f8fafc;
    }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(3, 7, 18, 0.96), rgba(30, 41, 59, 0.96));
        color: white;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.24);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    .hero .eyebrow {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: rgba(255,255,255,0.68);
        margin-bottom: 0.6rem;
    }
    .hero h1 {
        margin: 0;
        font-size: clamp(2rem, 3.4vw, 3.2rem);
        line-height: 1.05;
        letter-spacing: -0.05em;
    }
    .hero p {
        margin: 0.85rem 0 0;
        max-width: 60ch;
        color: rgba(255,255,255,0.78);
        font-size: 1rem;
    }
    .stat-card {
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
        height: 100%;
    }
    .stat-label {
        color: rgba(248, 250, 252, 0.72);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.45rem;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .movie-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.94));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem;
        box-shadow: 0 10px 22px rgba(2, 6, 23, 0.24);
        margin-bottom: 0.85rem;
    }
    .movie-title {
        font-size: 1.12rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.2rem 0;
    }
    .movie-meta {
        color: rgba(226, 232, 240, 0.80);
        font-size: 0.9rem;
        margin: 0 0 0.65rem 0;
    }
    .score-pill {
        display: inline-block;
        padding: 0.28rem 0.68rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.16);
        color: #dbeafe;
        font-size: 0.84rem;
        font-weight: 800;
    }
    .section-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.15rem 0 0.75rem;
    }
    .soft-panel {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 22px;
        padding: 1rem 1rem 0.8rem;
        box-shadow: 0 14px 28px rgba(2, 6, 23, 0.22);
    }
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 22px;
        padding: 1rem 1rem 0.4rem;
        box-shadow: 0 14px 28px rgba(2, 6, 23, 0.22);
    }
    div[data-testid="stForm"] label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    div[data-testid="stForm"] input,
    div[data-testid="stForm"] [role="combobox"],
    div[data-testid="stForm"] textarea {
        background: #0f172a !important;
        color: #ffffff !important;
    }
    .stSidebar {
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
    }
    .stSidebar *, .stSidebar .stMarkdown, .stSidebar label, .stSidebar p, .stSidebar span, .stSidebar div {
        color: #e5e7eb !important;
    }
    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .sidebar-note {
        color: rgba(229, 231, 235, 0.78) !important;
        margin-bottom: 1rem;
        line-height: 1.45;
    }
    .recommend-hint {
        padding: 0.8rem 1rem;
        border-radius: 16px;
        background: rgba(59, 130, 246, 0.16);
        border: 1px solid rgba(96, 165, 250, 0.20);
        color: #eff6ff;
        margin-bottom: 0.9rem;
        font-weight: 600;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    ratings = pd.read_csv(ROOT / "data" / "ratings.csv")
    movies = pd.read_csv(ROOT / "data" / "movies.csv")
    return ratings, movies


@st.cache_resource
def build_models(ratings: pd.DataFrame, movies: pd.DataFrame, knn_neighbors: int, seed: int):
    n_users = int(ratings["user_id"].nunique())
    n_items = int(movies["movie_id"].nunique())
    ui = build_ui_matrix(ratings, n_users, n_items)
    content_sims, _ = build_content_item_sims(movies)
    n_comp = max(2, min(50, min(ui.shape) - 1))
    collab = collaborative_scores(ui, n_components=n_comp, seed=seed)
    knn_model, item_matrix = build_item_knn_model(ui, n_neighbors=knn_neighbors)
    return ui, content_sims, collab, knn_model, item_matrix


def score_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    content_sims,
    collab,
    knn_model,
    item_matrix,
    knn_neighbors: int,
    alpha: float,
    knn_weight: float,
    model_name: str,
):
    n_items = len(movies)
    user_ratings = ratings[ratings["user_id"] == user_id]
    seen = set((user_ratings["movie_id"].values - 1).tolist())
    liked = set((user_ratings.loc[user_ratings["rating"] >= 4, "movie_id"].values - 1).tolist())
    rating_map = {
        int(mid) - 1: float(rating)
        for mid, rating in zip(user_ratings["movie_id"].values, user_ratings["rating"].values)
    }

    user_idx = user_id - 1
    collab_row = collab[user_idx] if user_idx < collab.shape[0] else np.zeros(n_items)
    content_row = content_sims[list(liked)].mean(axis=0) if liked else None
    knn_row = knn_item_scores(seen, rating_map, knn_model, item_matrix, n_items, n_neighbors=knn_neighbors)

    if model_name == "Collaborative":
        scores = collab_row.copy()
    elif model_name == "Content-based":
        scores = content_row.copy() if content_row is not None else np.zeros(n_items)
    elif model_name == "KNN":
        scores = knn_row.copy()
    else:
        scores = alpha * collab_row
        if content_row is not None:
            scores += (1 - alpha) * content_row
        scores += knn_weight * knn_row

    scores = np.array(scores, dtype=float)
    scores[list(seen)] = -np.inf
    return scores, seen, liked, collab_row, content_row, knn_row


def top_genres_for_user(liked_ids: set[int], movies: pd.DataFrame) -> str:
    if not liked_ids:
        return "No rated favorites yet"
    genre_series = movies.iloc[list(liked_ids)]["genres"].dropna().astype(str)
    genres = (
        genre_series.str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(3)
        .index
        .tolist()
    )
    return ", ".join(genres) if genres else "No genre signal yet"


def recommendation_explanation(model_name: str, alpha: float, knn_weight: float) -> str:
    if model_name == "Collaborative":
        return "Bu mod, sana benzeyen kullanıcıların beğendiği filmlere bakarak öneri üretir."
    if model_name == "Content-based":
        return "Bu mod, seçili kullanıcının sevdiği türlere benzeyen filmleri önerir."
    if model_name == "KNN":
        return "Bu mod, izlediğin filmlere en benzer filmleri KNN ile bulur."
    return (
        f"Hibrit modda öneri skoru = collaborative ({alpha:.2f}) + content ({1 - alpha:.2f}) + "
        f"KNN ({knn_weight:.2f})."
    )


def signal_tags(model_name: str) -> list[str]:
    if model_name == "Collaborative":
        return ["Similar users", "Seen history", "Collaborative patterns"]
    if model_name == "Content-based":
        return ["Liked genres", "Genre overlap", "Movie metadata"]
    if model_name == "KNN":
        return ["Similar movies", "Nearest neighbors", "Item proximity"]
    return ["Hybrid score", "User taste", "Movie similarity"]


def render_movie_card(row: pd.Series, score: float):
    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-title">{row["title"]}</div>
            <div class="movie-meta">{row.get("genres", "Unknown genre")}</div>
            <span class="score-pill">Score {score:.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


ratings, movies = load_data()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Movie recommendation engine</div>
        <h1>Find the next film faster.</h1>
        <p>
            A cleaner interface for hybrid recommendations powered by content-based, collaborative,
            and KNN signals.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="soft-panel" style="margin-top: 0.9rem; margin-bottom: 1rem;">
        <div class="section-title">What this studio does</div>
        <p style="margin: 0; color: rgba(248,250,252,0.88); line-height: 1.6;">
            Pick a user, choose a recommendation model, and the app ranks unseen movies based on
            your selected signal: similar users, similar genres, nearest-neighbor movies, or a hybrid of all three.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "search_ready" not in st.session_state:
    st.session_state.search_ready = False
if "form_user_id" not in st.session_state:
    st.session_state.form_user_id = 1
if "form_model_name" not in st.session_state:
    st.session_state.form_model_name = "Hybrid"
if "form_top_n" not in st.session_state:
    st.session_state.form_top_n = 8
if "form_alpha" not in st.session_state:
    st.session_state.form_alpha = 0.6
if "form_knn_weight" not in st.session_state:
    st.session_state.form_knn_weight = 0.3
if "form_knn_neighbors" not in st.session_state:
    st.session_state.form_knn_neighbors = 20
if "form_seed" not in st.session_state:
    st.session_state.form_seed = 42


st.markdown(
    """
    <div class="soft-panel" style="margin-top: 1rem; margin-bottom: 1rem;">
        <div class="section-title">Search form</div>
        <p style="margin: 0; color: rgba(248,250,252,0.88); line-height: 1.6;">
            Fill the form, choose a model, then press <strong>Ara</strong>. The app will recommend unseen movies for the selected user.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("recommendation_form", clear_on_submit=False):
    left_form, right_form = st.columns([1.1, 0.9], gap="large")
    with left_form:
        st.markdown('<div class="section-title">User & model</div>', unsafe_allow_html=True)
        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=int(ratings["user_id"].max()),
            value=int(st.session_state.form_user_id),
            step=1,
        )
        model_name = st.selectbox(
            "Recommendation model",
            ["Hybrid", "Collaborative", "Content-based", "KNN"],
            index=["Hybrid", "Collaborative", "Content-based", "KNN"].index(st.session_state.form_model_name),
        )
        top_n = st.slider("How many recommendations?", 5, 20, int(st.session_state.form_top_n))
        seed = st.number_input("Seed", value=int(st.session_state.form_seed), step=1)

    with right_form:
        st.markdown('<div class="section-title">Model weights</div>', unsafe_allow_html=True)
        alpha = st.slider("Collaborative weight", 0.0, 1.0, float(st.session_state.form_alpha), 0.05)
        knn_weight = st.slider("KNN weight", 0.0, 1.0, float(st.session_state.form_knn_weight), 0.05)
        knn_neighbors = st.slider("KNN neighbors", 5, 50, int(st.session_state.form_knn_neighbors), 1)
        st.markdown(
            '<div class="recommend-hint" style="margin-top: 0.75rem;">Hybrid uses all three signals. The weights only matter when Hybrid is selected.</div>',
            unsafe_allow_html=True,
        )

    submitted = st.form_submit_button("Ara", use_container_width=True)

if submitted:
    st.session_state.search_ready = True
    st.session_state.form_user_id = int(user_id)
    st.session_state.form_model_name = model_name
    st.session_state.form_top_n = int(top_n)
    st.session_state.form_alpha = float(alpha)
    st.session_state.form_knn_weight = float(knn_weight)
    st.session_state.form_knn_neighbors = int(knn_neighbors)
    st.session_state.form_seed = int(seed)

if not st.session_state.search_ready:
    st.info("Formu doldurup **Ara** butonuna basınca öneriler burada görünecek.")
    st.stop()

user_id = int(st.session_state.form_user_id)
model_name = st.session_state.form_model_name
top_n = int(st.session_state.form_top_n)
alpha = float(st.session_state.form_alpha)
knn_weight = float(st.session_state.form_knn_weight)
knn_neighbors = int(st.session_state.form_knn_neighbors)
seed = int(st.session_state.form_seed)

ui, content_sims, collab, knn_model, item_matrix = build_models(
    ratings,
    movies,
    knn_neighbors=knn_neighbors,
    seed=seed,
)

scores, seen, liked, collab_row, content_row, knn_row = score_user(
    user_id=user_id,
    ratings=ratings,
    movies=movies,
    content_sims=content_sims,
    collab=collab,
    knn_model=knn_model,
    item_matrix=item_matrix,
    knn_neighbors=knn_neighbors,
    alpha=alpha,
    knn_weight=knn_weight,
    model_name=model_name,
)

top_idx = np.argpartition(scores, -top_n)[-top_n:]
top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
recommended = movies.iloc[top_idx].copy()
recommended["score"] = scores[top_idx]

avg_rating = float(ratings["rating"].mean())
top_movie_row = (
    ratings.groupby("movie_id")["rating"]
    .mean()
    .sort_values(ascending=False)
    .index[0]
)
top_movie_title = movies.loc[movies["movie_id"] == top_movie_row, "title"].iloc[0]
top_genres = top_genres_for_user(liked, movies)
why_text = recommendation_explanation(model_name, float(alpha), float(knn_weight))
tags = signal_tags(model_name)

st.markdown(
    f"""
    <div class="soft-panel" style="margin-top: 0.9rem; margin-bottom: 1rem;">
        <div class="section-title">How recommendations are chosen</div>
        <p style="margin: 0 0 0.7rem 0; color: rgba(248,250,252,0.88); line-height: 1.6;">{why_text}</p>
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
            {''.join(f'<span class="score-pill" style="background: rgba(59,130,246,0.16); color:#eff6ff;">{tag}</span>' for tag in tags)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

stat_a, stat_b, stat_c = st.columns(3)
with stat_a:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Total movies</div>
            <p class="stat-value">{len(movies):,}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with stat_b:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Average rating</div>
            <p class="stat-value">{avg_rating:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with stat_c:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Top rated title</div>
            <p class="stat-value">{top_movie_title}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_reco, tab_data = st.tabs(["Recommendations", "Explore data"])

with tab_reco:
    left, right = st.columns([1.45, 0.85], gap="large")

    with left:
        st.markdown(
            '<div class="recommend-hint">Film önerileri burada gösteriliyor. Her kart bir filmi ve öneri skorunu gösterir.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-title">Recommended for you</div>', unsafe_allow_html=True)
        for _, row in recommended.iterrows():
            render_movie_card(row, float(row["score"]))
        st.markdown(
            """
            <div class="soft-panel" style="margin-top: 1rem;">
                <div class="section-title">Why these titles?</div>
                <p style="margin: 0; color: rgba(248,250,252,0.88); line-height: 1.6;">
                    The app scores unseen movies using the selected model and removes anything the user already watched.
                    The highest-scoring movies are shown first.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Profile summary</div>', unsafe_allow_html=True)
        st.write(f"Selected user: **{user_id}**")
        st.write(f"Seen movies: **{len(seen)}**")
        st.write(f"Liked movies: **{len(liked)}**")
        st.write(f"Primary taste: **{top_genres}**")
        st.write(f"Model: **{model_name}**")
        st.write(f"KNN neighbors: **{knn_neighbors}**")
        st.write(f"Recommendation logic: **{why_text}**")
        st.caption("Seen movies are filtered out. The app ranks only unseen titles using the selected model.")
        st.progress(min(len(seen) / max(len(movies), 1), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="soft-panel" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Liked movies</div>', unsafe_allow_html=True)
        liked_titles = movies.loc[movies["movie_id"].isin([i + 1 for i in sorted(liked)])].head(8)
        if liked_titles.empty:
            st.info("No liked movies found for this user yet.")
        else:
            st.dataframe(liked_titles[["title", "genres"]], use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_data:
    data_left, data_right = st.columns([1, 1], gap="large")

    with data_left:
        st.markdown('<div class="section-title">Most popular movies</div>', unsafe_allow_html=True)
        popular = (
            ratings.groupby("movie_id")["rating"]
            .agg(["mean", "count"])
            .reset_index()
            .merge(movies, on="movie_id")
            .sort_values(["mean", "count"], ascending=False)
            .head(12)
        )
        st.dataframe(
            popular[["title", "genres", "mean", "count"]],
            use_container_width=True,
            hide_index=True,
        )

    with data_right:
        st.markdown('<div class="section-title">Rating distribution</div>', unsafe_allow_html=True)
        hist = np.histogram(ratings["rating"], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])[0]
        st.bar_chart(pd.DataFrame({"rating": [1, 2, 3, 4, 5], "count": hist}).set_index("rating"))

        st.markdown('<div class="section-title" style="margin-top: 1rem;">Genre counts</div>', unsafe_allow_html=True)
        genre_counts = (
            movies["genres"]
            .fillna("")
            .str.split(",")
            .explode()
            .str.strip()
            .value_counts()
            .head(10)
        )
        st.bar_chart(genre_counts)
