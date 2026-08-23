"""United States Top 50 Playlist Performance and Song Popularity Trend Analysis.

Run with: streamlit run app.py
Expected input: Atlantic_United_States.csv in this directory.
"""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = Path(__file__).with_name("Atlantic_United_States.csv")
ROLLING_WINDOW = 3  # Assumption: a short 3-day window smooths daily API noise.
REQUIRED_COLUMNS = {
    "date", "position", "song", "artist", "popularity", "duration_ms",
    "album_type", "total_tracks", "is_explicit", "album_cover_url",
}

st.set_page_config(
    page_title="Atlantic Top 50 Analytics",
    page_icon="music_note",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stPlotlyChart"] { width: 100%; }
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.75rem 2rem; }
        [data-testid="stHeader"] { height: 2.5rem; }
        h1 { font-size: 1.75rem !important; line-height: 1.15 !important; }
        h2, h3 { font-size: 1.2rem !important; }
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.75rem !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        [data-testid="stDataFrame"] { overflow-x: auto; }
        [data-testid="stPlotlyChart"] { min-height: 280px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_bool(value: object) -> bool:
    """Convert common CSV boolean representations to a real bool."""
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def standardize_artist(value: object) -> str:
    """Trim artist text and make collaboration separators visually consistent."""
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s*(?:&|feat\.?|featuring)\s*", " & ", text, flags=re.I)
    return re.sub(r"(?:\s*&\s*)+", " & ", text).strip()


def split_artists(value: str) -> list[str]:
    """Split collaborations while retaining the standardized display string."""
    return [name.strip() for name in re.split(r"\s*&\s*", value) if name.strip()]


def validation_report(raw: pd.DataFrame) -> dict[str, object]:
    """Return data-quality checks without silently changing the uploaded data."""
    missing = raw.isna().sum()
    duplicate_count = int(raw.duplicated(subset=["song", "artist", "date"]).sum())
    positions = pd.to_numeric(raw["position"], errors="coerce")
    invalid_position_count = int((positions.isna() | ~positions.between(1, 50)).sum())
    return {
        "rows": len(raw),
        "missing": missing[missing.gt(0)].to_dict(),
        "duplicate_rows": duplicate_count,
        "invalid_position_rows": invalid_position_count,
    }


@st.cache_data(show_spinner="Loading and validating playlist data...")
def load_data(path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the CSV, parse types, validate, and engineer analysis features."""
    raw = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    report = validation_report(raw)
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"], format="%d-%m-%Y", errors="coerce")
    data["position"] = pd.to_numeric(data["position"], errors="coerce")
    data["popularity"] = pd.to_numeric(data["popularity"], errors="coerce")
    data["duration_ms"] = pd.to_numeric(data["duration_ms"], errors="coerce")
    data["total_tracks"] = pd.to_numeric(data["total_tracks"], errors="coerce")
    data["artist_original"] = data["artist"].astype("string")
    data["artist"] = data["artist"].map(standardize_artist)
    data["artist_list"] = data["artist"].map(lambda value: tuple(split_artists(value)))
    data["primary_artist"] = data["artist_list"].str[0].fillna("")
    data["album_type"] = data["album_type"].astype("string").str.strip().str.lower()
    data["is_explicit"] = data["is_explicit"].map(normalize_bool)
    data["duration_minutes"] = data["duration_ms"] / 60000
    data["position"] = data["position"].clip(lower=1, upper=50)
    data = data.dropna(subset=["date", "song", "position"]).sort_values(["song", "date"])

    # Per-song chart-run metrics: each distinct date counts once as a chart day.
    song_metrics = data.groupby("song", as_index=False).agg(
        days_on_chart=("date", "nunique"),
        average_rank=("position", "mean"),
        best_rank_achieved=("position", "min"),
        rank_volatility_index=("position", "std"),
        entry_date=("date", "min"),
        exit_date=("date", "max"),
        average_popularity=("popularity", "mean"),
    )
    dataset_end = data["date"].max()
    song_metrics["rank_volatility_index"] = song_metrics["rank_volatility_index"].fillna(0)
    song_metrics["still_charting"] = song_metrics["exit_date"].eq(dataset_end)
    data = data.merge(song_metrics, on="song", how="left")

    # A 3-day rolling mean, calculated within each song after chronological sorting.
    data["popularity_trend_score"] = (
        data.sort_values(["song", "date"])
        .groupby("song", group_keys=False)["popularity"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    data["date_label"] = data["date"].dt.strftime("%d-%b-%Y")
    return data, report


@st.cache_data(show_spinner=False)
def artist_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate artist breadth, chart-days, and a transparent dominance score.

    ADI = 100 * (0.70 * chart-day share + 0.30 * unique-song share).
    Chart-day share rewards sustained presence; song share rewards breadth while
    limiting the score of an artist represented by one unusually long run.
    Collaboration rows contribute to every listed artist's artist-level totals.
    """
    exploded = data[["date", "song", "artist_list"]].explode("artist_list")
    exploded = exploded.rename(columns={"artist_list": "artist"}).dropna(subset=["artist"])
    exploded["song_date"] = exploded["song"].astype("string") + "|" + exploded["date"].astype("string")
    metrics = exploded.groupby("artist", as_index=False).agg(
        unique_songs=("song", "nunique"),
        total_chart_days=("song_date", "nunique"),
    )
    total_days = metrics["total_chart_days"].sum()
    total_songs = metrics["unique_songs"].sum()
    metrics["artist_dominance_index"] = 100 * (
        0.70 * metrics["total_chart_days"] / total_days if total_days else 0
    ) + 100 * (
        0.30 * metrics["unique_songs"] / total_songs if total_songs else 0
    )
    return metrics.sort_values("artist_dominance_index", ascending=False)


def filter_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Render sidebar controls and return the filtered rows plus active settings."""
    st.sidebar.header("Filters")
    min_date, max_date = data["date"].min().date(), data["date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        date_range = (min_date, max_date)
    artists = sorted({artist for values in data["artist_list"] for artist in values})
    selected_artists = st.sidebar.multiselect("Artist", artists)
    song_search = st.sidebar.text_input("Song search")
    rank_range = st.sidebar.slider("Rank range", 1, 50, (1, 50))
    album_types = st.sidebar.multiselect("Album type", ["single", "album"], default=["single", "album"])
    explicit_options = st.sidebar.multiselect("Explicit content", ["Explicit", "Non-explicit"], default=["Explicit", "Non-explicit"])

    mask = data["date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    mask &= data["position"].between(*rank_range)
    if selected_artists:
        mask &= data["artist_list"].map(lambda values: bool(set(values) & set(selected_artists)))
    if song_search:
        mask &= data["song"].str.contains(song_search, case=False, na=False)
    if album_types:
        mask &= data["album_type"].isin(album_types)
    else:
        mask &= False
    explicit_values = {"Explicit": True, "Non-explicit": False}
    if explicit_options:
        mask &= data["is_explicit"].isin([explicit_values[value] for value in explicit_options])
    else:
        mask &= False
    settings = {"artists": selected_artists, "date_range": date_range, "rank_range": rank_range}
    return data.loc[mask].copy(), settings


def kpi_table(data: pd.DataFrame, artist_data: pd.DataFrame) -> pd.DataFrame:
    """Build the requested KPI summary for the currently filtered data."""
    if data.empty:
        return pd.DataFrame(columns=["Metric", "Value"])
    selected_songs = data["song"].nunique()
    selected_days = data["date"].nunique()
    dominance = artist_data["artist_dominance_index"].sum() if not artist_data.empty else 0
    return pd.DataFrame({
        "Metric": ["Days on Chart", "Average Rank", "Rank Volatility Index", "Popularity Score Trend", "Artist Dominance Index", "Explicit Content Share"],
        "Value": [
            f"{selected_days:,}",
            f"{data['position'].mean():.2f}",
            f"{data['rank_volatility_index'].mean():.2f}",
            f"{data['popularity_trend_score'].mean():.2f}",
            f"{dominance:.2f}",
            f"{data['is_explicit'].mean() * 100:.1f}%",
        ],
    })


def empty_state() -> None:
    st.info("No rows match the active filters. Widen the date, rank, artist, or content filters to continue.")


def show_chart(figure: go.Figure, height: int = 420) -> None:
    """Apply a compact, responsive Plotly layout consistently across the dashboard."""
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 48, "r": 18, "t": 62, "b": 48},
        legend={"orientation": "h", "y": -0.18, "x": 0},
        hoverlabel={"namelength": -1},
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": False},
    )


def ranking_analysis(data: pd.DataFrame) -> None:
    st.subheader("Playlist Ranking Analysis")
    daily = data.groupby("date", as_index=False).agg(avg_rank=("position", "mean"), median_rank=("position", "median"))
    fig = px.line(daily, x="date", y=["avg_rank", "median_rank"], markers=True, title="Daily rank distribution summary")
    fig.update_yaxes(autorange="reversed", title="Rank (1 is best)")
    show_chart(fig, 360)
    movement = data.sort_values(["song", "date"]).assign(rank_change=lambda frame: frame.groupby("song")["position"].diff())
    movement_summary = movement.groupby("song", as_index=False).agg(mean_rank_change=("rank_change", "mean"), observations=("rank_change", "count"))
    risers = movement_summary.sort_values("mean_rank_change").head(5).rename(columns={"mean_rank_change": "mean daily rank change"})
    decliners = movement_summary.sort_values("mean_rank_change", ascending=False).head(5).rename(columns={"mean_rank_change": "mean daily rank change"})
    left, right = st.columns(2)
    left.markdown("**Fast risers** (negative rank change is better)")
    left.dataframe(risers, use_container_width=True, hide_index=True)
    right.markdown("**Slow decliners**")
    right.dataframe(decliners, use_container_width=True, hide_index=True)


def song_analysis(data: pd.DataFrame) -> None:
    st.subheader("Song-Level Performance")
    songs = data.drop_duplicates("song").sort_values("days_on_chart", ascending=False)
    left, right = st.columns(2)
    left.markdown("**Longest-charting songs**")
    left.dataframe(songs[["song", "artist", "days_on_chart", "entry_date", "exit_date", "still_charting"]].head(10), use_container_width=True, hide_index=True)
    right.markdown("**Highest average popularity**")
    right.dataframe(songs.sort_values("average_popularity", ascending=False)[["song", "artist", "average_popularity", "best_rank_achieved"]].head(10), use_container_width=True, hide_index=True)
    fig = px.scatter(songs, x="days_on_chart", y="best_rank_achieved", size="average_popularity", color="still_charting", hover_name="song", title="Peak rank versus chart longevity", labels={"days_on_chart": "Days on chart", "best_rank_achieved": "Best rank", "average_popularity": "Avg popularity"})
    fig.update_yaxes(autorange="reversed", title="Best rank achieved")
    show_chart(fig)
    st.caption("Descriptive insight: songs in the upper-left are high-peak/short-run titles; songs toward the lower-right combine longevity with strong peak performance.")


def artist_analysis(data: pd.DataFrame) -> None:
    st.subheader("Artist Dominance Leaderboard")
    metrics = artist_metrics(data)
    if metrics.empty:
        empty_state()
        return
    fig = px.bar(metrics.head(15), x="artist_dominance_index", y="artist", orientation="h", color="total_chart_days", text="artist_dominance_index", title="Artist Dominance Index | Top 15", labels={"artist_dominance_index": "Dominance score", "total_chart_days": "Chart-days"})
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    show_chart(fig, 520)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    daily_artist = data[["date", "song", "artist_list"]].explode("artist_list").rename(columns={"artist_list": "leader_artist"}).groupby(["date", "leader_artist"], as_index=False).agg(chart_days=("song", "nunique"))
    daily_artist = daily_artist.rename(columns={"leader_artist": "artist"})
    leaders = daily_artist.sort_values(["date", "chart_days"], ascending=[True, False]).groupby("date").head(3)
    st.caption("Leaderboard over time: the table below shows the top three artists by songs present on each date.")
    st.dataframe(leaders, use_container_width=True, hide_index=True)


def popularity_analysis(data: pd.DataFrame) -> None:
    st.subheader("Popularity Analytics")
    correlation = data[["popularity", "position"]].corr().iloc[0, 1] if len(data) > 1 else np.nan
    st.metric("Popularity/rank correlation", "Not available" if pd.isna(correlation) else f"{correlation:.3f}", help="Negative values indicate higher popularity scores tend to align with better ranks.")
    bands = pd.cut(data["position"], [0, 10, 20, 50], labels=["Top 10", "11-20", "21-50"])
    band_data = data.assign(rank_band=bands).groupby("rank_band", observed=False, as_index=False).agg(avg_popularity=("popularity", "mean"), songs=("song", "nunique"))
    band_fig = px.bar(band_data, x="rank_band", y="avg_popularity", text="avg_popularity", title="Popularity by rank band", labels={"rank_band": "Playlist band", "avg_popularity": "Average popularity"})
    band_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    show_chart(band_fig, 340)
    stability_fig = px.scatter(data.drop_duplicates("song"), x="rank_volatility_index", y="average_popularity", size="days_on_chart", hover_name="song", title="Popularity stability versus rank volatility", labels={"rank_volatility_index": "Rank volatility (SD)", "average_popularity": "Average popularity", "days_on_chart": "Days on chart"})
    show_chart(stability_fig)


def content_analysis(data: pd.DataFrame) -> None:
    st.subheader("Content Attributes")
    comparison = data.groupby("is_explicit", as_index=False).agg(avg_rank=("position", "mean"), avg_popularity=("popularity", "mean"), avg_duration=("duration_minutes", "mean"), songs=("song", "nunique"))
    comparison["content"] = comparison["is_explicit"].map({True: "Explicit", False: "Non-explicit"})
    st.dataframe(comparison[["content", "songs", "avg_rank", "avg_popularity", "avg_duration"]], use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    left_fig = px.box(data, x="album_type", y="popularity", color="album_type", title="Popularity: single versus album", labels={"album_type": "Release type", "popularity": "Popularity score"})
    left_fig.update_layout(showlegend=False)
    left.plotly_chart(left_fig, use_container_width=True, config={"responsive": True, "displaylogo": False})
    duration_data = data.sample(min(5000, len(data)), random_state=42)
    right_fig = px.scatter(duration_data, x="duration_minutes", y="popularity", color="album_type", hover_name="song", title="Duration versus popularity", labels={"duration_minutes": "Duration (minutes)", "popularity": "Popularity score", "album_type": "Release type"})
    right.plotly_chart(right_fig, use_container_width=True, config={"responsive": True, "displaylogo": False})
    album_data = data[data["album_type"].eq("album")]
    album_fig = px.scatter(album_data, x="total_tracks", y="position", color="is_explicit", hover_name="song", title="Album size versus song rank", labels={"total_tracks": "Tracks on album", "position": "Playlist rank", "is_explicit": "Explicit"})
    album_fig.update_yaxes(autorange="reversed", range=[50.5, 0.5])
    show_chart(album_fig)


def timeline_tab(data: pd.DataFrame) -> None:
    st.subheader("Playlist Timeline Explorer")
    songs = sorted(data["song"].unique())
    selected = st.multiselect("Songs to plot", songs, default=songs[: min(5, len(songs))], key="timeline_songs")
    if not selected:
        st.info("Select at least one song to draw its rank trajectory.")
        return
    chart_data = data[data["song"].isin(selected)]
    fig = px.line(chart_data, x="date", y="position", color="song", markers=True, hover_data=["artist", "popularity", "popularity_trend_score"])
    fig.update_yaxes(autorange="reversed", range=[50.5, 0.5], title="Playlist rank")
    show_chart(fig, 420)
    covers = data[data["song"].isin(selected)][["song", "album_cover_url"]].drop_duplicates("song")
    covers = covers[covers["album_cover_url"].fillna("").ne("")].head(8)
    if not covers.empty:
        st.markdown("**Selected album covers**")
        cover_columns = st.columns(min(4, len(covers)))
        for index, (_, row) in enumerate(covers.iterrows()):
            with cover_columns[index % len(cover_columns)]:
                st.image(row["album_cover_url"], caption=row["song"], use_container_width=True)
    st.dataframe(chart_data.sort_values("date")[["song", "artist", "date", "position", "popularity", "album_cover_url"]], use_container_width=True, hide_index=True)


def main() -> None:
    st.title("United States Top 50 Playlist Performance")
    st.caption("Atlantic Recording Corporation | Historical, descriptive analysis only")
    if not DATA_PATH.exists():
        st.error(f"CSV not found at {DATA_PATH}. Place Atlantic_United_States.csv beside app.py and reload the app.")
        st.stop()
    try:
        data, report = load_data(str(DATA_PATH))
    except Exception as error:
        st.error(f"The dataset could not be loaded: {error}")
        st.stop()
    with st.sidebar.expander("Data quality report", expanded=False):
        st.write(f"Rows loaded: {report['rows']:,}")
        st.write(f"Duplicate song/artist/date rows: {report['duplicate_rows']:,}")
        st.write(f"Invalid position rows: {report['invalid_position_rows']:,}")
        if report["missing"]:
            st.write("Missing values:", report["missing"])
        else:
            st.write("Missing values: none detected")
    filtered, settings = filter_data(data)
    st.caption(f"Showing {len(filtered):,} of {len(data):,} observations | Rolling popularity window: {ROLLING_WINDOW} days")
    if filtered.empty:
        empty_state()
        return
    artists = artist_metrics(filtered)
    st.dataframe(kpi_table(filtered, artists), use_container_width=True, hide_index=True)
    view = st.radio(
        "Analysis view",
        ["Timeline Explorer", "Song Ranking Trends", "Artist Dominance", "Popularity vs Rank", "Content Attributes"],
        horizontal=True,
    )
    if view == "Timeline Explorer":
        timeline_tab(filtered)
    elif view == "Song Ranking Trends":
        ranking_analysis(filtered)
        song_analysis(filtered)
    elif view == "Artist Dominance":
        artist_analysis(filtered)
    elif view == "Popularity vs Rank":
        popularity_analysis(filtered)
    else:
        content_analysis(filtered)


if __name__ == "__main__":
    main()
