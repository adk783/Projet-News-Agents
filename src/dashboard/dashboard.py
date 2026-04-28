"""
dashboard.py

Dashboard Streamlit pour visualiser les résultats du pipeline POC 2.
Lit depuis la base SQLite produite par news_pipeline.py + agent_pipeline.py.

Usage :
    streamlit run dashboard.py
"""

import json
import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="News Pipeline — POC 2", page_icon="📈", layout="wide")

DB_PATH = "data/news_database.db"


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def load_articles() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL;")
        df = pd.read_sql(
            """
            SELECT
                ticker, sector, industry, title, date_utc,
                signal_filtrage, score_filtrage,
                signal_final, consensus_rate, impact_strength,
                argument_dominant, consensus_model, url
            FROM articles
            ORDER BY date_utc DESC
        """,
            conn,
        )
        conn.close()
        return df
    except Exception as e:
        st.error(f"Erreur de connexion à la base : {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_articles_filtres() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL;")
        df = pd.read_sql(
            "SELECT url, ticker, title, date_utc, motif, match_count FROM articles_filtres ORDER BY date_utc DESC", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_transcription(url: str) -> str:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute("SELECT transcription_debat FROM articles WHERE url = ?", (url,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else "Transcription non disponible."
    except Exception:
        return "Erreur lors du chargement."


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.title("📈 News Pipeline — POC 2")
st.caption("DistilRoBERTa filter → ABSA taxonomique → Débat multi-agent AutoGen → Consensus Agent")

df = load_articles()
df_filtres = load_articles_filtres()

df_traites = df[df["signal_final"].notna()] if not df.empty else pd.DataFrame()
df_en_attente = df[df["signal_final"].isna()] if not df.empty else pd.DataFrame()
df_signaux = (
    df_traites[df_traites["signal_final"].isin(["Achat", "Vente", "Neutre"])]
    if not df_traites.empty
    else pd.DataFrame()
)

# KPIs
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Collectés", len(df))
with c2:
    st.metric("Traités par agents", len(df_traites))
with c3:
    st.metric("En attente", len(df_en_attente))
with c4:
    achat = len(df_signaux[df_signaux["signal_final"] == "Achat"]) if not df_signaux.empty else 0
    st.metric("Achat", achat)
with c5:
    vente = len(df_signaux[df_signaux["signal_final"] == "Vente"]) if not df_signaux.empty else 0
    st.metric("Vente", vente)
with c6:
    if not df_signaux.empty and "impact_strength" in df_signaux.columns:
        st.metric("Impact Strength moyen", f"{df_signaux['impact_strength'].mean():.2f}")
    else:
        st.metric("Impact Strength moyen", "—")

st.divider()

# Filtres
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    tickers_dispo = ["Tous"] + sorted(df["ticker"].unique().tolist()) if not df.empty else ["Tous"]
    ticker_filtre = st.selectbox("Ticker", tickers_dispo)
with col_f2:
    signal_filtre = st.selectbox("Signal final", ["Tous", "Achat", "Vente", "Neutre", "En attente"])
with col_f3:
    distilroberta_filtre = st.selectbox("Signal DistilRoBERTa", ["Tous", "positive", "negative", "neutral"])

df_filtered = df.copy()
if ticker_filtre != "Tous":
    df_filtered = df_filtered[df_filtered["ticker"] == ticker_filtre]
if signal_filtre == "En attente":
    df_filtered = df_filtered[df_filtered["signal_final"].isna()]
elif signal_filtre != "Tous":
    df_filtered = df_filtered[df_filtered["signal_final"] == signal_filtre]
if distilroberta_filtre != "Tous":
    df_filtered = df_filtered[df_filtered["signal_filtrage"] == distilroberta_filtre]

# Tableau
st.subheader(f"Articles ({len(df_filtered)})")

if not df_filtered.empty:

    def color_signal(val):
        mapping = {
            "Achat": "background-color: #d4edda; color: #155724",
            "Vente": "background-color: #f8d7da; color: #721c24",
            "Neutre": "background-color: #fff3cd; color: #856404",
        }
        return mapping.get(val, "")

    display_cols = [
        c
        for c in [
            "ticker",
            "sector",
            "date_utc",
            "title",
            "signal_filtrage",
            "score_filtrage",
            "signal_final",
            "consensus_model",
            "consensus_rate",
            "impact_strength",
            "argument_dominant",
        ]
        if c in df_filtered.columns
    ]

    st.dataframe(
        df_filtered[display_cols].style.map(color_signal, subset=["signal_final"]), width="stretch", hide_index=True
    )
else:
    st.info("Aucun article ne correspond aux filtres.")

# Graphiques
st.divider()
col_g1, col_g2, col_g3 = st.columns(3)

with col_g1:
    st.subheader("Distribution des signaux")
    if not df_signaux.empty:
        st.bar_chart(df_signaux["signal_final"].value_counts())

with col_g2:
    st.subheader("Impact Strength par ticker")
    if not df_signaux.empty and "impact_strength" in df_signaux.columns:
        st.bar_chart(df_signaux.groupby("ticker")["impact_strength"].mean().sort_values(ascending=False))

with col_g3:
    st.subheader("Taux passage filtre DistilRoBERTa")
    if not df.empty:
        st.bar_chart(
            pd.Series(
                {
                    "Passés": df["signal_filtrage"].isin(["positive", "negative"]).sum(),
                    "Rejetés (neutre)": (df["signal_filtrage"] == "neutral").sum(),
                    "Non traités": df["signal_filtrage"].isna().sum(),
                }
            )
        )

# Transcription
st.divider()
st.subheader("🔍 Transcription du débat")
df_avec_debat = df_filtered[df_filtered["signal_final"].notna()] if not df_filtered.empty else pd.DataFrame()

if not df_avec_debat.empty:
    df_avec_debat = df_avec_debat.reset_index(drop=True)
    options = [f"{row['title']} ({str(row['date_utc'])[:10]})" for _, row in df_avec_debat.iterrows()]
    urls = df_avec_debat["url"].tolist()
    selection = st.selectbox("Sélectionne un article", range(len(options)), format_func=lambda i: options[i])
    if selection is not None:
        st.text_area("Transcription complète", load_transcription(urls[selection]), height=400)
else:
    st.info("Aucun article avec débat disponible.")

with st.expander(f"Articles filtrés hors-sujet par news_pipeline ({len(df_filtres)})"):
    if not df_filtres.empty:
        st.dataframe(
            df_filtres[["ticker", "title", "date_utc", "motif", "match_count"]], width="stretch", hide_index=True
        )
    else:
        st.info("Aucun article filtré.")

# ---------------------------------------------------------------------------
# Viewer ABSA
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔎 Aspects ABSA détectés")


def load_absa(url: str) -> str:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute("SELECT absa_json FROM articles WHERE url = ?", (url,))
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            data = json.loads(row[0])
            aspects = data.get("aspects", [])
            if not aspects:
                return "Aucun aspect économique détecté."
            lines = []
            for a in aspects:
                icon = {"positive": "▲", "negative": "▼", "neutral": "◆"}.get(a["sentiment"], "?")
                lines.append(f"• {icon} **{a['aspect'].upper()}** [{a['sentiment']}]")
                lines.append(f"  _Evidence_ : {a.get('evidence', '')}")
                lines.append(f"  _Reason_   : {a.get('reason', '')}")
                lines.append("")
            return "\n".join(lines)
        return "ABSA non disponible (article non encore traité)."
    except Exception as e:
        return f"Erreur : {e}"


if not df_filtered.empty and "signal_filtrage" in df_filtered.columns:
    df_avec_absa = df_filtered[df_filtered["signal_filtrage"].isin(["positive", "negative"])]
else:
    df_avec_absa = pd.DataFrame()

if not df_avec_absa.empty:
    df_avec_absa = df_avec_absa.reset_index(drop=True)
    options_absa = [f"{row['title']} ({str(row['date_utc'])[:10]})" for _, row in df_avec_absa.iterrows()]
    urls_absa = df_avec_absa["url"].tolist()
    selection_absa = st.selectbox(
        "Article (ABSA)", range(len(options_absa)), format_func=lambda i: options_absa[i], key="absa_sel"
    )
    if selection_absa is not None:
        st.markdown(load_absa(urls_absa[selection_absa]))
else:
    st.info("Aucun article avec ABSA disponible (les articles doivent avoir passé le filtre DistilRoBERTa).")

st.caption("Rafraîchissement toutes les 30s · POC 2 · DistilRoBERTa + ABSA + AutoGen multi-agent")
