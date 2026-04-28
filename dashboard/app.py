"""
Locus - Dashboard de supervision

Lancement :
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Locus - Supervision",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

DB = Path("data/news_database.db")
TRADES = Path("logs/dry_run_trades.jsonl")
COSTS = Path("reports/llm_cost_daily")
LOG = Path("logs/pipeline.log")
CALIB = Path("models/calibrator.meta.json")
PROFILE = Path("data/investor_profile.json")
PORTFOLIO = Path("data/portfolio_state.json")

# Evaluations scientifiques (walk-forward, frais, regimes, alpha)
EVAL_DIR        = Path("eval/eval_results")
WALK_FORWARD    = EVAL_DIR / "walk_forward.json"
EXEC_COSTS_GLOB = "*_execution_costs_layer10"
REGIMES_GLOB    = "*_market_regimes_layer12"
MARKET_GLOB     = "*_market_layer3"
LATENCY_GLOB    = "*_latency_layer11"

OBJECTIVE_LABELS = {
    "preservation_capital": "Preservation du capital",
    "revenu":               "Generation de revenu",
    "croissance_capital":   "Croissance du capital",
    "speculation":          "Speculation",
}
HORIZON_LABELS = {
    "intraday":     "Intra-journalier",
    "court_terme":  "Court terme (jusqu'a 3 mois)",
    "moyen_terme":  "Moyen terme (3 mois a 1 an)",
    "long_terme":   "Long terme (> 1 an)",
}
RISK_LABELS = {
    "conservateur": "Conservateur",
    "modere":       "Modere",
    "agressif":     "Agressif",
}

PALETTE = {
    "bg": "#0b101a",
    "panel": "#131a26",
    "panel_alt": "#1a2230",
    "border": "#222d3d",
    "text": "#e6ebf3",
    "text_soft": "#aab1c0",
    "muted": "#6b7585",
    "accent": "#3b82f6",
    "buy": "#10b981",
    "sell": "#ef4444",
    "neutral": "#9ca3af",
    "warn": "#f59e0b",
    "info": "#60a5fa",
}

DECISION_LABELS = {
    "Achat": "Acheter",
    "Vente": "Vendre",
    "Neutre": "Attendre",
    "HOLD_SYSTEMIC": "Bloque (crise)",
    "HOLD_SECTOR_CAP": "Bloque (limite secteur)",
    "Erreur": "Erreur d'analyse",
    "Rejete (Filtre)": "Rejete (filtre)",
    "Rejeté (Filtre)": "Rejete (filtre)",
}

REJECTED_VALUES = {"Erreur", "Rejete (Filtre)", "Rejeté (Filtre)"}

REGIME_LABELS = {
    "BULL": "Hausse",
    "BEAR": "Baisse",
    "SIDEWAYS": "Stable",
    "HIGH_VOL": "Tres volatile",
    "UNKNOWN": "Inconnu",
}

DECISION_COLOR = {
    "Acheter": PALETTE["buy"],
    "Vendre": PALETTE["sell"],
    "Attendre": PALETTE["neutral"],
    "Bloque (crise)": PALETTE["warn"],
    "Bloque (limite secteur)": PALETTE["warn"],
    "Erreur d'analyse": PALETTE["muted"],
    "Rejete (filtre)": PALETTE["muted"],
}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: {PALETTE["bg"]};
}}
[data-testid="stHeader"] {{
    background: transparent;
}}
[data-testid="stSidebar"] {{
    background: {PALETTE["panel"]};
    border-right: 1px solid {PALETTE["border"]};
}}
[data-testid="stSidebar"] * {{
    color: {PALETTE["text"]};
}}
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {PALETTE["text"]};
}}
h1 {{
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    color: {PALETTE["text"]} !important;
    margin-bottom: 0.15rem !important;
    letter-spacing: -0.01em;
}}
h2, h3, h4 {{
    color: {PALETTE["text"]} !important;
    font-weight: 600 !important;
    letter-spacing: -0.005em;
}}
h2 {{ font-size: 1.05rem !important; }}
h3 {{ font-size: 0.95rem !important; }}
p, li, span, label {{
    color: {PALETTE["text_soft"]};
}}
.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px;
}}
[data-testid="metric-container"] {{
    background: {PALETTE["panel"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 10px;
    padding: 0.95rem 1.1rem;
}}
[data-testid="metric-container"] label {{
    color: {PALETTE["muted"]} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500 !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {PALETTE["text"]} !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.75rem !important;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {PALETTE["border"]};
    border-radius: 8px;
    overflow: hidden;
}}
[data-testid="stExpander"] {{
    background: {PALETTE["panel"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 8px;
}}
hr {{
    border-color: {PALETTE["border"]} !important;
    margin: 1.5rem 0 1.2rem 0 !important;
    opacity: 0.6;
}}
.stProgress > div > div > div {{
    background: {PALETTE["accent"]};
}}
.stAlert {{
    border-radius: 8px;
    border-width: 1px !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 1.5rem;
    border-bottom: 1px solid {PALETTE["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    color: {PALETTE["muted"]};
    padding: 0.5rem 0.25rem !important;
    font-size: 0.88rem;
}}
.stTabs [aria-selected="true"] {{
    color: {PALETTE["text"]} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    background: {PALETTE["accent"]} !important;
}}
.locus-eyebrow {{
    font-size: 0.7rem;
    color: {PALETTE["muted"]};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-bottom: 0.25rem;
}}
.locus-subtitle {{
    color: {PALETTE["text_soft"]};
    font-size: 0.92rem;
    margin-top: 0.1rem;
    margin-bottom: 0.4rem;
}}
.locus-section-title {{
    color: {PALETTE["text"]};
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0.5rem 0 0.7rem 0;
    letter-spacing: -0.005em;
}}
.locus-pill {{
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    border: 1px solid;
}}
.locus-pill.ok    {{ background: rgba(16, 185, 129, 0.10); color: #34d399; border-color: rgba(16,185,129,0.35); }}
.locus-pill.warn  {{ background: rgba(245, 158, 11, 0.10); color: #fbbf24; border-color: rgba(245,158,11,0.35); }}
.locus-pill.err   {{ background: rgba(239, 68, 68, 0.10); color: #f87171; border-color: rgba(239,68,68,0.35); }}
.locus-pill.info  {{ background: rgba(59, 130, 246, 0.10); color: #60a5fa; border-color: rgba(59,130,246,0.35); }}
.locus-pill.muted {{ background: rgba(107, 117, 133, 0.10); color: #9ca3af; border-color: rgba(107,117,133,0.30); }}
.locus-card {{
    background: {PALETTE["panel"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 10px;
    padding: 1rem 1.15rem;
}}
.locus-help {{
    font-size: 0.78rem;
    color: {PALETTE["muted"]};
    line-height: 1.45;
    margin-top: 0.4rem;
}}
[data-testid="stSidebar"] .stRadio label {{
    color: {PALETTE["text_soft"]} !important;
    font-size: 0.92rem !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    color: {PALETTE["text"]} !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except sqlite3.Error:
        return []


@st.cache_data(ttl=20)
def load_articles() -> pd.DataFrame:
    if not DB.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB), timeout=10)
        cols = _table_columns(conn, "articles")
        wanted = [
            "ticker", "title", "date_utc", "signal_filtrage", "score_filtrage",
            "signal_final", "consensus_rate", "impact_strength",
            "argument_dominant", "consensus_model", "risk_level",
            "sector", "industry", "source", "url",
            "market_regime", "vix_at_decision", "dry_run", "kill_switch_active",
        ]
        select = [c for c in wanted if c in cols]
        if not select:
            conn.close()
            return pd.DataFrame()
        df = pd.read_sql(
            f"SELECT {', '.join(select)} FROM articles ORDER BY date_utc DESC LIMIT 1000",
            conn,
        )
        conn.close()
        return df
    except sqlite3.Error:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_trades() -> pd.DataFrame:
    if not TRADES.exists():
        return pd.DataFrame()
    try:
        rows = [
            json.loads(line)
            for line in TRADES.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_llm_cost() -> dict | None:
    if not COSTS.exists():
        return None
    files = sorted(COSTS.glob("*.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_calibration() -> dict | None:
    if not CALIB.exists():
        return None
    try:
        return json.loads(CALIB.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_profile() -> dict | None:
    if not PROFILE.exists():
        return None
    try:
        return json.loads(PROFILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_portfolio() -> dict | None:
    if not PORTFOLIO.exists():
        return None
    try:
        return json.loads(PORTFOLIO.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_eval_dir(pattern: str) -> Path | None:
    """Renvoie le repertoire d'eval le plus recent matchant le pattern (ordonne par nom = par timestamp)."""
    if not EVAL_DIR.exists():
        return None
    candidates = sorted(EVAL_DIR.glob(pattern))
    return candidates[-1] if candidates else None


def load_walk_forward() -> dict | None:
    """Backtest walk-forward (Bailey-Lopez 2014) : Sharpe par fenetre, hit rate, stability."""
    if not WALK_FORWARD.exists():
        return None
    try:
        return json.loads(WALK_FORWARD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_execution_costs() -> dict | None:
    """Couts d'execution par tier de courtier (Almgren-Chriss 2001)."""
    d = _latest_eval_dir(EXEC_COSTS_GLOB)
    if d is None:
        return None
    files = list(d.glob("*.json"))
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_market_regimes() -> dict | None:
    """Performance par regime de marche (BULL/BEAR/SIDEWAYS/HIGH_VOL)."""
    d = _latest_eval_dir(REGIMES_GLOB)
    if d is None:
        return None
    files = list(d.glob("*.json"))
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_market_trades() -> pd.DataFrame:
    """Trades evalues vs benchmark SPY (alpha, return, signal_correct)."""
    d = _latest_eval_dir(MARKET_GLOB)
    if d is None:
        return pd.DataFrame()
    files = list(d.glob("trades.json"))
    if not files:
        return pd.DataFrame()
    try:
        rows = json.loads(files[0].read_text(encoding="utf-8"))
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()


def portfolio_summary(portfolio: dict | None, profile: dict | None) -> dict:
    """
    Calcule un resume du portefeuille en parite avec PortfolioState (src/strategy/portfolio_state.py).
    Renvoie : capital_initial, cash, valeur_positions, valeur_totale, pnl_eur, pnl_pct,
              drawdown_actuel, drawdown_vs_peak, n_positions, max_positions,
              cash_reserve_eur, cash_investissable, expo_sectorielle.
    """
    if portfolio is None:
        return {}
    cap_init = float(portfolio.get("capital_initial", 0.0))
    cash = float(portfolio.get("cash_disponible", 0.0))
    peak = float(portfolio.get("peak_valeur", cap_init or 0.0))
    positions = portfolio.get("positions", {}) or {}

    val_positions = 0.0
    expo_sect: dict[str, float] = {}
    for _, p in positions.items():
        nb = float(p.get("nb_actions", 0.0))
        prix = float(p.get("prix_actuel") or p.get("prix_entree") or 0.0)
        v = nb * prix
        val_positions += v
        sect = p.get("secteur") or "Inconnu"
        expo_sect[sect] = expo_sect.get(sect, 0.0) + v

    val_totale = val_positions + cash
    pnl_eur = val_totale - cap_init
    pnl_pct = (pnl_eur / cap_init) if cap_init > 0 else 0.0
    dd_init = (val_totale - cap_init) / cap_init if cap_init > 0 else 0.0
    dd_peak = (val_totale - peak) / peak if peak > 0 else 0.0

    cash_reserve_pct = float((profile or {}).get("cash_reserve_min", 0.0))
    cash_reserve_eur = cap_init * cash_reserve_pct
    cash_investissable = max(0.0, cash - cash_reserve_eur)

    expo_sect_pct = {
        s: (v / val_totale) for s, v in expo_sect.items()
    } if val_totale > 0 else {}

    return {
        "capital_initial":     cap_init,
        "cash":                cash,
        "valeur_positions":    val_positions,
        "valeur_totale":       val_totale,
        "pnl_eur":             pnl_eur,
        "pnl_pct":             pnl_pct,
        "drawdown_actuel":     dd_init,
        "drawdown_vs_peak":    dd_peak,
        "n_positions":         len(positions),
        "max_positions":       int((profile or {}).get("max_positions_simultanees", 0)),
        "cash_reserve_eur":    cash_reserve_eur,
        "cash_investissable":  cash_investissable,
        "expo_sectorielle":    expo_sect_pct,
        "peak":                peak,
        "derniere_maj":        portfolio.get("derniere_maj", ""),
    }


def load_recent_logs(n: int = 200) -> list[str]:
    if not LOG.exists():
        return []
    try:
        return LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def load_transcription(url: str) -> str:
    try:
        conn = sqlite3.connect(str(DB), timeout=10)
        row = conn.execute(
            "SELECT transcription_debat FROM articles WHERE url = ?",
            (url,),
        ).fetchone()
        conn.close()
        return (row[0] or "Aucun detail disponible.") if row else "Aucun detail disponible."
    except sqlite3.Error:
        return "Erreur de chargement."


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="locus-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<div class="locus-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)


def section(title: str, help_text: str | None = None) -> None:
    st.markdown(f'<div class="locus-section-title">{title}</div>', unsafe_allow_html=True)
    if help_text:
        st.markdown(f'<div class="locus-help">{help_text}</div>', unsafe_allow_html=True)


def status_pill(label: str, kind: str = "info") -> str:
    return f'<span class="locus-pill {kind}">{label}</span>'


def alt_theme() -> dict:
    return {
        "config": {
            "background": PALETTE["panel"],
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": PALETTE["text_soft"],
                "titleColor": PALETTE["text_soft"],
                "labelFontSize": 11,
                "titleFontSize": 11,
                "gridColor": PALETTE["border"],
                "domainColor": PALETTE["border"],
                "tickColor": PALETTE["border"],
                "labelFontWeight": "normal",
            },
            "legend": {
                "labelColor": PALETTE["text_soft"],
                "titleColor": PALETTE["text_soft"],
                "labelFontSize": 11,
            },
            "title": {
                "color": PALETTE["text"],
                "fontSize": 12,
                "fontWeight": "normal",
            },
        }
    }


alt.themes.register("locus", alt_theme)
alt.themes.enable("locus")


def bar_chart(df: pd.DataFrame, x: str, y: str, color_field: str | None = None,
              color_map: dict[str, str] | None = None, height: int = 220) -> alt.Chart:
    color_arg: alt.Color | alt.value
    if color_map and color_field:
        color_arg = alt.Color(
            f"{color_field}:N",
            scale=alt.Scale(
                domain=list(color_map.keys()),
                range=list(color_map.values()),
            ),
            legend=None,
        )
    else:
        color_arg = alt.value(PALETTE["accent"])
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=28)
        .encode(
            x=alt.X(f"{x}:N", sort="-y", axis=alt.Axis(title=None, labelAngle=0)),
            y=alt.Y(f"{y}:Q", axis=alt.Axis(title=None)),
            color=color_arg,
            tooltip=[x, y],
        )
        .properties(height=height)
    )


def line_chart(df: pd.DataFrame, x: str, y: str, color: str = None, height: int = 220) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2, color=color or PALETTE["accent"])
        .encode(
            x=alt.X(f"{x}:T", axis=alt.Axis(title=None)),
            y=alt.Y(f"{y}:Q", axis=alt.Axis(title=None)),
            tooltip=[x, y],
        )
        .properties(height=height)
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

PAGES = [
    ("Recommandations", "Recommandations IA"),
    ("Simulation", "Simulation de portefeuille"),
    ("Surveillance", "Surveillance systeme"),
]

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:0.4rem 0 0.6rem 0">
            <div style="font-size:1.15rem;font-weight:600;color:{PALETTE['text']};letter-spacing:-0.01em">
                Locus
            </div>
            <div style="font-size:0.78rem;color:{PALETTE['muted']};margin-top:0.15rem">
                Tableau de bord
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:0.5rem 0 1rem 0'/>", unsafe_allow_html=True)

    page_key = st.radio(
        "Navigation",
        options=[p[0] for p in PAGES],
        format_func=lambda k: dict(PAGES)[k],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin:1rem 0'/>", unsafe_allow_html=True)
    if st.button("Actualiser les donnees", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        f"""
        <div style="margin-top:1.5rem;padding:0.85rem 0.95rem;background:{PALETTE['panel_alt']};
                    border:1px solid {PALETTE['border']};border-radius:8px">
            <div style="font-size:0.7rem;color:{PALETTE['muted']};text-transform:uppercase;
                        letter-spacing:0.06em;font-weight:500;margin-bottom:0.45rem">
                A quoi sert ce tableau de bord
            </div>
            <div style="font-size:0.8rem;color:{PALETTE['text_soft']};line-height:1.55">
                Locus lit en continu les actualites financieres et les soumet a trois
                intelligences artificielles qui jouent chacune un role&nbsp;:
                <strong style="color:{PALETTE['text']}">l'optimiste</strong>,
                <strong style="color:{PALETTE['text']}">le pessimiste</strong> et
                <strong style="color:{PALETTE['text']}">l'arbitre</strong>.<br/><br/>
                Apres debat, le systeme emet une recommandation
                <strong style="color:{PALETTE['text']}">Acheter</strong>,
                <strong style="color:{PALETTE['text']}">Vendre</strong> ou
                <strong style="color:{PALETTE['text']}">Attendre</strong> sur l'action concernee.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1 - Recommandations
# ===========================================================================


def page_recommandations() -> None:
    page_header(
        eyebrow="Vue principale",
        title="Recommandations",
        subtitle=(
            "Pour chaque actualite financiere lue, le systeme emet une recommandation "
            "<strong>Acheter</strong>, <strong>Vendre</strong> ou <strong>Attendre</strong> "
            "sur l'action concernee, accompagnee d'un indicateur de coherence des analyses."
        ),
    )

    df = load_articles()
    if df.empty:
        st.info(
            "Aucune donnee disponible. Le pipeline doit avoir tourne au moins une fois "
            "pour alimenter la base `data/news_database.db`."
        )
        return

    # Normalise / enrich
    df["Decision"] = df["signal_final"].map(DECISION_LABELS).fillna(df["signal_final"])
    if "market_regime" in df.columns:
        df["Regime"] = df["market_regime"].map(REGIME_LABELS).fillna(df["market_regime"])
    else:
        df["Regime"] = "—"
    df["Amplitude"] = df.get("impact_strength", pd.Series(dtype=float)).round(3)
    df["Coherence"] = (df.get("consensus_rate", pd.Series(dtype=float)).fillna(0) * 100).round(0).astype(int)
    df["Date"] = df["date_utc"].astype(str).str[:16].str.replace("T", " ")

    treated = df[df["signal_final"].notna()].copy()
    signals = treated[treated["signal_final"].isin(["Achat", "Vente", "Neutre"])].copy()
    n_total = len(df)
    n_treated = len(treated)
    n_failed = int(treated["signal_final"].isin(REJECTED_VALUES).sum())

    last_date = df["date_utc"].dropna().max() if not df.empty else None

    # Crisis alert
    if "kill_switch_active" in df.columns and df["kill_switch_active"].fillna(0).astype(int).sum() > 0:
        st.error(
            "Volatilite des marches superieure au seuil critique. "
            "Toutes les recommandations sont actuellement suspendues."
        )

    # ---- KPIs ----
    section(
        "Vue d'ensemble",
        help_text=(
            "Recapitulatif des analyses produites par le systeme. "
            f"{n_total} article(s) collectes au total, dont {len(signals)} avec une "
            "recommandation exploitable. "
            f"{n_failed} ont ete rejetes (article non financier ou erreur d'analyse)."
        ),
    )
    n_buy = int((signals["signal_final"] == "Achat").sum())
    n_sell = int((signals["signal_final"] == "Vente").sum())
    n_hold = int((signals["signal_final"] == "Neutre").sum())
    avg_amplitude = signals["impact_strength"].mean() if not signals.empty else None
    avg_coherence = signals["consensus_rate"].mean() if not signals.empty else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Recommandations exploitables",
        len(signals),
        help=f"Sur {n_total} article(s) collectes.",
    )
    c2.metric("Acheter", n_buy)
    c3.metric("Vendre", n_sell)
    c4.metric("Attendre", n_hold)
    c5.metric(
        "Coherence moyenne",
        f"{avg_coherence*100:.0f}%" if avg_coherence is not None else "—",
        help=(
            "Mesure dans quelle proportion les deux methodes d'analyse linguistique "
            "(FinBERT et ABSA) pointent dans la meme direction. "
            "Une coherence elevee suggere un signal robuste, mais ce n'est PAS "
            "une probabilite de succes du trade."
        ),
    )

    if last_date:
        st.markdown(
            f'<div class="locus-help" style="margin-top:0.4rem">'
            f'Derniere analyse&nbsp;: <strong style="color:{PALETTE["text_soft"]}">'
            f'{str(last_date)[:16].replace("T"," ")}</strong> (UTC).</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ---- Filters + table ----
    section(
        "Liste des recommandations",
        help_text=(
            "Chaque ligne correspond a un article et a la recommandation produite par le systeme. "
            "Par defaut, seuls les articles ayant abouti a une recommandation exploitable sont affiches."
        ),
    )

    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with f1:
        tickers = ["Toutes"] + sorted(signals["ticker"].dropna().unique().tolist())
        ticker = st.selectbox("Action", tickers)
    with f2:
        avail_decisions = sorted(signals["Decision"].dropna().unique().tolist())
        decisions = ["Toutes"] + avail_decisions
        decision = st.selectbox("Recommandation", decisions)
    with f3:
        regimes = ["Tous"] + sorted([r for r in df["Regime"].dropna().unique().tolist() if r != "—"])
        regime = st.selectbox("Etat du marche", regimes if len(regimes) > 1 else ["Tous"])
    with f4:
        min_coh = st.slider("Coherence minimum (%)", 0, 100, 0, step=5)

    filtered = signals.copy()
    if ticker != "Toutes":
        filtered = filtered[filtered["ticker"] == ticker]
    if decision != "Toutes":
        filtered = filtered[filtered["Decision"] == decision]
    if regime not in ("Tous", None) and "Regime" in filtered.columns:
        filtered = filtered[filtered["Regime"] == regime]
    filtered = filtered[filtered["Coherence"].fillna(0) >= min_coh]

    st.markdown(
        f'<div class="locus-help">{len(filtered)} recommandation(s) affichee(s)</div>',
        unsafe_allow_html=True,
    )

    table_cols = ["ticker", "Date", "title", "Decision", "Coherence", "Amplitude", "Regime"]
    table_cols = [c for c in table_cols if c in filtered.columns]
    if not filtered.empty:
        rename = {
            "ticker": "Action",
            "title": "Titre de l'article",
            "Regime": "Marche",
        }
        st.dataframe(
            filtered[table_cols].rename(columns=rename),
            width="stretch",
            hide_index=True,
            height=320,
            column_config={
                "Coherence": st.column_config.ProgressColumn(
                    "Coherence",
                    min_value=0,
                    max_value=100,
                    format="%d%%",
                    help=(
                        "Accord entre les deux methodes d'analyse linguistique "
                        "(FinBERT et ABSA). Pas une probabilite de succes."
                    ),
                ),
                "Amplitude": st.column_config.NumberColumn(
                    "Amplitude",
                    format="%.2f",
                    help="Ampleur du mouvement de prix attendu, sur une echelle de 0 a 1.",
                ),
            },
        )
    else:
        st.info("Aucun resultat pour ces filtres.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ---- Charts ----
    g1, g2 = st.columns(2)
    with g1:
        section("Repartition des recommandations")
        if not signals.empty:
            counts = (
                signals["Decision"].value_counts().reset_index()
            )
            counts.columns = ["Decision", "Nombre"]
            st.altair_chart(
                bar_chart(counts, "Decision", "Nombre", "Decision", DECISION_COLOR),
                width="stretch",
            )
        else:
            st.info("Pas encore de signaux exploitables.")
    with g2:
        section("Volume d'analyses par action")
        if not treated.empty:
            counts = treated["ticker"].value_counts().head(10).reset_index()
            counts.columns = ["Action", "Articles"]
            st.altair_chart(
                bar_chart(counts, "Action", "Articles"),
                width="stretch",
            )
        else:
            st.info("Aucun ticker traite.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ---- Debate detail ----
    section(
        "Detail d'une analyse",
        help_text=(
            "Selectionnez un article pour consulter le raisonnement complet des trois "
            "agents (Haussier, Baissier, Neutre) ainsi que la justification du signal final."
        ),
    )

    with_debate = filtered[filtered["signal_final"].notna()].reset_index(drop=True)
    if not with_debate.empty:
        labels = [
            f"{row['ticker']:>6}  -  {row['Date']}  -  {str(row['title'])[:80]}"
            for _, row in with_debate.iterrows()
        ]
        idx = st.selectbox(
            "Article",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
            label_visibility="collapsed",
        )
        selected = with_debate.iloc[idx]
        meta_cols = st.columns(4)
        meta_cols[0].markdown(
            f"**Action**<br><span style='color:{PALETTE['text']};font-size:1rem'>{selected['ticker']}</span>",
            unsafe_allow_html=True,
        )
        meta_cols[1].markdown(
            f"**Decision**<br>{status_pill(selected['Decision'], 'ok' if selected['signal_final']=='Achat' else 'err' if selected['signal_final']=='Vente' else 'muted')}",
            unsafe_allow_html=True,
        )
        meta_cols[2].markdown(
            f"**Coherence**<br><span style='color:{PALETTE['text']};font-size:1rem'>{int(selected['Coherence'])}%</span>",
            unsafe_allow_html=True,
        )
        meta_cols[3].markdown(
            f"**Amplitude**<br><span style='color:{PALETTE['text']};font-size:1rem'>{selected['Amplitude']:.2f}</span>"
            if pd.notna(selected.get("Amplitude")) else "**Amplitude**<br>—",
            unsafe_allow_html=True,
        )

        if pd.notna(selected.get("argument_dominant")):
            st.markdown(
                f"<div class='locus-card' style='margin-top:1rem'>"
                f"<div class='locus-eyebrow'>Argument dominant</div>"
                f"<div style='color:{PALETTE['text']};font-size:0.92rem;line-height:1.55;margin-top:0.3rem'>"
                f"{selected['argument_dominant']}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
        st.text_area(
            "Transcription du debat",
            load_transcription(selected["url"]),
            height=320,
            label_visibility="collapsed",
        )
    else:
        st.info("Aucune analyse disponible pour les filtres en cours.")


# ===========================================================================
# PAGE 2 - Simulation
# ===========================================================================


def _fmt_eur(x: float) -> str:
    return f"{x:,.0f} EUR".replace(",", " ")


def _fmt_pct(x: float, sign: bool = False) -> str:
    return f"{x*100:+.2f}%" if sign else f"{x*100:.2f}%"


def page_simulation() -> None:
    page_header(
        eyebrow="Performance et portefeuille",
        title="Performance",
        subtitle=(
            "Mesures scientifiques de la performance historique du systeme "
            "(backtest walk-forward, frais d'execution, regimes de marche) et "
            "etat actuel du portefeuille fictif."
        ),
    )

    df = load_trades()
    profile = load_profile()
    portfolio = load_portfolio()
    summary = portfolio_summary(portfolio, profile)
    wf = load_walk_forward()
    costs_eval = load_execution_costs()
    regimes_eval = load_market_regimes()
    market_trades = load_market_trades()
    calib_meta = load_calibration()

    # =====================================================================
    # SECTION A — Performance historique (walk-forward backtest)
    # Reference : Bailey & Lopez de Prado (2014) — backtest sans biais
    # =====================================================================
    section(
        "Performance historique du systeme",
        help_text=(
            "Resultats du backtest <strong>walk-forward</strong> (Bailey &amp; Lopez de Prado 2014) "
            "sur des fenetres de 3 mois successives. Chaque fenetre est evaluee independamment "
            "pour eviter le biais de sur-apprentissage. Ces chiffres repondent a la question "
            "&laquo; ce systeme a-t-il un avantage statistique ? &raquo;."
        ),
    )

    if wf:
        median_sharpe = float(wf.get("median_sharpe", 0))
        mean_sharpe = float(wf.get("mean_sharpe", 0))
        min_sharpe = float(wf.get("min_sharpe", 0))
        max_sharpe = float(wf.get("max_sharpe", 0))
        std_sharpe = float(wf.get("std_sharpe", 0))
        hit_rate = float(wf.get("mean_hit_rate", 0))
        stability = float(wf.get("stability_score", 0))
        consistency = float(wf.get("consistency_ratio", 0))
        n_active = int(wf.get("n_windows_active", 0))
        n_total = int(wf.get("n_windows_total", 0))
        trades_per_window = float(wf.get("mean_trades_per_window", 0))

        a, b, c, d = st.columns(4)
        a.metric(
            "Sharpe median",
            f"{median_sharpe:+.2f}",
            help=(
                "Rendement excedentaire par unite de risque (volatilite annualisee). "
                "Mediane plus robuste que la moyenne aux fenetres extremes. "
                "Reperes : >1 = bon, >2 = excellent, <0 = perte."
            ),
        )
        b.metric(
            "Hit rate moyen",
            f"{hit_rate*100:.1f}%",
            help=(
                "Proportion de trades gagnants par fenetre. "
                "Un edge requiert hit rate combine au payoff ratio qui depasse "
                "le break-even (~43% si payoff = 1.3 typique de cette strategie)."
            ),
        )
        c.metric(
            "Stability score",
            f"{stability:.2f}",
            help=(
                "Coherence inter-temporelle : 1.00 = stable, 0 = chaotique. "
                "Calcule sur la dispersion des Sharpes par fenetre. "
                "Ce systeme : un Sharpe median positif mais une stabilite moyenne."
            ),
        )
        d.metric(
            "Couverture du backtest",
            f"{n_active} / {n_total}",
            help=(
                "Fenetres avec au moins un trade vs total evalue. "
                "Une couverture trop faible reduit la robustesse statistique."
            ),
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        e, f, g, h = st.columns(4)
        e.metric(
            "Sharpe moyen",
            f"{mean_sharpe:+.2f}",
            delta=f"σ = {std_sharpe:.2f}",
            delta_color="off",
            help="Moyenne des Sharpes par fenetre. Un ecart-type eleve signale une instabilite reelle.",
        )
        f.metric(
            "Pire fenetre",
            f"{min_sharpe:+.2f}",
            help="Sharpe le plus bas observe sur toutes les fenetres. Permet d'apprehender le worst-case.",
        )
        g.metric(
            "Meilleure fenetre",
            f"{max_sharpe:+.2f}",
            help="Sharpe le plus eleve observe.",
        )
        h.metric(
            "Trades / fenetre",
            f"{trades_per_window:.1f}",
            help=(
                "Frequence de signaux exploitables par fenetre de 3 mois. "
                "Trop bas = pas assez d'opportunites pour rentabiliser apres frais fixes."
            ),
        )

        # Distribution des Sharpe ratios
        windows = wf.get("windows", []) or []
        sharpes = [w.get("sharpe_ann", 0) for w in windows if w.get("n_trades", 0) > 0]
        if sharpes:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            section(
                "Distribution des Sharpes par fenetre",
                help_text=(
                    "Histogramme des Sharpe ratios observes. "
                    "Plus la masse est a droite de zero, plus le systeme est rentable apres ajustement du risque. "
                    "Une distribution centree autour de zero indique une performance equivalente au hasard."
                ),
            )
            sh_df = pd.DataFrame({"Sharpe": sharpes})
            hist = (
                alt.Chart(sh_df)
                .mark_bar(color=PALETTE["accent"], opacity=0.85)
                .encode(
                    x=alt.X(
                        "Sharpe:Q",
                        bin=alt.Bin(maxbins=20),
                        axis=alt.Axis(title="Sharpe ratio annualise"),
                    ),
                    y=alt.Y("count():Q", axis=alt.Axis(title="Nombre de fenetres")),
                    tooltip=[alt.Tooltip("count():Q", title="Fenetres"),
                             alt.Tooltip("Sharpe:Q", title="Sharpe", format=".2f")],
                )
                .properties(height=210)
            )
            zero_line = (
                alt.Chart(pd.DataFrame({"x": [0]}))
                .mark_rule(color=PALETTE["warn"], strokeDash=[4, 4], strokeWidth=1.5)
                .encode(x="x:Q")
            )
            st.altair_chart(hist + zero_line, width="stretch")
    else:
        st.info(
            "Aucun backtest walk-forward disponible. "
            "Lancer `python eval/run_eval.py --walk-forward` pour generer ces metriques."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # =====================================================================
    # SECTION B — Performance par regime de marche
    # Repond a "quand le systeme fonctionne-t-il le mieux ?"
    # =====================================================================
    if regimes_eval and regimes_eval.get("regime_metrics"):
        section(
            "Performance par regime de marche",
            help_text=(
                "Decomposition de la performance par contexte macro : marche haussier (BULL), "
                "baissier (BEAR), lateral (SIDEWAYS), tres volatile (HIGH_VOL). "
                "Repond a la question &laquo; le systeme fonctionne-t-il dans tous les regimes ? &raquo;."
            ),
        )
        rows = []
        for regime, m in regimes_eval["regime_metrics"].items():
            if not isinstance(m, dict) or m.get("n", 0) == 0:
                continue
            rows.append({
                "Regime": REGIME_LABELS.get(regime, regime),
                "Trades": int(m.get("n", 0)),
                "Win rate": float(m.get("win_rate", 0)),
                "Rendement moyen (%)": float(m.get("mean_return", 0)),
                "Rendement total (%)": float(m.get("total_return", 0)),
                "Alpha vs SPY (%)": float(m.get("alpha_vs_spy", 0)),
            })
        if rows:
            reg_df = pd.DataFrame(rows)
            st.dataframe(
                reg_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Win rate": st.column_config.ProgressColumn(
                        "Win rate",
                        min_value=0.0, max_value=1.0, format="%.0f%%",
                        help="Pourcentage de trades gagnants dans ce regime.",
                    ),
                    "Rendement moyen (%)": st.column_config.NumberColumn(
                        "Rendement moyen",
                        format="%.2f%%",
                        help="Rendement moyen par trade dans ce regime.",
                    ),
                    "Rendement total (%)": st.column_config.NumberColumn(
                        "Rendement total",
                        format="%+.2f%%",
                    ),
                    "Alpha vs SPY (%)": st.column_config.NumberColumn(
                        "Alpha vs SPY",
                        format="%+.2f%%",
                        help=(
                            "Excedent de rendement par rapport au benchmark passif S&P 500. "
                            "Positif = le systeme bat l'indice dans ce regime."
                        ),
                    ),
                },
            )
        st.markdown("<hr/>", unsafe_allow_html=True)

    # =====================================================================
    # SECTION C — Couts d'execution (par tier de courtier)
    # Reference : Almgren-Chriss (2001) — modele de cout d'impact
    # =====================================================================
    if costs_eval:
        section(
            "Performance nette apres frais d'execution",
            help_text=(
                "Modele <strong>Almgren-Chriss (2001)</strong> qui decompose les frais d'execution "
                "en commission, spread, impact de marche et slippage. Repond a la question "
                "&laquo; les frais vont-ils manger l'edge ? &raquo;. "
                "Compare trois profils de courtier representatifs."
            ),
        )

        rows = []
        for tier_key, t in costs_eval.items():
            if not isinstance(t, dict):
                continue
            label = t.get("broker_label") or tier_key
            gross = t.get("gross", {})
            net = t.get("net", {})
            costs = t.get("costs", {})
            be = t.get("break_even", {})
            rows.append({
                "Courtier": label,
                "Sharpe brut": float(gross.get("sharpe", 0)),
                "Sharpe net": float(net.get("sharpe", 0)),
                "Rendement net (%)": float(net.get("total_return", 0)),
                "Frais cumules (%)": float(costs.get("avg_total_cost_pct", 0)),
                "Hit rate min. requis": float(be.get("breakeven_accuracy", 0)),
                "Payoff ratio net": float(be.get("payoff_ratio_net", 0)),
            })

        if rows:
            costs_df = pd.DataFrame(rows)
            st.dataframe(
                costs_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Sharpe brut": st.column_config.NumberColumn(
                        "Sharpe brut",
                        format="%+.2f",
                        help="Sharpe avant frais d'execution.",
                    ),
                    "Sharpe net": st.column_config.NumberColumn(
                        "Sharpe net",
                        format="%+.2f",
                        help="Sharpe apres frais. Difference = cout reel pour cet investisseur.",
                    ),
                    "Rendement net (%)": st.column_config.NumberColumn(
                        "Rendement net",
                        format="%+.2f%%",
                    ),
                    "Frais cumules (%)": st.column_config.NumberColumn(
                        "Frais cumules",
                        format="%.2f%%",
                        help="Frais moyens par trade (commission + spread + impact + slippage).",
                    ),
                    "Hit rate min. requis": st.column_config.ProgressColumn(
                        "Hit rate min. requis",
                        min_value=0.0, max_value=1.0, format="%.0f%%",
                        help=(
                            "Pourcentage minimum de victoires necessaire pour etre rentable apres frais "
                            "(break-even accuracy). Si le hit rate observe est inferieur, le systeme perd de l'argent."
                        ),
                    ),
                    "Payoff ratio net": st.column_config.NumberColumn(
                        "Payoff ratio net",
                        format="%.2f",
                        help=(
                            "Rapport entre la taille moyenne d'un gain net et celle d'une perte nette. "
                            ">1 = les gains sont plus grands que les pertes."
                        ),
                    ),
                },
            )

        st.markdown("<hr/>", unsafe_allow_html=True)

    # =====================================================================
    # SECTION D — Qualite du modele (calibration ECE)
    # =====================================================================
    if calib_meta:
        section(
            "Calibration du modele",
            help_text=(
                "Mesure dans quelle proportion les confiances annoncees correspondent "
                "a la realite. Une calibration parfaite signifie qu'une recommandation "
                "annoncee a 80% de confiance se realise effectivement 80% du temps."
            ),
        )
        ece_before = float(calib_meta.get("ece_before", 0))
        ece_after = float(calib_meta.get("ece_after", 0))
        ok = bool(calib_meta.get("ece_target_met", False))
        method = calib_meta.get("method", "—")

        ka, kb, kc, kd = st.columns(4)
        ka.metric(
            "Methode utilisee",
            method,
            help="Algorithme de recalibration (Platt scaling ou Isotonic regression / PAVA).",
        )
        kb.metric(
            "Erreur avant correction",
            f"{ece_before:.3f}",
            help="ECE brut. 0 = parfait, >0.10 = mediocre.",
        )
        kc.metric(
            "Erreur apres correction",
            f"{ece_after:.3f}",
            delta=f"{ece_after - ece_before:+.3f}",
            delta_color="inverse",
            help="ECE apres recalibration. Cible scientifique : < 0.05.",
        )
        kd.metric(
            "Objectif < 0.05",
            "Atteint" if ok else "Non atteint",
            help="Seuil ECE < 0.05 = calibration consideree fiable (Guo et al. 2017).",
        )
        st.markdown("<hr/>", unsafe_allow_html=True)

    # =====================================================================
    # 1. STRATEGIE — Cadre de decision (issu de investor_profile.json)
    # =====================================================================
    if profile:
        section(
            "Cadre de la strategie",
            help_text=(
                "Parametres du profil investisseur qui contraignent toutes les decisions "
                "du systeme. Ce cadre est defini avant que la simulation commence."
            ),
        )

        risk_label = RISK_LABELS.get(profile.get("risk_tolerance"), "—")
        horizon_label = HORIZON_LABELS.get(profile.get("horizon"), "—")
        objectif_label = OBJECTIVE_LABELS.get(profile.get("objectif"), "—")
        cap_init = float(profile.get("capital_total", 0))
        alloc_max_pct = float(profile.get("allocation_max_par_position", 0))
        alloc_max_eur = cap_init * alloc_max_pct
        cash_min_pct = float(profile.get("cash_reserve_min", 0))
        cash_min_eur = cap_init * cash_min_pct
        sl_pct = float(profile.get("stop_loss_par_trade", 0))
        tp_pct = float(profile.get("take_profit_par_trade", 0))
        max_dd = float(profile.get("max_drawdown_tolerance", 0))
        sect_max = float(profile.get("secteur_max_exposition", 0))
        max_pos = int(profile.get("max_positions_simultanees", 0))

        sa, sb, sc = st.columns(3)
        sa.markdown(
            f"""
            <div class='locus-card'>
              <div class='locus-eyebrow'>Capital de depart</div>
              <div style='font-size:1.4rem;font-weight:600;color:{PALETTE['text']};margin-top:0.2rem'>
                {_fmt_eur(cap_init)}
              </div>
              <div class='locus-help'>Dotation initiale fictive du portefeuille.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sb.markdown(
            f"""
            <div class='locus-card'>
              <div class='locus-eyebrow'>Profil &amp; horizon</div>
              <div style='font-size:0.95rem;color:{PALETTE['text']};margin-top:0.2rem;line-height:1.5'>
                <strong>{risk_label}</strong><br/>
                {horizon_label}
              </div>
              <div class='locus-help'>Objectif&nbsp;: {objectif_label.lower()}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sc.markdown(
            f"""
            <div class='locus-card'>
              <div class='locus-eyebrow'>Drawdown maximum tolere</div>
              <div style='font-size:1.4rem;font-weight:600;color:{PALETTE['warn']};margin-top:0.2rem'>
                {max_dd*100:.0f}%
              </div>
              <div class='locus-help'>
                Au-dela de cette perte par rapport au capital initial,
                aucun nouvel achat n'est autorise.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

        ra, rb, rc, rd = st.columns(4)
        ra.metric(
            "Max par position",
            f"{alloc_max_pct*100:.0f}%",
            help=f"Soit {_fmt_eur(alloc_max_eur)} maximum par titre.",
        )
        rb.metric(
            "Reserve de cash",
            f"{cash_min_pct*100:.0f}%",
            help=f"Soit {_fmt_eur(cash_min_eur)} qui ne doivent jamais etre investis.",
        )
        rc.metric(
            "Stop-loss / Take-profit",
            f"{sl_pct*100:.0f}% / +{tp_pct*100:.0f}%",
            help=(
                "Niveau de perte qui declenche une cloture (stop-loss) "
                "et niveau de gain cible (take-profit) par trade."
            ),
        )
        rd.metric(
            "Concentration secteur max",
            f"{sect_max*100:.0f}%",
            help=f"Aucun secteur ne peut depasser cette part du portefeuille (max {max_pos} positions ouvertes simultanement).",
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

    # =====================================================================
    # 2. ETAT ACTUEL DU PORTEFEUILLE — depuis portfolio_state.json
    # =====================================================================
    if summary and summary.get("capital_initial", 0) > 0:
        section(
            "Etat actuel du portefeuille",
            help_text=(
                "Photographie temps reel du portefeuille fictif. La valeur totale = "
                "valeur des positions ouvertes + cash disponible."
            ),
        )

        cap_init = summary["capital_initial"]
        val_tot = summary["valeur_totale"]
        pnl_eur = summary["pnl_eur"]
        pnl_pct = summary["pnl_pct"]
        cash = summary["cash"]
        n_pos = summary["n_positions"]
        max_pos = summary["max_positions"]
        cash_inv = summary["cash_investissable"]
        dd_init = summary["drawdown_actuel"]
        dd_peak = summary["drawdown_vs_peak"]

        pa, pb, pc, pd_, pe = st.columns(5)
        pa.metric(
            "Valeur totale",
            _fmt_eur(val_tot),
            delta=_fmt_pct(pnl_pct, sign=True) if cap_init > 0 else None,
            help="Cash disponible + valeur de marche des positions ouvertes.",
        )
        pb.metric(
            "P&L cumule",
            _fmt_eur(pnl_eur),
            delta=f"{pnl_eur:+.0f} EUR vs {_fmt_eur(cap_init)}".replace(",", " "),
            help="Variation absolue de la valeur du portefeuille depuis l'origine.",
        )
        pc.metric(
            "Cash disponible",
            _fmt_eur(cash),
            help=(
                f"Dont {_fmt_eur(cash_inv)} reellement investissable apres deduction "
                f"de la reserve obligatoire de {_fmt_eur(summary['cash_reserve_eur'])}."
            ),
        )
        pd_.metric(
            "Positions ouvertes",
            f"{n_pos} / {max_pos}" if max_pos else f"{n_pos}",
            help="Nombre de titres detenus simultanement, vs maximum autorise.",
        )
        dd_color_help = (
            "Si la valeur tombe en dessous du capital initial, la difference est "
            "qualifiee de drawdown. Au-dela du seuil tolere, le systeme refuse tout nouvel achat."
        )
        pe.metric(
            "Drawdown vs initial",
            _fmt_pct(dd_init, sign=True),
            help=dd_color_help,
        )

        # Sectoral exposure if any
        if summary["expo_sectorielle"]:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            section("Exposition par secteur")
            expo_df = pd.DataFrame(
                [{"Secteur": s, "Part": v} for s, v in summary["expo_sectorielle"].items()]
            ).sort_values("Part", ascending=False)
            chart = (
                alt.Chart(expo_df)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=28)
                .encode(
                    x=alt.X("Secteur:N", sort="-y", axis=alt.Axis(title=None, labelAngle=0)),
                    y=alt.Y("Part:Q", axis=alt.Axis(title=None, format=".0%")),
                    color=alt.value(PALETTE["accent"]),
                    tooltip=["Secteur", alt.Tooltip("Part:Q", format=".1%")],
                )
                .properties(height=200)
            )
            st.altair_chart(chart, width="stretch")

        st.markdown("<hr/>", unsafe_allow_html=True)
    elif portfolio is None:
        st.info(
            "Etat du portefeuille indisponible : `data/portfolio_state.json` introuvable. "
            "Le pipeline doit s'executer au moins une fois pour initialiser le portefeuille."
        )

    # =====================================================================
    # 3. ACTIVITE SIMULEE — depuis logs/dry_run_trades.jsonl
    # =====================================================================
    if df.empty:
        st.info(
            "Aucun ordre simule pour l'instant.\n\n"
            "Pour activer le mode simulation, mettez `DRY_RUN=1` dans `.env` "
            "puis relancez le pipeline."
        )
        return

    # Enrichissements
    cap_init_for_pct = float((profile or {}).get("capital_total", 10000.0))
    if "montant_eur" in df.columns:
        df["capital_deploye_cumule"] = df["montant_eur"].abs().cumsum()
        df["pct_du_capital"] = df["montant_eur"].abs() / cap_init_for_pct
        # Pertes max et gains cibles attendus par ordre
        sl = abs(float((profile or {}).get("stop_loss_par_trade", 0.05)))
        tp = abs(float((profile or {}).get("take_profit_par_trade", 0.12)))
        df["risque_max_eur"] = df["montant_eur"].abs() * sl
        df["objectif_eur"]   = df["montant_eur"].abs() * tp

    df["Decision"] = (
        df["signal"].map(DECISION_LABELS).fillna(df.get("signal", pd.Series(dtype=str)))
        if "signal" in df.columns
        else "—"
    )
    if "market_regime" in df.columns:
        df["Regime"] = df["market_regime"].map(REGIME_LABELS).fillna(df["market_regime"])

    section(
        "Activite simulee",
        help_text=(
            "Statistiques sur l'ensemble des ordres fictifs emis par le systeme. "
            "Chaque ordre dimensionne le montant via la formule de Kelly fractionnee + "
            "Fixed Fractional (minimum des deux, approche conservatrice — Thorp 2008, "
            "MacLean et al. 2010), capee par les regles de la strategie."
        ),
    )
    n = len(df)
    n_buy = int((df["signal"] == "Achat").sum()) if "signal" in df.columns else 0
    n_sell = int((df["signal"] == "Vente").sum()) if "signal" in df.columns else 0
    avg_size_eur = df["montant_eur"].abs().mean() if "montant_eur" in df.columns else None
    avg_size_pct = (avg_size_eur / cap_init_for_pct) if avg_size_eur else None
    risque_total_eur = df["risque_max_eur"].sum() if "risque_max_eur" in df.columns else None
    avg_win = df["win_prob"].mean() if "win_prob" in df.columns else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ordres simules", f"{n:,}".replace(",", " "))
    c2.metric(
        "Repartition Achat / Vente",
        f"{n_buy} / {n_sell}",
        help="Nombre d'ordres d'achat et de vente sur la periode.",
    )
    c3.metric(
        "Taille moyenne par ordre",
        _fmt_eur(avg_size_eur) if avg_size_eur is not None else "—",
        delta=f"{avg_size_pct*100:.1f}% du capital" if avg_size_pct is not None else None,
        delta_color="off",
        help="Montant moyen alloue a un ordre, en euros et en pourcentage du capital initial.",
    )
    c4.metric(
        "Perte maximale theorique",
        _fmt_eur(risque_total_eur) if risque_total_eur is not None else "—",
        help=(
            "Somme des pertes potentielles si tous les ordres simules touchaient leur stop-loss. "
            "Plafond superieur du risque cumule."
        ),
    )
    c5.metric(
        "Probabilite de succes moyenne",
        f"{avg_win*100:.0f}%" if avg_win is not None and pd.notna(avg_win) else "—",
        help=(
            "Probabilite bayesienne qu'un ordre individuel soit gagnant. "
            "Distincte de la 'Coherence' (page Recommandations) qui mesure l'accord entre methodes d'analyse."
        ),
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Capital chart - exprimee en % du capital initial
    if "capital_deploye_cumule" in df.columns and "timestamp" in df.columns:
        section(
            "Capital deploye dans le temps",
            help_text=(
                "Volume cumule des montants engages par les ordres simules, en euros et "
                "en pourcentage du capital initial. Cette courbe represente l'activite, pas un P&L."
            ),
        )
        chart_df = df[["timestamp", "capital_deploye_cumule"]].dropna().copy()
        chart_df["pct_capital"] = chart_df["capital_deploye_cumule"] / cap_init_for_pct
        if not chart_df.empty:
            chart = (
                alt.Chart(chart_df)
                .mark_area(
                    line={"color": PALETTE["accent"], "strokeWidth": 2},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color=PALETTE["accent"], offset=0),
                            alt.GradientStop(color=PALETTE["panel"], offset=1),
                        ],
                        x1=1, x2=1, y1=1, y2=0,
                    ),
                    opacity=0.5,
                )
                .encode(
                    x=alt.X("timestamp:T", axis=alt.Axis(title=None)),
                    y=alt.Y(
                        "capital_deploye_cumule:Q",
                        axis=alt.Axis(title="Capital deploye (EUR)"),
                    ),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Date"),
                        alt.Tooltip("capital_deploye_cumule:Q", title="Cumul (EUR)", format=",.0f"),
                        alt.Tooltip("pct_capital:Q", title="% du capital initial", format=".1%"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width="stretch")

    st.markdown("<hr/>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        section("Repartition par decision")
        if "Decision" in df.columns and not df["Decision"].dropna().empty:
            counts = df["Decision"].value_counts().reset_index()
            counts.columns = ["Decision", "Nombre"]
            st.altair_chart(
                bar_chart(counts, "Decision", "Nombre", "Decision", DECISION_COLOR),
                width="stretch",
            )
    with g2:
        section(
            "Indice de peur (VIX)",
            help_text=(
                "Au-dela de 45, le systeme suspend automatiquement toute decision "
                "(comme observe lors du COVID en 2020 ou de la crise de 2008)."
            ),
        )
        if "vix" in df.columns and "timestamp" in df.columns:
            vix_df = df[["timestamp", "vix"]].dropna(subset=["vix"])
            if not vix_df.empty:
                st.altair_chart(
                    line_chart(vix_df, "timestamp", "vix", color=PALETTE["warn"], height=220),
                    width="stretch",
                )

    st.markdown("<hr/>", unsafe_allow_html=True)

    section(
        "Historique des ordres simules",
        help_text=(
            "Pour chaque ordre&nbsp;: montant en euros et part du capital initial, "
            "perte maximale theorique (au stop-loss) et gain cible (au take-profit), "
            "probabilite de succes estimee."
        ),
    )
    cols_order = [
        "timestamp", "ticker", "Decision", "prix", "quantite", "montant_eur",
        "pct_du_capital", "risque_max_eur", "objectif_eur",
        "win_prob", "risk_level", "Regime", "vix",
    ]
    cols_order = [c for c in cols_order if c in df.columns]
    rename = {
        "timestamp": "Date",
        "ticker": "Action",
        "prix": "Prix",
        "quantite": "Quantite",
        "montant_eur": "Montant",
        "pct_du_capital": "% capital",
        "risque_max_eur": "Perte max",
        "objectif_eur": "Gain cible",
        "win_prob": "Prob. succes",
        "risk_level": "Risque",
        "Regime": "Marche",
        "vix": "VIX",
    }
    table = df[cols_order].sort_values("timestamp", ascending=False).rename(columns=rename) if "timestamp" in df.columns else df[cols_order].rename(columns=rename)

    column_config = {}
    if "Prob. succes" in table.columns:
        column_config["Prob. succes"] = st.column_config.ProgressColumn(
            "Prob. succes", min_value=0.0, max_value=1.0, format="%.0f%%"
        )
    if "Montant" in table.columns:
        column_config["Montant"] = st.column_config.NumberColumn(
            "Montant", format="%.0f EUR"
        )
    if "% capital" in table.columns:
        column_config["% capital"] = st.column_config.NumberColumn(
            "% capital", format="%.1f%%",
            help="Part du capital initial engagee par cet ordre.",
        )
    if "Perte max" in table.columns:
        column_config["Perte max"] = st.column_config.NumberColumn(
            "Perte max", format="-%.0f EUR",
            help="Perte si l'ordre touche le stop-loss du profil.",
        )
    if "Gain cible" in table.columns:
        column_config["Gain cible"] = st.column_config.NumberColumn(
            "Gain cible", format="+%.0f EUR",
            help="Gain si l'ordre atteint le take-profit du profil.",
        )

    # Convert % capital to fraction of 100 for display in NumberColumn
    if "% capital" in table.columns:
        table = table.copy()
        table["% capital"] = (table["% capital"] * 100).round(2)

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=320,
        column_config=column_config,
    )

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Telecharger en CSV",
        data=csv,
        file_name="locus_simulation.csv",
        mime="text/csv",
    )


# ===========================================================================
# PAGE 3 - Surveillance systeme
# ===========================================================================


def page_surveillance() -> None:
    page_header(
        eyebrow="Etat technique",
        title="Surveillance",
        subtitle=(
            "Etat de fonctionnement du systeme&nbsp;: securites automatiques, "
            "consommation d'IA et journal d'evenements."
        ),
    )

    logs = load_recent_logs(200)
    cost = load_llm_cost()

    # ---- Status overview ----
    crisis = any("KILL-SWITCH" in line or "HOLD_SYSTEMIC" in line for line in logs)
    errors = [line for line in logs if "ERROR" in line or "CRITICAL" in line]
    warnings = [line for line in logs if "WARNING" in line]

    section("Etat global")
    c1, c2, c3 = st.columns(3)

    crisis_pill = status_pill("Suspendu", "err") if crisis else status_pill("Operationnel", "ok")
    c1.markdown(
        f"<div class='locus-card'><div class='locus-eyebrow'>Securite marche</div>"
        f"<div style='margin-top:0.5rem'>{crisis_pill}</div>"
        f"<div class='locus-help'>{'Trading suspendu' if crisis else 'Aucune alerte de crise'}</div></div>",
        unsafe_allow_html=True,
    )

    if cost is None:
        budget_pill = status_pill("Inconnu", "muted")
        budget_detail = "Pas de donnees du jour."
    else:
        pct = cost.get("budget_used_pct") or 0
        if pct >= 100:
            budget_pill = status_pill("Depasse", "err")
        elif pct >= 80:
            budget_pill = status_pill("Eleve", "warn")
        else:
            budget_pill = status_pill("Sous controle", "ok")
        budget_detail = f"{pct:.0f}% utilise"
    c2.markdown(
        f"<div class='locus-card'><div class='locus-eyebrow'>Budget IA</div>"
        f"<div style='margin-top:0.5rem'>{budget_pill}</div>"
        f"<div class='locus-help'>{budget_detail}</div></div>",
        unsafe_allow_html=True,
    )

    if not logs:
        log_pill = status_pill("Indisponible", "muted")
        log_detail = "Pas de journal."
    elif errors:
        log_pill = status_pill(f"{len(errors)} erreur(s)", "err")
        log_detail = "Voir les erreurs ci-dessous."
    elif warnings:
        log_pill = status_pill(f"{len(warnings)} avertissement(s)", "warn")
        log_detail = "Aucune erreur critique."
    else:
        log_pill = status_pill("Aucune anomalie", "ok")
        log_detail = "Le pipeline tourne normalement."
    c3.markdown(
        f"<div class='locus-card'><div class='locus-eyebrow'>Journal</div>"
        f"<div style='margin-top:0.5rem'>{log_pill}</div>"
        f"<div class='locus-help'>{log_detail}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Securite marche",
        "Budget IA",
        "Journal",
    ])

    # ---- Tab 1: Crisis ----
    with tab1:
        if crisis:
            st.error(
                "La volatilite des marches a depasse le seuil critique. "
                "Toutes les decisions ont ete suspendues automatiquement pour proteger le portefeuille."
            )
        else:
            st.success("Aucune alerte. Le systeme fonctionne normalement.")
        st.markdown(
            f"""
            <div class='locus-help' style='margin-top:1rem'>
                Le systeme surveille en continu l'indice <strong>VIX</strong> (mesure de la volatilite
                attendue du S&amp;P 500). Si le VIX depasse <strong>45</strong>, toutes les
                recommandations de trading sont automatiquement suspendues. Ce seuil correspond a
                des episodes historiques majeurs (COVID 2020, crise financiere 2008).
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Tab 2: LLM Budget ----
    with tab2:
        if cost is None:
            st.info("Aucune donnee de consommation disponible pour l'instant.")
        else:
            total = cost.get("total_usd", 0.0)
            budget = cost.get("budget_usd", 5.0)
            pct = cost.get("budget_used_pct") or 0.0

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Date", cost.get("date", "—"))
            cc2.metric("Cout du jour", f"${total:.4f}")
            cc3.metric("Budget consomme", f"{pct:.0f}%")

            st.progress(
                min(1.0, total / budget if budget > 0 else 0),
                text=f"${total:.4f} depenses sur ${budget:.2f} autorises ce jour",
            )

            if pct >= 100:
                st.error("Budget depasse. Le pipeline s'est arrete proprement et reprendra demain.")
            elif pct >= 80:
                st.warning("Plus de 80% du budget journalier utilise.")
            else:
                st.success("Budget sous controle.")

            models = cost.get("calls_by_model", {})
            if models:
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                section("Detail par modele")
                rows = [
                    {
                        "Modele": m,
                        "Appels": v.get("calls", 0),
                        "Tokens": v.get("prompt_tokens", 0) + v.get("completion_tokens", 0),
                        "Cout (USD)": v.get("usd", 0),
                    }
                    for m, v in models.items()
                ]
                model_df = pd.DataFrame(rows)
                st.dataframe(
                    model_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Tokens": st.column_config.NumberColumn(format="%d"),
                        "Cout (USD)": st.column_config.NumberColumn(format="$%.4f"),
                    },
                )

    # ---- Tab 3: Logs ----
    with tab3:
        if not logs:
            st.info("Aucun journal disponible. Activez `LOG_FILE=logs/pipeline.log` dans `.env`.")
        else:
            ll1, ll2 = st.columns(2)
            with ll1:
                if errors:
                    st.error(f"{len(errors)} erreur(s) recente(s)")
                    with st.expander("Voir les erreurs"):
                        st.code("\n".join(errors[-10:]), language=None)
                else:
                    st.success("Aucune erreur recente")
            with ll2:
                if warnings:
                    st.warning(f"{len(warnings)} avertissement(s)")
                    with st.expander("Voir les avertissements"):
                        st.code("\n".join(warnings[-10:]), language=None)
                else:
                    st.success("Aucun avertissement recent")

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            n = st.slider("Nombre de lignes", 20, 200, 60, step=10)
            st.text_area(
                "Journal",
                "\n".join(logs[-n:]),
                height=340,
                label_visibility="collapsed",
            )
            st.caption(f"Source : {LOG}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page_key == "Recommandations":
    page_recommandations()
elif page_key == "Simulation":
    page_simulation()
else:
    page_surveillance()
