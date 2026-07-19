"""Visual system for the SignalGlass Streamlit application.

The UI deliberately uses one shared stylesheet instead of page-level overrides.  This
keeps Streamlit widgets, Plotly surfaces, and the small HTML presentation components
visually coherent while still allowing the native widgets to remain keyboard usable.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class ThemeTokens:
    """Immutable SignalGlass design tokens shared by CSS and Plotly."""

    canvas: str = "#06111d"
    canvas_deep: str = "#040b13"
    surface: str = "#0a1725"
    surface_raised: str = "#0e1e2f"
    border: str = "#26384a"
    border_soft: str = "rgba(139, 161, 184, 0.16)"
    text: str = "#f4f7fb"
    text_secondary: str = "#aab6c6"
    text_muted: str = "#738297"
    cobalt: str = "#1687ff"
    cobalt_soft: str = "rgba(22, 135, 255, 0.16)"
    positive: str = "#51d47a"
    negative: str = "#ff524a"
    warning: str = "#ffb951"
    purple: str = "#ad70e8"
    radius: str = "12px"
    radius_small: str = "8px"


TOKENS = ThemeTokens()


def plotly_layout_defaults(*, height: int | None = None) -> dict[str, object]:
    """Return a fresh Plotly layout dictionary matching the application theme."""

    layout: dict[str, object] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            "color": TOKENS.text_secondary,
            "size": 12,
        },
        "margin": {"l": 42, "r": 28, "t": 18, "b": 38},
        "hoverlabel": {
            "bgcolor": TOKENS.surface_raised,
            "bordercolor": TOKENS.border,
            "font": {"color": TOKENS.text},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": TOKENS.text_secondary},
        },
        "hovermode": "x unified",
    }
    if height is not None:
        layout["height"] = height
    return layout


def apply_theme() -> None:
    """Install SignalGlass' responsive, accessible Streamlit stylesheet."""

    st.markdown(_STYLES, unsafe_allow_html=True)


inject_theme = apply_theme


_STYLES = f"""
<style>
:root {{
  --sg-canvas: {TOKENS.canvas};
  --sg-canvas-deep: {TOKENS.canvas_deep};
  --sg-surface: {TOKENS.surface};
  --sg-surface-raised: {TOKENS.surface_raised};
  --sg-border: {TOKENS.border};
  --sg-border-soft: {TOKENS.border_soft};
  --sg-text: {TOKENS.text};
  --sg-text-secondary: {TOKENS.text_secondary};
  --sg-text-muted: {TOKENS.text_muted};
  --sg-cobalt: {TOKENS.cobalt};
  --sg-cobalt-soft: {TOKENS.cobalt_soft};
  --sg-positive: {TOKENS.positive};
  --sg-negative: {TOKENS.negative};
  --sg-warning: {TOKENS.warning};
  --sg-purple: {TOKENS.purple};
  --sg-radius: {TOKENS.radius};
}}

html {{ color-scheme: dark; }}
body, .stApp {{
  background:
    radial-gradient(circle at 76% -18%, rgba(22,135,255,.07), transparent 35rem),
    linear-gradient(180deg, var(--sg-canvas-deep), var(--sg-canvas));
  color: var(--sg-text);
  font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
}}
.stApp {{ min-height: 100vh; }}
[data-testid="stAppViewContainer"] > .main {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{
  max-width: 1560px;
  padding: .6rem 2.1rem 2rem;
}}
[data-testid="stHeader"] {{
  background: transparent;
}}
[data-testid="stToolbar"], [data-testid="stAppDeployButton"], #MainMenu {{
  visibility: hidden;
}}
footer {{ visibility: hidden; }}

h1, h2, h3, h4, p {{ letter-spacing: -.015em; }}
h1 {{ font-size: clamp(2rem, 3vw, 3rem) !important; font-weight: 650 !important; }}
h2 {{ font-size: 1.2rem !important; font-weight: 650 !important; }}
h3 {{ font-size: 1rem !important; font-weight: 630 !important; }}
p, li {{ color: var(--sg-text-secondary); }}
a {{ color: var(--sg-cobalt); }}

/* Native Streamlit controls */
.stButton > button, .stDownloadButton > button, button[kind] {{
  min-height: 2.65rem;
  border: 1px solid var(--sg-border) !important;
  border-radius: 8px !important;
  background: rgba(8, 19, 31, .82) !important;
  color: var(--sg-text) !important;
  box-shadow: none !important;
  font-size: .86rem !important;
  font-weight: 560 !important;
  transition: border-color 150ms ease, background 150ms ease, transform 150ms ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--sg-cobalt) !important;
  background: var(--sg-cobalt-soft) !important;
  transform: translateY(-1px);
}}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
[data-baseweb="select"] > div:focus-within {{
  outline: 2px solid var(--sg-cobalt) !important;
  outline-offset: 2px;
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(180deg, #147ee9, #0867cd) !important;
  border-color: #2692ff !important;
}}
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-testid="stDateInput"] [data-baseweb="input"] > div {{
  min-height: 2.65rem;
  background: rgba(8,19,31,.88) !important;
  border-color: var(--sg-border) !important;
  border-radius: 8px !important;
}}
[data-testid="stMetric"] {{
  padding: .1rem .25rem;
}}
[data-testid="stMetricLabel"] p {{ color: var(--sg-text-secondary); font-size: .8rem; }}
[data-testid="stMetricValue"] {{ color: var(--sg-text); letter-spacing: -.035em; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* App shell */
.sg-header {{
  min-height: 58px;
  display: grid;
  grid-template-columns: minmax(180px, 1fr) auto minmax(180px, 1fr);
  align-items: center;
  gap: 1.2rem;
  margin: -.55rem -2.1rem 1.15rem;
  padding: 0 2.1rem;
  border-bottom: 1px solid var(--sg-border-soft);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
}}
.sg-brand {{ display:flex; align-items:center; gap:.68rem; color:var(--sg-text); font-size:1.25rem; font-weight:680; }}
.sg-brand-mark {{ color:var(--sg-cobalt); width:30px; height:24px; display:inline-flex; }}
.sg-nav {{ display:flex; gap:2.6rem; height:58px; align-items:center; }}
.sg-nav-item {{ position:relative; color:var(--sg-text-secondary); font-size:.86rem; white-space:nowrap; text-decoration:none; }}
.sg-nav-item.is-active {{ color:var(--sg-text); }}
.sg-nav-item.is-active::after {{ content:""; position:absolute; left:0; right:0; bottom:-20px; height:2px; background:var(--sg-cobalt); }}
.sg-status {{ display:flex; justify-content:flex-end; align-items:center; gap:.6rem; color:var(--sg-text-secondary); font-size:.8rem; }}
.sg-status-dot {{ width:7px; height:7px; border-radius:50%; background:var(--sg-positive); box-shadow:0 0 10px rgba(81,212,122,.45); }}
.sg-mobile-menu {{ display:none; color:var(--sg-text-secondary); font-size:1.45rem; }}

/* Shared surfaces */
.sg-panel {{
  background: linear-gradient(145deg, rgba(11,25,40,.90), rgba(7,17,28,.90));
  border: 1px solid var(--sg-border);
  border-radius: var(--sg-radius);
  padding: 1rem 1.15rem;
  box-shadow: inset 0 1px rgba(255,255,255,.018);
}}
.sg-panel-title {{ color:var(--sg-text); font-size:1rem; font-weight:640; margin:0 0 .9rem; }}
.sg-section-heading {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; border-bottom:1px solid var(--sg-border-soft); padding-bottom:.65rem; margin-bottom:.4rem; }}
.sg-section-heading h2 {{ margin:0 !important; }}
.sg-kicker {{ color:var(--sg-text-secondary); font-size:.77rem; text-transform:uppercase; letter-spacing:.08em; }}
.sg-rule {{ border:0; border-top:1px solid var(--sg-border-soft); margin:.65rem 0; }}
.sg-positive {{ color:var(--sg-positive) !important; }}
.sg-negative {{ color:var(--sg-negative) !important; }}
.sg-muted {{ color:var(--sg-text-muted) !important; }}
.sg-link {{ color:var(--sg-cobalt); font-size:.8rem; }}

.sg-hero {{ display:flex; align-items:end; justify-content:space-between; gap:2rem; padding:.35rem .1rem 1.05rem; }}
.sg-symbol-line {{ display:flex; align-items:baseline; flex-wrap:wrap; gap:.85rem; }}
.sg-symbol {{ margin:0; color:var(--sg-text); font-size:clamp(2rem,3vw,2.75rem); line-height:1; font-weight:680; letter-spacing:.01em; }}
.sg-company {{ color:var(--sg-text-secondary); font-size:.98rem; }}
.sg-price {{ color:var(--sg-text); font-size:clamp(2.25rem,3.4vw,3rem); line-height:1.05; font-weight:650; margin:.7rem 0 .2rem; letter-spacing:.01em; }}
.sg-change {{ color:var(--sg-positive); font-size:1.15rem; font-weight:620; }}
.sg-period {{ display:flex; align-items:center; gap:.15rem; padding:.25rem; border:1px solid var(--sg-border); border-radius:9px; background:rgba(5,13,22,.5); }}
.sg-period span {{ min-width:3.2rem; text-align:center; padding:.55rem .75rem; border-radius:6px; color:var(--sg-text-secondary); font-size:.84rem; }}
.sg-period .is-active {{ color:var(--sg-text); background:var(--sg-cobalt-soft); border:1px solid var(--sg-cobalt); }}

.sg-metrics {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); margin:.2rem 0 1rem; }}
.sg-metric {{ text-align:center; padding:.15rem 1rem; border-left:1px solid var(--sg-border); }}
.sg-metric:first-child {{ border-left:0; }}
.sg-metric-label {{ color:var(--sg-text-secondary); font-size:.78rem; margin-bottom:.28rem; }}
.sg-metric-value {{ color:var(--sg-text); font-size:1.65rem; font-weight:620; letter-spacing:-.025em; }}

.sg-evidence {{ display:grid; grid-template-columns:58px minmax(0,1fr) auto; gap:.78rem; align-items:start; padding:.75rem 0; border-bottom:1px solid var(--sg-border-soft); }}
.sg-evidence:last-child {{ border-bottom:0; }}
.sg-source-icon {{ width:50px; height:50px; display:grid; place-items:center; border:1px solid var(--sg-border); border-radius:9px; color:var(--sg-text); font-size:1.3rem; font-weight:700; background:rgba(4,11,19,.65); }}
.sg-source-icon.reuters {{ color:#ff7900; font-size:1rem; letter-spacing:-.05em; }}
.sg-evidence-source, .sg-evidence-date {{ color:var(--sg-text-muted); font-size:.72rem; }}
.sg-evidence-title {{ color:var(--sg-text); font-size:.88rem; line-height:1.3; margin:.18rem 0; }}
.sg-evidence-date {{ text-align:right; white-space:nowrap; }}

.sg-list-row {{ display:grid; grid-template-columns:minmax(120px,1.1fr) minmax(84px,.55fr) minmax(80px,.55fr); gap:.65rem; align-items:center; padding:.72rem .25rem; border-bottom:1px solid var(--sg-border-soft); }}
.sg-list-row:last-child {{ border-bottom:0; }}
.sg-list-title {{ color:var(--sg-text); font-size:.88rem; font-weight:570; }}
.sg-list-subtitle {{ color:var(--sg-text-muted); font-size:.72rem; margin-top:.18rem; }}
.sg-tag {{ display:inline-block; justify-self:end; border:1px solid #2a7950; color:var(--sg-positive); background:rgba(81,212,122,.07); border-radius:5px; padding:.22rem .6rem; font-size:.69rem; }}
.sg-tag.neutral {{ color:#9ccaff; border-color:#315a85; background:rgba(22,135,255,.07); }}

.sg-lab-heading {{ display:flex; justify-content:space-between; align-items:start; gap:1.5rem; margin:.35rem .25rem .8rem; }}
.sg-lab-heading h1 {{ margin:0 0 .15rem !important; }}
.sg-lab-heading p {{ margin:0; font-size:.86rem; }}
.sg-scorebar {{ display:grid; grid-template-columns:1.15fr repeat(4,1fr); align-items:center; border:1px solid var(--sg-border); border-radius:var(--sg-radius); margin:.8rem 0; padding:1rem 0; background:linear-gradient(145deg,rgba(11,25,40,.88),rgba(7,17,28,.78)); }}
.sg-score {{ padding:0 1.8rem; border-left:1px solid var(--sg-border-soft); }}
.sg-score:first-child {{ border-left:0; }}
.sg-score-label {{ color:var(--sg-text-secondary); font-size:.75rem; }}
.sg-score-value {{ color:var(--sg-text); font-size:1.5rem; font-weight:630; margin-top:.25rem; }}
.sg-score:first-child .sg-score-value {{ color:var(--sg-positive); font-size:2rem; }}

.sg-driver {{ display:grid; grid-template-columns:48px minmax(0,1fr) 64px; gap:.8rem; align-items:center; padding:.72rem 0; border-bottom:1px solid var(--sg-border-soft); }}
.sg-driver:last-child {{ border-bottom:0; }}
.sg-driver-icon {{ width:46px; height:46px; display:grid; place-items:center; border:1px solid var(--sg-border); border-radius:9px; color:var(--sg-cobalt); font-size:1.1rem; }}
.sg-driver-name {{ color:var(--sg-text); font-size:.84rem; font-weight:600; }}
.sg-driver-copy {{ color:var(--sg-text-secondary); font-size:.72rem; line-height:1.3; }}
.sg-driver-value {{ font-size:.9rem; font-weight:630; text-align:right; }}
.sg-process {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; }}
.sg-process-step {{ display:grid; grid-template-columns:40px 1fr; gap:.65rem; align-items:start; }}
.sg-process-number {{ width:38px; height:38px; display:grid; place-items:center; border:1px solid var(--sg-border); border-radius:8px; color:var(--sg-cobalt); font-size:.75rem; }}
.sg-process-title {{ color:var(--sg-text); font-size:.76rem; font-weight:600; }}
.sg-process-copy {{ color:var(--sg-text-muted); font-size:.67rem; line-height:1.45; margin-top:.15rem; }}
.sg-footer {{ color:var(--sg-text-muted); text-align:center; font-size:.78rem; padding:1.1rem 0 .35rem; letter-spacing:.01em; }}

/* Plotly, alerts, dataframe */
[data-testid="stPlotlyChart"] {{ border-radius:8px; overflow:hidden; }}
[data-testid="stAlert"] {{ background:rgba(11,25,40,.8); border:1px solid var(--sg-border); color:var(--sg-text-secondary); }}
[data-testid="stDataFrame"] {{ border:1px solid var(--sg-border); border-radius:8px; overflow:hidden; }}

@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ transition:none !important; scroll-behavior:auto !important; }} }}
@media (max-width: 1100px) {{
  [data-testid="stMainBlockContainer"] {{ padding:.4rem 1rem 5.5rem; }}
  div[data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; gap:.75rem; }}
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
    flex:1 1 min(100%, 420px) !important;
    width:auto !important;
    min-width:min(100%, 320px) !important;
  }}
  .sg-header {{ grid-template-columns:1fr auto auto; margin:-.4rem -1rem .9rem; padding:0 1rem; backdrop-filter:none; -webkit-backdrop-filter:none; }}
  .sg-nav {{ position:fixed; z-index:999; left:0; right:0; bottom:0; height:72px; display:grid; grid-template-columns:repeat(4,1fr); gap:0; border-top:1px solid var(--sg-border); background:rgba(5,14,24,.96); backdrop-filter:blur(20px); }}
  .sg-nav-item {{ text-align:center; font-size:.69rem; padding-top:1.85rem; }}
  .sg-nav-item.is-active::after {{ top:0; bottom:auto; left:30%; right:30%; }}
  .sg-status {{ font-size:.76rem; }}
  .sg-updated {{ display:none; }}
  .sg-mobile-menu {{ display:block; }}
  .sg-hero {{ align-items:start; }}
  .sg-metrics {{ margin-top:.6rem; }}
  .sg-metric {{ padding:.1rem .45rem; }}
  .sg-metric-value {{ font-size:1.35rem; }}
  .sg-scorebar {{ grid-template-columns:repeat(2,1fr); gap:.9rem 0; }}
  .sg-score:nth-child(3) {{ border-left:0; }}
  .sg-process {{ grid-template-columns:repeat(2,1fr); }}
}}
@media (max-width: 600px) {{
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{ flex-basis:100% !important; min-width:100% !important; }}
  .sg-brand {{ font-size:1.12rem; }}
  .sg-brand-mark {{ width:27px; }}
  .sg-header {{ min-height:62px; }}
  .sg-hero {{ display:grid; grid-template-columns:1fr auto; gap:1rem; }}
  .sg-period {{ align-self:end; }}
  .sg-period span {{ min-width:2.35rem; padding:.48rem .45rem; }}
  .sg-price {{ font-size:2.35rem; }}
  .sg-metrics {{ grid-template-columns:repeat(4, minmax(0,1fr)); overflow:hidden; }}
  .sg-metric-label {{ font-size:.66rem; }}
  .sg-metric-value {{ font-size:1.1rem; }}
  .sg-panel {{ padding:.85rem .8rem; }}
  .sg-evidence {{ grid-template-columns:48px 1fr; }}
  .sg-source-icon {{ width:44px; height:44px; }}
  .sg-evidence-date {{ grid-column:2; text-align:left; }}
  .sg-scorebar {{ grid-template-columns:repeat(2,1fr); }}
  .sg-score {{ padding:0 1rem; }}
  .sg-process {{ grid-template-columns:1fr; }}
  .sg-list-row {{ grid-template-columns:1fr auto; }}
  .sg-list-row > :nth-child(2) {{ display:none; }}
}}
</style>
"""
