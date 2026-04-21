import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import os
import io
import tempfile
import hashlib
import joblib

from utils import load_dataset, get_dataset_info
from ml_engine import run_pipeline
from llm_report import generate_llm_report, generate_local_report
from pdf_export import generate_pdf
from evaluation import (
    build_confusion_matrix_chart,
    build_residuals_chart,
    build_learning_curve_chart,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataDoctor AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

PDF_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "datadoctor_pdfs")

# ─────────────────────────────────────────────
#  CUSTOM CSS  — Dark Cyber Dashboard Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap&font-display=swap');

:root {
    --bg-deep:      #020b18;
    --bg-card:      #060f1f;
    --bg-panel:     #0a1628;
    --accent-cyan:  #00d4ff;
    --accent-blue:  #0066ff;
    --accent-green: #00ff88;
    --accent-purple:#a855f7;
    --text-primary: #e0f4ff;
    --text-muted:   #5a7a9a;
    --border:       #0d2a4a;
    --glow-cyan:    0 0 20px rgba(0,212,255,0.3);
    --glow-blue:    0 0 20px rgba(0,102,255,0.3);
    --glow-green:   0 0 20px rgba(0,255,136,0.3);
}

.stApp {
    background: var(--bg-deep) !important;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0,102,255,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(0,212,255,0.05) 0%, transparent 50%),
        linear-gradient(180deg, #020b18 0%, #030e1a 100%) !important;
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] li { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--accent-cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

.block-container { padding: 1.5rem 2rem !important; }

.stApp > section > div > div > div > h1,
[data-testid="stSidebar"] h1 {
    font-family: 'Orbitron', monospace !important;
    font-size: 2.2rem !important;
    font-weight: 900 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stVerticalBlock"] > div > div > h2 {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.4rem !important;
    color: var(--accent-cyan) !important;
}
[data-testid="stVerticalBlock"] > div > div > h3 {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.1rem !important;
    color: var(--accent-blue) !important;
}

[data-testid="metric-container"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent-cyan) !important;
    border-radius: 0 4px 4px 0 !important;
    padding: 1rem !important;
    box-shadow: var(--glow-cyan) !important;
}
[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.8rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: #000 !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.6rem 2rem !important;
    cursor: pointer !important;
    transition: box-shadow 0.3s ease, transform 0.2s ease !important;
    box-shadow: var(--glow-blue) !important;
}
.stButton > button:hover {
    box-shadow: 0 0 30px rgba(0,212,255,0.5) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

[data-testid="stFileUploader"] {
    background: var(--bg-panel) !important;
    border: 1px dashed var(--accent-cyan) !important;
    border-radius: 4px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
}

[data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
    column-gap: 0.5rem !important;
}
[data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.2rem !important;
    border-radius: 4px 4px 0 0 !important;
}
[data-baseweb="tab-list"] [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
    background: var(--bg-panel) !important;
}

[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important; }
[data-testid="stProgress"] { background: var(--bg-panel) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; }
[data-testid="stAlert"] { background: var(--bg-panel) !important; border-radius: 4px !important; }
code, pre { font-family: 'Share Tech Mono', monospace !important; background: var(--bg-panel) !important; color: var(--accent-green) !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

.dd-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    min-height: 60px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.dd-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-green));
}
.dd-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.dd-badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid var(--accent-cyan);
    color: var(--accent-cyan);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    margin: 0.2rem;
}
.dd-badge-green { background: rgba(0,255,136,0.1); border-color: var(--accent-green); color: var(--accent-green); }
.dd-badge-purple { background: rgba(168,85,247,0.1); border-color: var(--accent-purple); color: var(--accent-purple); }
.best-model-banner {
    background: linear-gradient(135deg, rgba(0,102,255,0.15), rgba(0,212,255,0.15));
    border: 1px solid var(--accent-cyan);
    border-radius: 6px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--glow-cyan);
    margin: 1rem 0;
}
.best-model-name {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    color: var(--accent-cyan);
    text-shadow: 0 0 20px rgba(0,212,255,0.5);
}
.score-display {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent;
}
.scan-line {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    margin: 1rem 0;
    animation: scan 2s ease-in-out infinite alternate;
}
@keyframes scan { from { opacity: 0.3; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) {
    .scan-line { animation: none; opacity: 0.6; }
    .stButton > button { transition: none !important; }
}
[data-testid="stExpander"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 1rem;">
    <div style="font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:0.3em; color:#5a7a9a; text-transform:uppercase; margin-bottom:0.3rem;">
        ◈ AUTOMATED MACHINE LEARNING PLATFORM ◈
    </div>
    <h1 style="font-family:'Orbitron',monospace; font-size:2.8rem; font-weight:900; margin:0;
               background:linear-gradient(135deg, #0066ff, #00d4ff, #00ff88);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
        🩺 DataDoctor AI
    </h1>
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#5a7a9a; margin-top:0.3rem; letter-spacing:0.1em;">
        UPLOAD → DETECT → PREPROCESS → TRAIN → ANALYZE
    </div>
</div>
<div class="scan-line"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.9rem; color:#00d4ff;
                letter-spacing:0.15em; padding:0.5rem 0; border-bottom:1px solid #0d2a4a; margin-bottom:1rem;">
        ⚙ CONTROL PANEL
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset",
        type=["csv", "xlsx", "xls"],
        help="Supported: CSV, Excel (.xlsx, .xls)",
    )

    cv_folds = st.slider("🔁 K-Fold CV Splits", min_value=3, max_value=10, value=5, step=1)

    task_override = st.selectbox(
        "🎯 Task Override",
        options=["Auto-detect", "Classification", "Regression"],
        index=0,
        help="Override automatic task detection if needed.",
    )
    task_forced = None
    if task_override == "Classification":
        task_forced = "classification"
    elif task_override == "Regression":
        task_forced = "regression"

    mode = st.selectbox(
        "⚡ AutoML Mode",
        options=["Fast (3-4 models)", "Full (all models)"],
        index=0,
        help="Fast = quicker run; Full = more thorough comparison",
    )
    mode_key = "fast" if "Fast" in mode else "full"

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#5a7a9a; margin-top:1rem; padding:0.8rem; background:#060f1f; border-radius:4px; border:1px solid #0d2a4a;">
        ◈ Auto-detects Classification / Regression<br>
        ◈ GridSearchCV — leak-free Pipeline<br>
        ◈ Imbalance handling: SMOTE / RandomOverSampler<br>
        ◈ XGBoost + LightGBM supported<br>
        ◈ LLM Report + PDF Export<br>
        ◈ SHAP · Confusion Matrix · Learning Curve
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.6rem; color:#0d2a4a; text-align:center; margin-top:2rem; letter-spacing:0.2em;">
        DATADOCTOR AI v3.0 ◈ 2025
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CACHE KEY helper
# ─────────────────────────────────────────────
def _make_cache_key(file_name: str, file_size: int, target: str, cv: int, task_f, mode: str) -> str:
    raw = f"{file_name}|{file_size}|{target}|{cv}|{task_f}|{mode}"
    return hashlib.md5(raw.encode()).hexdigest()


# ─────────────────────────────────────────────
#  STEP 1 — DATASET LOADED
# ─────────────────────────────────────────────
if uploaded_file is not None:
    try:
        df = load_dataset(uploaded_file)
        info = get_dataset_info(df)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    col4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    with st.expander("📊 Dataset Preview", expanded=False):
        tab1, tab2 = st.tabs(["DATA TABLE", "COLUMN INFO"])
        with tab1:
            st.dataframe(df.head(50), use_container_width=True, height=300)
        with tab2:
            info_df = pd.DataFrame({
                "Column": list(info["dtypes"].keys()),
                "Type": list(info["dtypes"].values()),
                "Missing": [info["missing"][c] for c in info["dtypes"]],
                "Missing %": [f"{info['missing_pct'][c]:.1f}%" for c in info["dtypes"]],
                "Unique": [info["nunique"][c] for c in info["dtypes"]],
            })
            st.dataframe(info_df, use_container_width=True)

    st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    #  STEP 2 — TARGET SELECTION
    # ─────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#00d4ff;
                letter-spacing:0.15em; margin:1rem 0 0.5rem;">
        ◈ SELECT TARGET VARIABLE
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 3])
    with col_a:
        target_col = st.selectbox(
            "Target Column",
            options=list(df.columns),
            index=len(df.columns) - 1,
            label_visibility="collapsed",
        )

    with col_b:
        if target_col:
            from task_inference import infer_task, get_task_explanation, is_ambiguous_task
            auto_task = infer_task(df, target_col)
            effective_task = task_forced if task_forced else auto_task
            task_color = "#00d4ff" if effective_task == "classification" else "#00ff88"
            task_icon = "🏷️" if effective_task == "classification" else "📈"
            override_note = " (overridden)" if task_forced else ""
            st.markdown(f"""
            <div style="background:#060f1f; border:1px solid {task_color}; border-radius:4px;
                        padding:0.6rem 1rem; display:flex; align-items:center; gap:0.5rem; margin-top:0.2rem;">
                <span style="font-size:1.2rem;">{task_icon}</span>
                <div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#5a7a9a;">DETECTED TASK{override_note.upper()}</div>
                    <div style="font-family:'Orbitron',monospace; font-size:0.9rem; color:{task_color}; font-weight:700;">
                        {effective_task.upper()}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not task_forced and is_ambiguous_task(df, target_col):
                st.warning(
                    "⚠️ **Ambiguous target detected.** The auto-detection result may be uncertain. "
                    "Use the **Task Override** in the sidebar to force Classification or Regression if needed."
                )

    # ─────────────────────────────────────────────
    #  Validate dataset size
    # ─────────────────────────────────────────────
    min_rows_required = cv_folds * 2
    if len(df) < min_rows_required:
        st.error(
            f"❌ **Dataset too small**: {len(df)} rows detected. "
            f"With {cv_folds} CV folds, you need at least **{min_rows_required} rows**. "
            f"Reduce CV folds in the sidebar or use a larger dataset."
        )
        st.stop()

    # ─────────────────────────────────────────────
    #  STEP 3 — LAUNCH TRAINING  (with cache)
    # ─────────────────────────────────────────────
    st.markdown("")
    launch_col, cache_col, _ = st.columns([2, 2, 3])
    with launch_col:
        run_clicked = st.button("🚀 LAUNCH TRAINING")

    # Compute cache key for the current config
    cache_key = _make_cache_key(
        uploaded_file.name,
        uploaded_file.size,
        target_col,
        cv_folds,
        task_forced,
        mode_key,
    )

    # Show cache hit info
    cached_key = st.session_state.get("cache_key")
    if cached_key == cache_key and "pipeline_result" in st.session_state:
        with cache_col:
            st.markdown(
                "<span style='font-family:Share Tech Mono,monospace; font-size:0.75rem; "
                "color:#00ff88;'>⚡ Cached results loaded</span>",
                unsafe_allow_html=True,
            )

    if run_clicked:
        # Check if we can reuse cached result
        if cached_key == cache_key and "pipeline_result" in st.session_state:
            st.info("⚡ Same configuration detected — using cached results. Change any parameter to re-train.")
        else:
            st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#00d4ff; letter-spacing:0.15em; margin:1rem 0 0.5rem;">
                ◈ PIPELINE EXECUTING
            </div>
            """, unsafe_allow_html=True)

            # ── Granular progress bar ──
            progress_bar = st.progress(0)
            status_text  = st.empty()

            def _status(msg: str, pct: int):
                status_text.markdown(
                    f"<span style='font-family:Share Tech Mono,monospace; font-size:0.85rem; color:#00d4ff;'>{msg}</span>",
                    unsafe_allow_html=True,
                )
                progress_bar.progress(pct)

            _status("🔬 Step 1/5 — Detecting task & validating data…", 5)
            time.sleep(0.2)
            _status("⚙️ Step 2/5 — Building preprocessing pipeline…", 15)
            time.sleep(0.2)
            _status("✂️ Step 3/5 — Train / holdout split…", 25)
            time.sleep(0.2)
            _status("🏋️ Step 4/5 — GridSearchCV + cross-validation (this may take a moment)…", 35)

            try:
                pipeline_result = run_pipeline(
                    df,
                    target_col,
                    cv_folds=cv_folds,
                    task_override=task_forced,
                    mode=mode_key,
                )
            except Exception as e:
                st.error(f"❌ Pipeline error: {e}")
                st.exception(e)
                st.stop()

            _status("📊 Step 5/5 — Computing SHAP values & charts…", 85)
            time.sleep(0.3)

            progress_bar.progress(100)
            status_text.markdown(
                "<span style='font-family:Share Tech Mono,monospace; font-size:0.85rem; color:#00ff88;'>✅ Pipeline complete!</span>",
                unsafe_allow_html=True,
            )
            time.sleep(0.8)
            progress_bar.empty()
            status_text.empty()

            st.session_state["pipeline_result"] = pipeline_result
            st.session_state["dataset_info"]    = info
            st.session_state["df"]              = df
            st.session_state["cache_key"]       = cache_key
            # Clear stale report when re-running
            st.session_state.pop("llm_report", None)
            st.session_state.pop("llm_filename", None)
            st.session_state.pop("pdf_bytes", None)

    # ─────────────────────────────────────────────
    #  STEP 4 — RESULTS
    # ─────────────────────────────────────────────
    if "pipeline_result" in st.session_state:
        result = st.session_state["pipeline_result"]
        best   = result["best_model"]
        task   = result["task"]
        metric_label = "Accuracy" if task == "classification" else "R² Score"

        st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#00d4ff; letter-spacing:0.15em; margin:1rem 0 0.5rem;">
            ◈ RESULTS
        </div>
        """, unsafe_allow_html=True)

        # ── Best model banner ──
        st.markdown(f"""
        <div class="best-model-banner">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#5a7a9a; letter-spacing:0.2em; margin-bottom:0.3rem;">
                ◈ CHAMPION MODEL ◈
            </div>
            <div class="best-model-name">🏆 {best['model']}</div>
            <div class="score-display">{best['cv_mean']:.4f}</div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#5a7a9a;">{metric_label} ± {best['cv_std']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Quick metrics ──
        cols = st.columns(len(result["results"]))
        for i, (col, r) in enumerate(zip(cols, result["results"])):
            with col:
                icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                st.metric(
                    label=f"{icon} {r['model']}",
                    value=f"{r['cv_mean']:.4f}",
                    delta=f"±{r['cv_std']:.4f}",
                )

        # ── Tabs ──
        tabs = st.tabs([
            "📊 GRAPHIQUES",
            "🔬 SHAP & HOLDOUT",
            "📉 DIAGNOSTICS",
            "🔮 PRÉDICTION",
            "🔍 DÉTAILS MODÈLES",
            "🧬 PRÉTRAITEMENT",
            "📝 RAPPORT LLM",
        ])

        # ── Tab 1: Charts ──
        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                if result["bar_chart"]:
                    st.plotly_chart(result["bar_chart"], use_container_width=True)
            with c2:
                if result["violin_chart"]:
                    st.plotly_chart(result["violin_chart"], use_container_width=True)

            if result["radar_chart"]:
                st.plotly_chart(result["radar_chart"], use_container_width=True)

            st.markdown("**Distribution de la variable cible**")
            if task == "classification":
                vc = st.session_state["df"][target_col].value_counts()
                fig_dist = go.Figure(go.Bar(
                    x=vc.index.astype(str), y=vc.values,
                    marker=dict(color=["#00d4ff", "#0066ff", "#00ff88", "#a855f7", "#ff6b6b"][:len(vc)]),
                    text=vc.values, textposition="outside", textfont=dict(color="#e0f4ff"),
                ))
            else:
                fig_dist = go.Figure(go.Histogram(
                    x=st.session_state["df"][target_col].dropna(),
                    marker_color="#00d4ff", opacity=0.8, nbinsx=40,
                ))
            fig_dist.update_layout(
                plot_bgcolor="#060f1f", paper_bgcolor="#020b18",
                font=dict(color="#e0f4ff"),
                xaxis=dict(gridcolor="#0d2a4a"), yaxis=dict(gridcolor="#0d2a4a"),
                height=300, margin=dict(t=20, b=30, l=40, r=20),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("**Matrice de corrélation (features numériques)**")
            _df_corr = st.session_state["df"].select_dtypes(include=["int64", "float64", "int32", "float32"])
            if _df_corr.shape[1] >= 2:
                _corr = _df_corr.corr().round(2)
                _cols_corr = list(_corr.columns)
                _z    = _corr.values.tolist()
                _text = [[str(v) for v in row] for row in _z]
                fig_corr = go.Figure(go.Heatmap(
                    z=_z, x=_cols_corr, y=_cols_corr,
                    text=_text, texttemplate="%{text}",
                    textfont=dict(size=9, color="#e0f4ff"),
                    colorscale=[[0.0, "#0066ff"], [0.5, "#050d1a"], [1.0, "#00ff88"]],
                    zmin=-1, zmax=1,
                    colorbar=dict(tickfont=dict(color="#e0f4ff"), title=dict(text="r", font=dict(color="#00d4ff"))),
                ))
                fig_corr.update_layout(
                    plot_bgcolor="#060f1f", paper_bgcolor="#020b18",
                    font=dict(color="#e0f4ff"),
                    xaxis=dict(tickangle=-40, tickfont=dict(size=9, color="#e0f4ff"), gridcolor="#0d2a4a"),
                    yaxis=dict(tickfont=dict(size=9, color="#e0f4ff"), gridcolor="#0d2a4a"),
                    margin=dict(t=20, b=80, l=80, r=20),
                    height=max(350, 40 * len(_cols_corr)),
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.caption("Not enough numeric columns for a correlation matrix.")

        # ── Tab 2: SHAP & Holdout ──
        with tabs[1]:
            shap_data = result.get("shap_data", {})
            holdout   = best.get("holdout", {})

            st.markdown("### 🔬 Holdout Test Performance")
            if holdout and not holdout.get("error"):
                if task == "classification":
                    h_cols = st.columns(3)
                    h_cols[0].metric("Accuracy (holdout)", f"{holdout.get('accuracy', 0):.4f}")
                    h_cols[1].metric("F1-Score (weighted)", f"{holdout.get('f1_weighted', 0):.4f}")
                    if "roc_auc" in holdout:
                        h_cols[2].metric("AUC-ROC", f"{holdout['roc_auc']:.4f}")
                else:
                    h_cols = st.columns(3)
                    h_cols[0].metric("R² (holdout)", f"{holdout.get('r2', 0):.4f}")
                    h_cols[1].metric("MAE", f"{holdout.get('mae', 0):.4f}")
                    h_cols[2].metric("RMSE", f"{holdout.get('rmse', 0):.4f}")
            else:
                st.warning("Holdout evaluation not available.")

            st.markdown("---")
            st.markdown("### 🧠 SHAP Feature Importance")
            if shap_data.get("available"):
                shap_fig = go.Figure(go.Bar(
                    x=shap_data["mean_abs_shap"][::-1],
                    y=shap_data["feature_names"][::-1],
                    orientation="h",
                    marker=dict(
                        color=shap_data["mean_abs_shap"][::-1],
                        colorscale=[[0, "#0066ff"], [0.5, "#00d4ff"], [1, "#00ff88"]],
                        showscale=True,
                        colorbar=dict(title="SHAP", tickfont=dict(color="#e0f4ff")),
                    ),
                    text=[f"{v:.4f}" for v in shap_data["mean_abs_shap"][::-1]],
                    textposition="outside",
                    textfont=dict(color="#e0f4ff", size=10),
                ))
                shap_fig.update_layout(
                    plot_bgcolor="#060f1f", paper_bgcolor="#020b18",
                    font=dict(color="#e0f4ff"),
                    xaxis=dict(gridcolor="#0d2a4a", title="Mean |SHAP value|"),
                    yaxis=dict(gridcolor="#0d2a4a", tickfont=dict(size=10)),
                    height=max(300, 28 * len(shap_data["feature_names"])),
                    margin=dict(t=20, b=40, l=200, r=80),
                    title=dict(text=f"Top Features — {best['model']}", font=dict(color="#00d4ff", size=15)),
                )
                st.plotly_chart(shap_fig, use_container_width=True)

                shap_df = pd.DataFrame({
                    "Feature": shap_data["feature_names"],
                    "Mean |SHAP|": [f"{v:.5f}" for v in shap_data["mean_abs_shap"]],
                })
                st.dataframe(shap_df, use_container_width=True, height=250)
            else:
                st.info(f"SHAP analysis not available: {shap_data.get('error', 'module not installed')}. Run `pip install shap` to enable.")

        # ── Tab 3: Diagnostics (NEW) ──
        with tabs[2]:
            st.markdown("### 📉 Model Diagnostics")
            estimator = best.get("estimator")

            if estimator is None:
                st.warning("Best estimator not available.")
            else:
                X_test = result.get("X_test")
                y_test = result.get("y_test")

                # ── Confusion Matrix (classification) ──
                if task == "classification":
                    st.markdown("#### 🟦 Confusion Matrix")
                    label_encoder = result.get("label_encoder")
                    cm_fig = build_confusion_matrix_chart(estimator, X_test, y_test, label_encoder)
                    if cm_fig:
                        st.plotly_chart(cm_fig, use_container_width=True)
                    else:
                        st.warning("Could not build confusion matrix.")

                # ── Residuals (regression) ──
                else:
                    st.markdown("#### 📈 Predicted vs Actual & Residuals")
                    scatter_fig, hist_fig = build_residuals_chart(estimator, X_test, y_test)
                    if scatter_fig:
                        r1, r2 = st.columns(2)
                        with r1:
                            st.plotly_chart(scatter_fig, use_container_width=True)
                        with r2:
                            st.plotly_chart(hist_fig, use_container_width=True)
                    else:
                        st.warning("Could not build residuals chart.")

                st.markdown("---")
                # ── Learning Curve (both tasks) ──
                st.markdown("#### 📚 Learning Curve")
                st.caption("Shows whether the model benefits from more data (high bias vs high variance diagnosis).")

                # Use combined train+test for learning curve (re-assemble from pipeline result)
                from preprocessing import auto_preprocess
                try:
                    (X_raw_full, y_full, _, _, _, _, _, _, _, _) = auto_preprocess(
                        st.session_state["df"], target_col, task
                    )
                    lc_fig = build_learning_curve_chart(
                        estimator, X_raw_full, y_full, task, cv=result.get("cv_folds", 5)
                    )
                    if lc_fig:
                        st.plotly_chart(lc_fig, use_container_width=True)
                        # Interpretation hint
                        st.markdown("""
                        <div class="dd-card" style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; line-height:1.8;">
                            <div class="dd-card-title">HOW TO READ THIS CHART</div>
                            <span style="color:#00d4ff;">►</span> <b>Training ≫ Validation</b> with a large gap → <span style="color:#ff6b6b;">Overfitting</span> — add regularisation or more data<br>
                            <span style="color:#00d4ff;">►</span> <b>Both curves low & close</b> → <span style="color:#ffd93d;">Underfitting</span> — try a more complex model or more features<br>
                            <span style="color:#00d4ff;">►</span> <b>Curves converge at high score</b> → <span style="color:#00ff88;">Good fit</span> — model is ready for deployment
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Could not compute learning curve.")
                except Exception as e:
                    st.warning(f"Learning curve error: {e}")

        # ── Tab 4: Prediction (NEW) ──
        with tabs[3]:
            st.markdown("### 🔮 Predict on New Data")
            st.caption("Fill in the feature values below to get a prediction from the best model.")

            estimator = best.get("estimator")
            if estimator is None:
                st.warning("Best estimator not available for prediction.")
            else:
                num_cols  = result.get("numeric_cols", [])
                cat_cols  = result.get("categorical_cols", [])
                all_feat  = num_cols + cat_cols

                if not all_feat:
                    st.warning("No feature columns found.")
                else:
                    with st.form("prediction_form"):
                        st.markdown("**Numeric Features**")
                        num_inputs = {}
                        if num_cols:
                            n_per_row = 3
                            for i in range(0, len(num_cols), n_per_row):
                                row_cols = st.columns(n_per_row)
                                for j, col_name in enumerate(num_cols[i:i+n_per_row]):
                                    col_data = st.session_state["df"][col_name].dropna()
                                    default_val = float(col_data.median()) if len(col_data) > 0 else 0.0
                                    num_inputs[col_name] = row_cols[j].number_input(
                                        col_name, value=default_val,
                                        format="%.4f", key=f"pred_num_{col_name}"
                                    )

                        if cat_cols:
                            st.markdown("**Categorical Features**")
                            cat_inputs = {}
                            c_per_row = 3
                            for i in range(0, len(cat_cols), c_per_row):
                                row_cols = st.columns(c_per_row)
                                for j, col_name in enumerate(cat_cols[i:i+c_per_row]):
                                    options = sorted(st.session_state["df"][col_name].dropna().unique().astype(str).tolist())
                                    cat_inputs[col_name] = row_cols[j].selectbox(
                                        col_name, options=options, key=f"pred_cat_{col_name}"
                                    )
                        else:
                            cat_inputs = {}

                        predict_btn = st.form_submit_button("🔮 PREDICT", use_container_width=False)

                    if predict_btn:
                        try:
                            input_data = {**num_inputs, **cat_inputs}
                            input_df = pd.DataFrame([input_data])
                            # Ensure column order matches training
                            input_df = input_df.reindex(columns=all_feat)

                            prediction = estimator.predict(input_df)[0]
                            label_encoder = result.get("label_encoder")

                            if task == "classification" and label_encoder is not None:
                                pred_label = label_encoder.inverse_transform([int(prediction)])[0]
                            else:
                                pred_label = prediction

                            # Confidence for classification
                            confidence_str = ""
                            if task == "classification" and hasattr(estimator, "predict_proba"):
                                proba = estimator.predict_proba(input_df)[0]
                                max_proba = float(np.max(proba))
                                confidence_str = f"<br><span style='font-family:Share Tech Mono,monospace; font-size:0.85rem; color:#5a7a9a;'>Confidence: <b style='color:#00ff88;'>{max_proba*100:.1f}%</b></span>"

                            st.markdown(f"""
                            <div class="best-model-banner" style="margin-top:1rem;">
                                <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#5a7a9a; letter-spacing:0.2em;">
                                    ◈ PREDICTION RESULT ◈
                                </div>
                                <div class="best-model-name" style="font-size:2.5rem; margin-top:0.5rem;">
                                    {pred_label}
                                </div>
                                {confidence_str}
                            </div>
                            """, unsafe_allow_html=True)

                            # Proba bar chart for classification
                            if task == "classification" and hasattr(estimator, "predict_proba"):
                                le = result.get("label_encoder")
                                class_labels = [str(c) for c in le.classes_] if le else [str(i) for i in range(len(proba))]
                                prob_fig = go.Figure(go.Bar(
                                    x=class_labels, y=proba,
                                    marker=dict(
                                        color=proba,
                                        colorscale=[[0, "#0066ff"], [1, "#00ff88"]],
                                        showscale=False,
                                    ),
                                    text=[f"{p*100:.1f}%" for p in proba],
                                    textposition="outside",
                                    textfont=dict(color="#e0f4ff"),
                                ))
                                prob_fig.update_layout(
                                    title=dict(text="Class Probabilities", font=dict(color="#00d4ff", size=14)),
                                    plot_bgcolor="#060f1f", paper_bgcolor="#020b18",
                                    font=dict(color="#e0f4ff"),
                                    xaxis=dict(gridcolor="#0d2a4a"),
                                    yaxis=dict(gridcolor="#0d2a4a", range=[0, 1.1], title="Probability"),
                                    height=280, margin=dict(t=40, b=30, l=40, r=20),
                                )
                                st.plotly_chart(prob_fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Prediction error: {e}")
                            st.exception(e)

            # ── Download model (joblib) ──
            st.markdown("---")
            st.markdown("### 💾 Download Trained Model")
            st.caption("Export the best pipeline (preprocessor + model) as a `.pkl` file for deployment.")

            if best.get("estimator") is not None:
                try:
                    buf = io.BytesIO()
                    joblib.dump(best["estimator"], buf)
                    buf.seek(0)
                    model_filename = f"DataDoctor_{best['model'].replace(' ', '_')}_{task}_pipeline.pkl"
                    st.download_button(
                        label="⬇️ DOWNLOAD MODEL (.pkl)",
                        data=buf.getvalue(),
                        file_name=model_filename,
                        mime="application/octet-stream",
                    )
                    st.markdown(
                        f"<span style='font-family:Share Tech Mono,monospace; font-size:0.75rem; color:#5a7a9a;'>"
                        f"Load with: <code>import joblib; model = joblib.load('{model_filename}')</code></span>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Could not serialize model: {e}")

        # ── Tab 5: Model Details ──
        with tabs[4]:
            st.markdown("### 🔍 All Models — Detailed Results")
            for i, r in enumerate(result["results"]):
                if r.get("cv_mean", -999) <= -999:
                    continue
                icon = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                with st.expander(f"{icon} {r['model']} — CV: {r['cv_mean']:.4f} ± {r['cv_std']:.4f}", expanded=(i == 0)):
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Best CV Score", f"{r['best_score']:.4f}")
                    d2.metric("CV Mean", f"{r['cv_mean']:.4f}")
                    d3.metric("CV Std", f"{r['cv_std']:.4f}")

                    if r.get("best_params"):
                        st.markdown("**Optimal Hyperparameters:**")
                        st.json(r["best_params"])

                    if r.get("holdout") and not r["holdout"].get("error"):
                        st.markdown("**Holdout Test Metrics:**")
                        st.json(r["holdout"])

                    if r.get("cv_scores"):
                        cv_fig = go.Figure(go.Box(
                            y=r["cv_scores"], name=r["model"],
                            marker_color="#00d4ff", line_color="#0066ff",
                            boxpoints="all", jitter=0.3, pointpos=-1.8,
                        ))
                        cv_fig.update_layout(
                            plot_bgcolor="#060f1f", paper_bgcolor="#020b18",
                            font=dict(color="#e0f4ff"), height=250,
                            yaxis=dict(gridcolor="#0d2a4a"),
                            margin=dict(t=10, b=10, l=40, r=10),
                        )
                        st.plotly_chart(cv_fig, use_container_width=True)

                    if r.get("error"):
                        st.error(f"Error: {r['error']}")

        # ── Tab 6: Preprocessing ──
        with tabs[5]:
            prep = result["preprocessing"]
            st.markdown(f"""
            <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1rem;">
                <div class="dd-card" style="text-align:center;">
                    <div class="dd-card-title">NUMERIC FEATURES</div>
                    <div style="font-family:'Orbitron',monospace; font-size:2rem; color:#00d4ff;">{prep['numeric_features']}</div>
                </div>
                <div class="dd-card" style="text-align:center;">
                    <div class="dd-card-title">CATEGORICAL FEATURES</div>
                    <div style="font-family:'Orbitron',monospace; font-size:2rem; color:#00ff88;">{prep['categorical_features']}</div>
                </div>
                <div class="dd-card" style="text-align:center;">
                    <div class="dd-card-title">MISSING VALUES TREATED</div>
                    <div style="font-family:'Orbitron',monospace; font-size:2rem; color:#a855f7;">{prep['total_missing_values']}</div>
                </div>
                <div class="dd-card" style="text-align:center;">
                    <div class="dd-card-title">TASK TYPE</div>
                    <div style="font-family:'Orbitron',monospace; font-size:1.2rem; color:#ff6b6b; text-transform:uppercase;">{task}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Task Detection:** {result['task_explanation']}")

            resampling = prep.get("resampling_info", {})
            imbalance_cols = st.columns(4)
            imbalance_cols[0].metric("Imbalance Detected", "YES" if prep.get("imbalance_detected") else "NO")
            imbalance_cols[1].metric("Imbalance Ratio", f"{prep.get('imbalance_ratio', 1.0):.2f}")
            imbalance_cols[2].metric("Resampling Method", resampling.get("method", "none"))
            k_txt = str(resampling.get("k_neighbors")) if resampling.get("k_neighbors") is not None else "—"
            imbalance_cols[3].metric("SMOTE k", k_txt)

            if prep.get("class_dist_before"):
                dist_df = pd.DataFrame({
                    "Class": list(prep["class_dist_before"].keys()),
                    "Count": list(prep["class_dist_before"].values()),
                })
                st.markdown("**Class distribution before resampling:**")
                st.dataframe(dist_df, use_container_width=True, height=180)

            if prep.get("high_card_dropped"):
                st.markdown("**High-cardinality columns dropped:**")
                for c in prep["high_card_dropped"]:
                    st.markdown(f'<span class="dd-badge dd-badge-purple">{c}</span>', unsafe_allow_html=True)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**Numeric Columns:**")
                if prep["numeric_cols"]:
                    for c in prep["numeric_cols"]:
                        st.markdown(f'<span class="dd-badge">{c}</span>', unsafe_allow_html=True)
                else:
                    st.caption("None found")
            with col_p2:
                st.markdown("**Categorical Columns:**")
                if prep["categorical_cols"]:
                    for c in prep["categorical_cols"]:
                        st.markdown(f'<span class="dd-badge dd-badge-green">{c}</span>', unsafe_allow_html=True)
                else:
                    st.caption("None found")

            st.markdown(f"""
            <div class="dd-card" style="margin-top:1rem;">
                <div class="dd-card-title">PIPELINE STEPS (LEAK-FREE + IMBALANCE HANDLING)</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#e0f4ff; line-height:2;">
                    <span style="color:#00d4ff;">►</span> Preprocessor + Model wrapped in {'imblearn <b>Pipeline</b>' if resampling.get('enabled') else 'sklearn <b>Pipeline</b>'}<br>
                    <span style="color:#00d4ff;">►</span> Numeric: <b>Median Imputation</b> → <b>StandardScaler</b><br>
                    <span style="color:#00ff88;">►</span> Categorical: <b>Mode Imputation</b> → <b>OneHotEncoder</b><br>
                    <span style="color:#a855f7;">►</span> High-cardinality columns (&gt;50 unique): <b>Dropped</b><br>
                    <span style="color:#ff6b6b;">►</span> Rows with missing target: <b>Dropped</b><br>
                    <span style="color:#00ff88;">►</span> Resampling: <b>{resampling.get('method', 'none')}</b> ({resampling.get('reason', 'n/a')})<br>
                    <span style="color:#00ff88;">►</span> Preprocessing and resampling fitted <b>only on training folds</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Tab 7: LLM Report + PDF Export ──
        with tabs[6]:
            st.markdown("""
            <div style="font-family:'Orbitron',monospace; font-size:0.9rem; color:#00d4ff;
                        letter-spacing:0.15em; margin-bottom:1rem;">
                ◈ RAPPORT D'ANALYSE IA (EN FRANÇAIS)
            </div>
            """, unsafe_allow_html=True)

            if "llm_report" not in st.session_state:
                gen_col, _ = st.columns([2, 4])
                with gen_col:
                    if st.button("🧠 GÉNÉRER LE RAPPORT"):
                        with st.spinner("🤖 Génération du rapport en cours..."):
                            report_text, report_filename = generate_llm_report(
                                st.session_state["pipeline_result"],
                                st.session_state["dataset_info"],
                            )
                            st.session_state["llm_report"]   = report_text
                            st.session_state["llm_filename"] = report_filename
                        st.rerun()
            else:
                report_text     = st.session_state["llm_report"]
                report_filename = st.session_state.get("llm_filename", "DataDoctor_report.pdf")

                st.markdown(
                    '<div style="background:#060f1f; border:1px solid #0d2a4a; border-radius:6px; padding:1.5rem;">',
                    unsafe_allow_html=True,
                )
                st.markdown(report_text)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)

                btn_col1, btn_col2, _ = st.columns([2, 2, 3])

                with btn_col1:
                    if st.button("🔄 RÉGÉNÉRER LE RAPPORT"):
                        del st.session_state["llm_report"]
                        st.session_state.pop("llm_filename", None)
                        st.rerun()

                with btn_col2:
                    if st.button("📄 GÉNÉRER PDF"):
                        with st.spinner("🖨️ Génération du PDF..."):
                            try:
                                saved_path = generate_pdf(
                                    report_text=report_text,
                                    pipeline_result=st.session_state["pipeline_result"],
                                    output_dir=PDF_OUTPUT_DIR,
                                    filename=report_filename,
                                )
                                with open(saved_path, "rb") as f:
                                    pdf_bytes = f.read()
                                st.session_state["pdf_bytes"]    = pdf_bytes
                                st.session_state["pdf_filename"] = report_filename
                                st.success("✅ PDF prêt au téléchargement !")
                            except Exception as e:
                                st.error(f"❌ PDF generation failed: {e}")
                                st.exception(e)

                if st.session_state.get("pdf_bytes"):
                    st.download_button(
                        label="⬇️ TÉLÉCHARGER LE PDF",
                        data=st.session_state["pdf_bytes"],
                        file_name=st.session_state.get("pdf_filename", "DataDoctor_report.pdf"),
                        mime="application/pdf",
                        use_container_width=False,
                    )

                st.markdown(
                    f'<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.75rem; '
                    f'color:#5a7a9a; margin-top:0.5rem;">📄 PDF filename: <b style="color:#00d4ff">'
                    f'{report_filename}</b></div>',
                    unsafe_allow_html=True,
                )

else:
    # ── Empty state ──
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; border:1px dashed #0d2a4a; border-radius:8px;
                background:rgba(6,15,31,0.5); margin-top:2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">🩺</div>
        <div style="font-family:'Orbitron',monospace; font-size:1.1rem; color:#00d4ff; letter-spacing:0.15em; margin-bottom:0.5rem;">
            AWAITING DATA INPUT
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#5a7a9a; max-width:400px; margin:0 auto; line-height:1.8;">
            Upload a CSV or Excel dataset from the sidebar control panel to begin the automated ML diagnosis pipeline.
        </div>
        <br>
        <div style="display:flex; justify-content:center; gap:1rem; flex-wrap:wrap;">
            <span class="dd-badge">CSV</span>
            <span class="dd-badge">XLSX</span>
            <span class="dd-badge dd-badge-green">LEAK-FREE PIPELINE</span>
            <span class="dd-badge dd-badge-purple">GRIDSEARCHCV</span>
            <span class="dd-badge dd-badge-purple">PDF EXPORT</span>
            <span class="dd-badge dd-badge-green">CONFUSION MATRIX</span>
            <span class="dd-badge">LEARNING CURVE</span>
            <span class="dd-badge dd-badge-purple">LIVE PREDICT</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
