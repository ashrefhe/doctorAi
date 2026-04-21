import copy
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, StratifiedKFold, KFold, learning_curve
)
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.figure_factory as ff

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE


DARK_THEME = {
    "bg": "#050d1a",
    "card": "#0a1628",
    "accent1": "#00d4ff",
    "accent2": "#0066ff",
    "accent3": "#00ff88",
    "text": "#e0f0ff",
    "grid": "#112240",
}


def _choose_resampler(y, cv: int, enabled: bool = True):
    """
    Select a safe imbalance strategy for classification, applied INSIDE CV folds.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)

    if len(classes) < 2:
        return {
            "enabled": False,
            "method": "none",
            "reason": "single_class_target",
            "imbalance_ratio": 1.0,
            "minority_count": int(counts.min()) if len(counts) else 0,
            "k_neighbors": None,
            "sampler": None,
        }

    min_count = int(counts.min())
    max_count = int(counts.max())
    ratio = float(max_count / max(min_count, 1))

    if (not enabled) or ratio <= 1.5:
        return {
            "enabled": False,
            "method": "none",
            "reason": "balanced_or_disabled",
            "imbalance_ratio": ratio,
            "minority_count": min_count,
            "k_neighbors": None,
            "sampler": None,
        }

    effective_train_min = max(1, int(np.floor(min_count * (cv - 1) / cv)))

    if effective_train_min < 2:
        return {
            "enabled": False,
            "method": "none",
            "reason": "too_few_minority_samples",
            "imbalance_ratio": ratio,
            "minority_count": min_count,
            "k_neighbors": None,
            "sampler": None,
        }

    if effective_train_min < 6:
        return {
            "enabled": True,
            "method": "RandomOverSampler",
            "reason": "minority_too_small_for_safe_smote",
            "imbalance_ratio": ratio,
            "minority_count": min_count,
            "k_neighbors": None,
            "sampler": RandomOverSampler(random_state=42),
        }

    k_neighbors = min(5, effective_train_min - 1)
    return {
        "enabled": True,
        "method": "SMOTE",
        "reason": "safe_for_smote",
        "imbalance_ratio": ratio,
        "minority_count": min_count,
        "k_neighbors": int(k_neighbors),
        "sampler": SMOTE(random_state=42, k_neighbors=int(k_neighbors)),
    }


def run_grid_search(models_params: dict, X, y, preprocessor, task: str, cv: int = 5):
    """
    Run GridSearchCV for each model using a leak-free pipeline.
    """
    results = []
    scoring = "accuracy" if task == "classification" else "r2"

    if task == "classification":
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        resampling_info = _choose_resampler(y, cv=cv, enabled=True)
    else:
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        resampling_info = {
            "enabled": False,
            "method": "none",
            "reason": "regression_task",
            "imbalance_ratio": 1.0,
            "minority_count": None,
            "k_neighbors": None,
            "sampler": None,
        }

    for name, (estimator, param_grid) in models_params.items():
        try:
            if task == "classification" and resampling_info["enabled"] and resampling_info["sampler"] is not None:
                pipe = ImbPipeline([
                    ("preprocessor", copy.deepcopy(preprocessor)),
                    ("sampler", copy.deepcopy(resampling_info["sampler"])),
                    ("model", estimator),
                ])
            else:
                pipe = Pipeline([
                    ("preprocessor", copy.deepcopy(preprocessor)),
                    ("model", estimator),
                ])

            pipe_param_grid = {f"model__{k}": v for k, v in param_grid.items()}

            gs = GridSearchCV(
                estimator=pipe,
                param_grid=pipe_param_grid,
                cv=kfold,
                scoring=scoring,
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X, y)

            cv_scores = cross_val_score(
                gs.best_estimator_,
                X,
                y,
                cv=kfold,
                scoring=scoring,
                n_jobs=-1,
            )

            display_params = {
                k.replace("model__", ""): v
                for k, v in gs.best_params_.items()
            }

            results.append({
                "model": name,
                "best_score": float(gs.best_score_),
                "best_params": display_params,
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
                "cv_scores": cv_scores.tolist(),
                "estimator": gs.best_estimator_,
            })
        except Exception as e:
            results.append({
                "model": name,
                "best_score": -999,
                "best_params": {},
                "cv_mean": -999,
                "cv_std": 0,
                "cv_scores": [],
                "estimator": None,
                "error": str(e),
            })

    results.sort(key=lambda x: x["best_score"], reverse=True)
    safe_resampling_info = {k: v for k, v in resampling_info.items() if k != "sampler"}
    return results, safe_resampling_info


def build_results_charts(results: list, task: str):
    valid = [r for r in results if r.get("cv_mean", -999) > -999]
    models = [r["model"] for r in valid]
    scores = [r["cv_mean"] for r in valid]
    stds = [r["cv_std"] for r in valid]
    metric_label = "Accuracy" if task == "classification" else "R² Score"

    colors = [
        DARK_THEME["accent1"], DARK_THEME["accent2"], DARK_THEME["accent3"],
        "#ff6b6b", "#ffd93d", "#a855f7",
    ]

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=models,
        y=scores,
        error_y=dict(type="data", array=stds, visible=True, color=DARK_THEME["accent3"]),
        marker=dict(color=colors[:len(models)], line=dict(color=DARK_THEME["accent1"], width=1)),
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        textfont=dict(color=DARK_THEME["text"], size=12),
    ))
    bar_fig.update_layout(
        title=dict(text=f"Model Comparison — {metric_label}", font=dict(color=DARK_THEME["accent1"], size=18)),
        plot_bgcolor=DARK_THEME["card"],
        paper_bgcolor=DARK_THEME["bg"],
        font=dict(color=DARK_THEME["text"]),
        xaxis=dict(gridcolor=DARK_THEME["grid"], tickfont=dict(color=DARK_THEME["text"])),
        yaxis=dict(gridcolor=DARK_THEME["grid"], tickfont=dict(color=DARK_THEME["text"]), title=metric_label),
        margin=dict(t=60, b=40, l=60, r=20),
        height=400,
    )

    violin_fig = go.Figure()
    for i, r in enumerate(valid):
        if r["cv_scores"]:
            violin_fig.add_trace(go.Violin(
                y=r["cv_scores"],
                name=r["model"],
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                opacity=0.7,
                line_color=DARK_THEME["accent1"],
            ))
    violin_fig.update_layout(
        title=dict(text="Cross-Validation Score Distribution", font=dict(color=DARK_THEME["accent1"], size=18)),
        plot_bgcolor=DARK_THEME["card"],
        paper_bgcolor=DARK_THEME["bg"],
        font=dict(color=DARK_THEME["text"]),
        yaxis=dict(gridcolor=DARK_THEME["grid"], title=metric_label),
        xaxis=dict(gridcolor=DARK_THEME["grid"]),
        showlegend=False,
        height=400,
    )

    radar_fig = None
    if len(valid) >= 3:
        radar_models = [r["model"] for r in valid[:4]]
        radar_scores = [r["cv_mean"] for r in valid[:4]]
        radar_stds = [max(0.0, 1 - r["cv_std"]) for r in valid[:4]]
        rank_proxy = [1 / (i + 1) for i in range(len(radar_models))]
        categories = ["CV Score", "Stability", "Rank Score"]

        radar_fig = go.Figure()
        for i, (m, s, st, rk) in enumerate(zip(radar_models, radar_scores, radar_stds, rank_proxy)):
            radar_fig.add_trace(go.Scatterpolar(
                r=[s, st, rk],
                theta=categories,
                fill="toself",
                name=m,
                line=dict(color=colors[i % len(colors)]),
                opacity=0.7,
            ))
        radar_fig.update_layout(
            polar=dict(
                bgcolor=DARK_THEME["card"],
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=DARK_THEME["grid"], color=DARK_THEME["text"]),
                angularaxis=dict(gridcolor=DARK_THEME["grid"], color=DARK_THEME["text"]),
            ),
            paper_bgcolor=DARK_THEME["bg"],
            font=dict(color=DARK_THEME["text"]),
            title=dict(text="Top Models — Radar Comparison", font=dict(color=DARK_THEME["accent1"], size=18)),
            showlegend=True,
            legend=dict(font=dict(color=DARK_THEME["text"])),
            height=420,
        )

    return bar_fig, violin_fig, radar_fig


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Confusion Matrix (Classification)
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion_matrix_chart(estimator, X_test, y_test, label_encoder=None):
    """
    Build an annotated confusion matrix heatmap using Plotly.
    Returns a go.Figure or None on failure.
    """
    try:
        y_pred = estimator.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        if label_encoder is not None:
            try:
                class_labels = [str(c) for c in label_encoder.classes_]
            except Exception:
                class_labels = [str(i) for i in range(cm.shape[0])]
        else:
            class_labels = [str(i) for i in range(cm.shape[0])]

        # Normalised version for colour scale
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        # Annotation text: count + percentage
        annotations = [
            [f"{cm[i][j]}<br>({cm_norm[i][j]*100:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ]

        fig = go.Figure(go.Heatmap(
            z=cm_norm.tolist(),
            x=class_labels,
            y=class_labels,
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#e0f4ff"),
            colorscale=[
                [0.0, "#020b18"],
                [0.4, "#0066ff"],
                [0.7, "#00d4ff"],
                [1.0, "#00ff88"],
            ],
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(
                title=dict(text="Rate", font=dict(color="#00d4ff")),
                tickfont=dict(color="#e0f4ff"),
            ),
        ))
        fig.update_layout(
            title=dict(text="Confusion Matrix (Holdout)", font=dict(color="#00d4ff", size=16)),
            plot_bgcolor="#060f1f",
            paper_bgcolor="#020b18",
            font=dict(color="#e0f4ff"),
            xaxis=dict(title="Predicted", tickfont=dict(color="#e0f4ff"), gridcolor="#0d2a4a"),
            yaxis=dict(title="Actual", tickfont=dict(color="#e0f4ff"), gridcolor="#0d2a4a", autorange="reversed"),
            height=max(350, 60 * len(class_labels) + 100),
            margin=dict(t=50, b=60, l=80, r=20),
        )
        return fig
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Residuals Plot (Regression)
# ─────────────────────────────────────────────────────────────────────────────

def build_residuals_chart(estimator, X_test, y_test):
    """
    Build predicted vs actual + residuals distribution charts for regression.
    Returns (scatter_fig, hist_fig) or (None, None) on failure.
    """
    try:
        y_pred = estimator.predict(X_test)
        residuals = np.array(y_test) - np.array(y_pred)

        # Predicted vs Actual
        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(
            x=y_pred, y=y_test,
            mode="markers",
            marker=dict(color="#00d4ff", size=6, opacity=0.7,
                        line=dict(color="#0066ff", width=0.5)),
            name="Predictions",
        ))
        # Perfect prediction line
        min_val = float(min(np.min(y_pred), np.min(y_test)))
        max_val = float(max(np.max(y_pred), np.max(y_test)))
        scatter_fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines",
            line=dict(color="#00ff88", dash="dash", width=2),
            name="Perfect fit",
        ))
        scatter_fig.update_layout(
            title=dict(text="Predicted vs Actual (Holdout)", font=dict(color="#00d4ff", size=16)),
            plot_bgcolor="#060f1f", paper_bgcolor="#020b18",
            font=dict(color="#e0f4ff"),
            xaxis=dict(title="Predicted", gridcolor="#0d2a4a", tickfont=dict(color="#e0f4ff")),
            yaxis=dict(title="Actual", gridcolor="#0d2a4a", tickfont=dict(color="#e0f4ff")),
            legend=dict(font=dict(color="#e0f4ff")),
            height=380,
            margin=dict(t=50, b=50, l=60, r=20),
        )

        # Residuals distribution
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            marker=dict(color="#0066ff", line=dict(color="#00d4ff", width=0.5)),
            opacity=0.8,
            name="Residuals",
        ))
        # Zero reference line
        hist_fig.add_vline(x=0, line_dash="dash", line_color="#00ff88", line_width=2,
                           annotation_text="Zero", annotation_font_color="#00ff88")
        hist_fig.update_layout(
            title=dict(text="Residuals Distribution", font=dict(color="#00d4ff", size=16)),
            plot_bgcolor="#060f1f", paper_bgcolor="#020b18",
            font=dict(color="#e0f4ff"),
            xaxis=dict(title="Residual (Actual − Predicted)", gridcolor="#0d2a4a",
                       tickfont=dict(color="#e0f4ff")),
            yaxis=dict(title="Count", gridcolor="#0d2a4a", tickfont=dict(color="#e0f4ff")),
            height=320,
            margin=dict(t=50, b=50, l=60, r=20),
        )

        return scatter_fig, hist_fig
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Learning Curve
# ─────────────────────────────────────────────────────────────────────────────

def build_learning_curve_chart(estimator, X, y, task: str, cv: int = 5):
    """
    Compute and plot train/validation learning curves for the best estimator.
    Returns a go.Figure or None on failure.
    """
    try:
        scoring = "accuracy" if task == "classification" else "r2"
        if task == "classification":
            kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

        n_samples = len(y)
        # Use 6 evenly-spaced training sizes from 10% to 100%
        train_sizes_abs = np.linspace(0.10, 1.0, 6)

        train_sizes, train_scores, val_scores = learning_curve(
            estimator,
            X, y,
            train_sizes=train_sizes_abs,
            cv=kfold,
            scoring=scoring,
            n_jobs=-1,
            shuffle=True,
            random_state=42,
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        metric_label = "Accuracy" if task == "classification" else "R²"

        fig = go.Figure()

        # Training score band
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]).tolist(),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]).tolist(),
            fill="toself",
            fillcolor="rgba(0,102,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes.tolist(), y=train_mean.tolist(),
            mode="lines+markers",
            line=dict(color="#0066ff", width=2),
            marker=dict(size=7, color="#0066ff"),
            name="Training score",
        ))

        # Validation score band
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]).tolist(),
            y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]).tolist(),
            fill="toself",
            fillcolor="rgba(0,255,136,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes.tolist(), y=val_mean.tolist(),
            mode="lines+markers",
            line=dict(color="#00ff88", width=2),
            marker=dict(size=7, color="#00ff88"),
            name="CV Validation score",
        ))

        fig.update_layout(
            title=dict(text=f"Learning Curve — {metric_label}", font=dict(color="#00d4ff", size=16)),
            plot_bgcolor="#060f1f", paper_bgcolor="#020b18",
            font=dict(color="#e0f4ff"),
            xaxis=dict(title="Training samples", gridcolor="#0d2a4a",
                       tickfont=dict(color="#e0f4ff")),
            yaxis=dict(title=metric_label, gridcolor="#0d2a4a",
                       tickfont=dict(color="#e0f4ff")),
            legend=dict(font=dict(color="#e0f4ff"), bgcolor="rgba(0,0,0,0)"),
            height=380,
            margin=dict(t=50, b=50, l=60, r=20),
        )
        return fig
    except Exception:
        return None
