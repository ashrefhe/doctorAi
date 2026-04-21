"""
llm_report.py — Génération du rapport en français (LLM via OpenRouter ou fallback local).
Inclut toujours une section SHAP.
"""
import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── Rapport local (fallback sans API) ────────────────────────────────────────

def generate_local_report(pipeline_result: dict, dataset_info: dict) -> tuple:
    """
    Génère un rapport structuré en français sans appel externe.
    Retourne (texte_rapport, nom_fichier).
    """
    results = pipeline_result["results"]
    best = pipeline_result["best_model"]
    task = pipeline_result["task"]
    prep = pipeline_result["preprocessing"]
    cv = pipeline_result["cv_folds"]
    shape = dataset_info["shape"]
    shap_data = pipeline_result.get("shap_data", {})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    metric = "Exactitude (Accuracy)" if task == "classification" else "R²"
    filename = f"DataDoctor_{best['model'].replace(' ', '_')}_{task}_{ts}.pdf"

    lines = []
    lines.append("# Rapport DataDoctor AI — Pipeline AutoML")
    lines.append(f"\n*Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}*\n")
    lines.append("---\n")

    # Résumé exécutif
    lines.append("## Résumé Exécutif")
    lines.append(
        f"Le pipeline AutoML a été exécuté sur un jeu de données de **{shape[0]} lignes × {shape[1]} colonnes**. "
        f"Tâche détectée : **{task.upper()}**. "
        f"Séparation train/test : **{prep.get('train_size', '?')} exemples d'entraînement** et "
        f"**{prep.get('holdout_test_size', '?')} exemples de test final (holdout)**. "
        f"Meilleur modèle : **{best['model']}** avec {metric} (CV) = **{best['cv_mean']:.4f} ± {best['cv_std']:.4f}** "
        f"({cv} plis de validation croisée).\n"
    )

    # Analyse de la tâche
    lines.append("## Analyse de la Tâche")
    lines.append(f"{pipeline_result.get('task_explanation', '')}\n")

    # Validation holdout
    lines.append("## Validation Finale sur le Jeu de Test Indépendant")
    holdout = best.get("holdout", {})
    if holdout and not holdout.get("error"):
        lines.append(
            f"Le modèle champion **{best['model']}** a été évalué sur {prep.get('holdout_test_size', '?')} "
            f"exemples totalement mis de côté avant tout entraînement :"
        )
        if task == "classification":
            lines.append(f"- **Accuracy holdout** : {holdout.get('accuracy', 'N/A'):.4f}")
            lines.append(f"- **F1-score pondéré** : {holdout.get('f1_weighted', 'N/A'):.4f}")
            if "roc_auc" in holdout:
                lines.append(f"- **AUC-ROC** : {holdout['roc_auc']:.4f}")
        else:
            lines.append(f"- **R² holdout** : {holdout.get('r2', 'N/A'):.4f}")
            lines.append(f"- **MAE** : {holdout.get('mae', 'N/A'):.4f}")
            lines.append(f"- **RMSE** : {holdout.get('rmse', 'N/A'):.4f}")
        cv_vs_holdout = ""
        if task == "classification" and "accuracy" in holdout:
            diff = holdout["accuracy"] - best["cv_mean"]
            cv_vs_holdout = f" (écart CV vs holdout : {diff:+.4f})"
        elif task == "regression" and "r2" in holdout:
            diff = holdout["r2"] - best["cv_mean"]
            cv_vs_holdout = f" (écart CV vs holdout : {diff:+.4f})"
        lines.append(f"\nCes scores reflètent la performance réelle de déploiement{cv_vs_holdout}.\n")
    else:
        lines.append("Évaluation holdout non disponible.\n")

    # Préprocessing
    lines.append("## Prétraitement des Données")
    lines.append(f"- **Features numériques** : {prep['numeric_features']} → Imputation médiane + StandardScaler")
    lines.append(f"- **Features catégorielles** : {prep['categorical_features']} → Imputation mode + OneHotEncoder")
    lines.append(f"- **Valeurs manquantes traitées** : {prep['total_missing_values']}")
    lines.append(f"- Les colonnes à forte cardinalité (> 50 valeurs uniques) ont été supprimées.")
    if task == "classification":
        resampling = prep.get("resampling_info", {})
        method = resampling.get("method", "none")
        if method != "none":
            lines.append(f"- **Rééchantillonnage** : {method} appliqué à l'intérieur des plis CV uniquement (pas de fuite).")
        else:
            lines.append(f"- **Rééchantillonnage** : aucun ({resampling.get('reason', 'non nécessaire')}).")
    lines.append("")

    # Comparaison des modèles
    lines.append("## Comparaison des Modèles")
    metric_col = "CV Accuracy" if task == "classification" else "CV R²"
    lines.append(f"| Rang | Modèle | {metric_col} | Écart-type | Meilleurs Hyperparamètres |")
    lines.append("|------|--------|-----------|------------|--------------------------|")
    for i, r in enumerate(results):
        if r.get("cv_mean", -999) <= -999:
            continue
        rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        icon = rank_icons[i] if i < len(rank_icons) else f"#{i+1}"
        params_str = ", ".join(f"{k}={v}" for k, v in r["best_params"].items()) or "défaut"
        lines.append(
            f"| {icon} | {r['model']} | {r['cv_mean']:.4f} | {r['cv_std']:.4f} | {params_str} |"
        )
    lines.append("")

    # Analyse du modèle champion
    lines.append("## Analyse du Modèle Champion")
    lines.append(f"**{best['model']}** a obtenu le meilleur score CV ({metric}) de **{best['cv_mean']:.4f}**.")
    if best["best_params"]:
        lines.append("\nHyperparamètres optimaux identifiés par GridSearchCV :")
        for k, v in best["best_params"].items():
            lines.append(f"- `{k}` = **{v}**")
    lines.append("")

    # Section SHAP — toujours présente
    lines.append("## Explicabilité du Modèle (SHAP)")
    if shap_data.get("available"):
        lines.append(
            f"Les valeurs SHAP mesurent la contribution moyenne de chaque variable à la prédiction "
            f"du modèle **{best['model']}** sur le jeu de test holdout. "
            f"Une valeur SHAP élevée indique une forte influence sur la sortie du modèle.\n"
        )
        lines.append("**Top variables par importance SHAP (|valeur SHAP| moyenne) :**\n")
        lines.append("| Rang | Variable | Importance SHAP moyenne |")
        lines.append("|------|----------|------------------------|")
        for i, (feat, val) in enumerate(zip(shap_data["feature_names"], shap_data["mean_abs_shap"])):
            icon = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
            lines.append(f"| {icon} | {feat} | {val:.4f} |")
        lines.append("")
        lines.append(
            "> **Interprétation** : Les variables en tête de ce classement sont les plus "
            "déterminantes pour les prédictions du modèle. Concentrez vos efforts de collecte "
            "de données et d'ingénierie sur ces variables en priorité."
        )
    else:
        err = shap_data.get("error", "module shap non disponible")
        lines.append(
            f"L'analyse SHAP n'a pas pu être effectuée pour ce pipeline ({err}). "
            "Pour l'activer, installez le module : `pip install shap`."
        )
    lines.append("")

    # Recommandations
    lines.append("## Recommandations")
    lines.append("- **Ingénierie des features** : Envisager des interactions entre les variables SHAP les plus importantes.")
    lines.append("- **Plus de données** : Si le score est inférieur à 0,85, collecter davantage d'exemples étiquetés.")
    lines.append("- **Déploiement** : Sérialiser le meilleur Pipeline avec joblib et l'exposer via une API REST.")
    lines.append("- **Monitoring** : Surveiller la dérive des prédictions en production et réentraîner périodiquement.")
    if task == "classification":
        lines.append("- **Déséquilibre de classes** : Vérifier la distribution ; SMOTE ou `class_weight='balanced'` si nécessaire.")
    lines.append("")

    # Avertissements
    lines.append("## Avertissements et Risques")
    if best["cv_std"] > 0.05:
        lines.append(
            f"- **Variance élevée** : L'écart-type CV = {best['cv_std']:.4f} suggère une instabilité. "
            "Considérer plus de données ou une régularisation plus forte."
        )
    else:
        lines.append("- L'écart-type CV est faible — le modèle semble stable entre les plis. ✅")
    lines.append(
        "- Ce pipeline utilise un split train/holdout strict AVANT toute validation croisée "
        "pour garantir l'absence totale de fuite de données."
    )
    lines.append("- Validez toujours les résultats sur des données de production réelles avant déploiement.")

    return "\n".join(lines), filename


# ── Construction du prompt LLM ────────────────────────────────────────────────

def build_prompt(pipeline_result: dict, dataset_info: dict) -> str:
    results = pipeline_result["results"]
    best = pipeline_result["best_model"]
    task = pipeline_result["task"]
    prep = pipeline_result["preprocessing"]
    shap_data = pipeline_result.get("shap_data", {})

    results_text = "\n".join([
        f"  - {r['model']}: CV Moyen={r['cv_mean']:.4f} ± {r['cv_std']:.4f}, Meilleurs Params={r['best_params']}, "
        f"Holdout={r.get('holdout', {})}"
        for r in results if r.get("cv_mean", -999) > -999
    ])

    shap_text = ""
    if shap_data.get("available"):
        top_shap = list(zip(shap_data["feature_names"][:10], shap_data["mean_abs_shap"][:10]))
        shap_text = "\n".join([f"  {i+1}. {f}: {v:.4f}" for i, (f, v) in enumerate(top_shap)])
    else:
        shap_text = f"Non disponible ({shap_data.get('error', 'shap non installé')})"

    prompt = f"""Tu es DataDoctor AI, un expert en Machine Learning. Analyse ce pipeline AutoML et produis un rapport structuré et professionnel EN FRANÇAIS.

RÈGLES STRICTES :
- Rédige TOUT en français.
- Ne jamais inventer des métriques ou noms de modèles non listés ci-dessous.
- Ne pas spéculer au-delà des données fournies.
- Si une information est manquante, écrire "non disponible".
- Le rapport doit être actionnable et précis.

## Informations sur le Jeu de Données
- Dimensions : {dataset_info['shape']}
- Tâche : {task.upper()}
- Features numériques : {prep['numeric_features']}
- Features catégorielles : {prep['categorical_features']}
- Valeurs manquantes traitées : {prep['total_missing_values']}
- Taille train : {prep.get('train_size', '?')} | Taille holdout : {prep.get('holdout_test_size', '?')}

## Modèles Évalués (GridSearchCV + {pipeline_result['cv_folds']} plis CV)
{results_text}

## Meilleur Modèle
- Nom : {best['model']}
- Score CV : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}
- Performance holdout (test final) : {best.get('holdout', {})}
- Hyperparamètres optimaux : {best['best_params']}

## Valeurs SHAP (Top 10 variables — importance absolue moyenne)
{shap_text}

## Sections Requises dans le Rapport :
1. **Résumé Exécutif** — Ce qui s'est passé dans ce pipeline ?
2. **Analyse de la Tâche** — Pourquoi s'agit-il d'un problème de {task} ?
3. **Validation Finale (Holdout)** — Interpréter les métriques de test final vs CV.
4. **Prétraitement des Données** — Quelles transformations ont été appliquées ?
5. **Comparaison des Modèles** — Comparer tous les modèles, expliquer pourquoi le meilleur a gagné.
6. **Analyse du Modèle Champion** — Expliquer le modèle gagnant et ses hyperparamètres.
7. **Explicabilité SHAP** — Interpréter les variables SHAP les plus importantes et leur impact métier.
8. **Recommandations** — Que doit faire l'utilisateur ensuite ?
9. **Avertissements / Risques** — Problèmes de qualité des données, risques de surapprentissage ?

Utilise du markdown avec des titres clairs et des emojis. Sois concis et orienté action.
"""
    return prompt


def suggest_filename(pipeline_result: dict) -> str:
    best_name = pipeline_result["best_model"]["model"].replace(" ", "_")
    task = pipeline_result["task"]
    score = pipeline_result["best_model"]["cv_mean"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"DataDoctor_{best_name}_{task}_score{score:.3f}_{ts}.pdf"


def generate_llm_report(pipeline_result: dict, dataset_info: dict) -> tuple:
    """
    Tente un rapport LLM via OpenRouter ; repli sur le rapport local en cas d'échec.
    Retourne (texte_rapport, nom_fichier).
    """
    filename = suggest_filename(pipeline_result)

    if not OPENROUTER_API_KEY:
        return generate_local_report(pipeline_result, dataset_info)

    prompt = build_prompt(pipeline_result, dataset_info)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://datadoctor.ai",
        "X-Title": "DataDoctor AI",
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es DataDoctor AI, un expert data scientist francophone. "
                    "Fournis des analyses ML claires, actionnables et professionnelles EN FRANÇAIS. "
                    "Ne jamais inventer de données. Analyser uniquement ce qui est explicitement fourni."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2500,
        "temperature": 0.7,
        "stream": False,
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"], filename
        elif "error" in data:
            local_text, local_fname = generate_local_report(pipeline_result, dataset_info)
            return (
                f"⚠️ Erreur API LLM : {data['error'].get('message', 'Inconnue')}. "
                f"Rapport local affiché à la place.\n\n---\n\n{local_text}",
                local_fname,
            )
        else:
            return generate_local_report(pipeline_result, dataset_info)

    except requests.exceptions.Timeout:
        local_text, local_fname = generate_local_report(pipeline_result, dataset_info)
        return f"⚠️ Délai d'attente LLM dépassé. Rapport local :\n\n---\n\n{local_text}", local_fname
    except requests.exceptions.RequestException as e:
        local_text, local_fname = generate_local_report(pipeline_result, dataset_info)
        return f"⚠️ Erreur réseau LLM ({e}). Rapport local :\n\n---\n\n{local_text}", local_fname
    except Exception as e:
        local_text, local_fname = generate_local_report(pipeline_result, dataset_info)
        return f"⚠️ Erreur inattendue ({e}). Rapport local :\n\n---\n\n{local_text}", local_fname
