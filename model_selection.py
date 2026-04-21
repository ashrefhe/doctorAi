from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ── Mode constants ──────────────────────────────────────────────────────────
MODE_FAST = "fast"
MODE_FULL = "full"

# ── Shared param grids ───────────────────────────────────────────────────────
_BOOST_PARAMS = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
}
_RF_PARAMS = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}
_KNN_PARAMS_CLF = {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}
_KNN_PARAMS_REG = {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}
_DT_PARAMS = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
}


def _lgbm_classifier():
    return LGBMClassifier(random_state=42, verbosity=-1)


def _lgbm_regressor():
    return LGBMRegressor(random_state=42, verbosity=-1)


def get_models_and_params(task: str, mode: str = MODE_FAST) -> dict:
    """
    Returns {model_name: (estimator, param_grid)} for the given task and mode.

    MODE_FAST  → 3-4 strong models, smaller grids (quicker runs)
    MODE_FULL  → all available models, wider grids (more thorough)

    FIX: LightGBM is now correctly added in both modes when available,
    regardless of whether XGBoost is also installed.
    """
    if task == "classification":
        models: dict = {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000, random_state=42),
                {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=42),
                _RF_PARAMS,
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(random_state=42),
                {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5],
                },
            ),
        }

        # Always add best available boosting library
        if HAS_XGB:
            models["XGBoost"] = (
                XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
                _BOOST_PARAMS,
            )
        if HAS_LGBM:
            # FIX: add LightGBM regardless of XGBoost presence
            models["LightGBM"] = (_lgbm_classifier(), _BOOST_PARAMS)

        if mode == MODE_FAST:
            # In fast mode keep at most 4 models: the 3 base + the best boosting lib
            # (XGBoost preferred, then LightGBM, then Gradient Boosting is already there)
            fast_keys = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
            if "XGBoost" in models:
                fast_keys.append("XGBoost")
            elif "LightGBM" in models:
                fast_keys.append("LightGBM")
            return {k: models[k] for k in fast_keys if k in models}

        # Full mode — add KNN + Decision Tree on top of everything
        models["K-Nearest Neighbors"] = (KNeighborsClassifier(), _KNN_PARAMS_CLF)
        models["Decision Tree"] = (DecisionTreeClassifier(random_state=42), _DT_PARAMS)
        return models

    else:  # regression
        models = {
            "Ridge Regression": (
                Ridge(),
                {"alpha": [0.01, 0.1, 1, 10, 100]},
            ),
            "Random Forest": (
                RandomForestRegressor(random_state=42),
                _RF_PARAMS,
            ),
            "Gradient Boosting": (
                GradientBoostingRegressor(random_state=42),
                {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5],
                },
            ),
        }

        if HAS_XGB:
            models["XGBoost"] = (
                XGBRegressor(random_state=42, verbosity=0),
                _BOOST_PARAMS,
            )
        if HAS_LGBM:
            # FIX: same fix for regression
            models["LightGBM"] = (_lgbm_regressor(), _BOOST_PARAMS)

        if mode == MODE_FAST:
            fast_keys = ["Ridge Regression", "Random Forest", "Gradient Boosting"]
            if "XGBoost" in models:
                fast_keys.append("XGBoost")
            elif "LightGBM" in models:
                fast_keys.append("LightGBM")
            return {k: models[k] for k in fast_keys if k in models}

        models["K-Nearest Neighbors"] = (KNeighborsRegressor(), _KNN_PARAMS_REG)
        models["Decision Tree"] = (DecisionTreeRegressor(random_state=42), _DT_PARAMS)
        return models
