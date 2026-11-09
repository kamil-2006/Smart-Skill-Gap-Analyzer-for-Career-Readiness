from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from .config import MODELS_DIR


REG_FEATURES = [
    "coverage",
    "missing_count",
    "hours_per_week",
    "experience_years",
]

CLS_FEATURES = REG_FEATURES


@dataclass(frozen=True)
class ModelBundle:
    lin_reg: LinearRegression
    rf_reg: RandomForestRegressor
    log_reg: LogisticRegression


def _ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _models_paths(models_dir: Path | None = None) -> dict[str, Path]:
    d = models_dir or MODELS_DIR
    return {
        "lin_reg": d / "time_linear.joblib",
        "rf_reg": d / "time_rf.joblib",
        "log_reg": d / "ready_logistic.joblib",
    }


def _make_synthetic_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    coverage = rng.uniform(0.0, 1.0, size=n)
    missing_count = rng.integers(0, 25, size=n)
    hours_per_week = rng.integers(1, 31, size=n)
    experience_years = rng.integers(0, 11, size=n)

    # A simple, monotonic function: more missing skills => more time;
    # more effort/experience => less time. Add noise.
    base = (missing_count * 2.2) + (1.0 - coverage) * 10.0
    effort_bonus = (hours_per_week / 30.0) * 8.0 + (experience_years / 10.0) * 6.0
    noise = rng.normal(0.0, 2.0, size=n)
    time_weeks = np.clip(base - effort_bonus + noise, 1.0, 52.0)

    # Job-ready is probabilistic: higher coverage, fewer missing, more effort.
    logits = (
        2.5 * (coverage - 0.6)
        - 0.12 * (missing_count - 6)
        + 0.06 * (hours_per_week - 10)
        + 0.08 * (experience_years - 1)
    )
    proba = 1.0 / (1.0 + np.exp(-logits))
    ready = rng.binomial(1, np.clip(proba, 0.01, 0.99), size=n)

    return pd.DataFrame(
        {
            "coverage": coverage,
            "missing_count": missing_count,
            "hours_per_week": hours_per_week,
            "experience_years": experience_years,
            "time_weeks": time_weeks,
            "ready": ready,
        }
    )


def train_and_save(models_dir: Path | None = None, *, seed: int = 42) -> ModelBundle:
    _ensure_models_dir()
    paths = _models_paths(models_dir)

    df = _make_synthetic_dataset(seed=seed)
    X_reg = df[REG_FEATURES]
    y_reg = df["time_weeks"]
    X_cls = df[CLS_FEATURES]
    y_cls = df["ready"]

    Xr_train, Xr_test, yr_train, _ = train_test_split(X_reg, y_reg, test_size=0.2, random_state=seed)
    Xc_train, Xc_test, yc_train, _ = train_test_split(X_cls, y_cls, test_size=0.2, random_state=seed)

    lin_reg = LinearRegression()
    lin_reg.fit(Xr_train, yr_train)

    rf_reg = RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        # Use single-threaded training by default to avoid environment restrictions
        # (e.g., blocked process/pipe creation in some sandboxes).
        n_jobs=1,
        max_depth=12,
        min_samples_leaf=2,
    )
    rf_reg.fit(Xr_train, yr_train)

    log_reg = LogisticRegression(max_iter=2000, solver="lbfgs")
    log_reg.fit(Xc_train, yc_train)

    joblib.dump(lin_reg, paths["lin_reg"])
    joblib.dump(rf_reg, paths["rf_reg"])
    joblib.dump(log_reg, paths["log_reg"])

    return ModelBundle(lin_reg=lin_reg, rf_reg=rf_reg, log_reg=log_reg)


def load_or_train(models_dir: Path | None = None) -> ModelBundle:
    _ensure_models_dir()
    paths = _models_paths(models_dir)
    if all(p.exists() for p in paths.values()):
        return ModelBundle(
            lin_reg=joblib.load(paths["lin_reg"]),
            rf_reg=joblib.load(paths["rf_reg"]),
            log_reg=joblib.load(paths["log_reg"]),
        )
    return train_and_save(models_dir)


@dataclass(frozen=True)
class Predictions:
    time_weeks_lr: float
    time_weeks_rf: float
    ready_proba: float


def predict(bundle: ModelBundle, *, coverage: float, missing_count: int, hours_per_week: int, experience_years: int) -> Predictions:
    row = pd.DataFrame(
        [
            {
                "coverage": float(coverage),
                "missing_count": int(missing_count),
                "hours_per_week": int(hours_per_week),
                "experience_years": int(experience_years),
            }
        ]
    )
    time_lr = float(bundle.lin_reg.predict(row[REG_FEATURES])[0])
    time_rf = float(bundle.rf_reg.predict(row[REG_FEATURES])[0])
    ready_proba = float(bundle.log_reg.predict_proba(row[CLS_FEATURES])[0, 1])
    return Predictions(time_weeks_lr=max(1.0, time_lr), time_weeks_rf=max(1.0, time_rf), ready_proba=ready_proba)
