"""Shared preprocessing, data integrity, and experiment recording."""

import gzip
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def read_dataset(project, filename, **kwargs):
    """Read the bundled dataset only after checking its uncompressed hash."""
    folder = ROOT / "projects" / project / "data"
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / (filename + ".gz")
    raw = gzip.decompress(path.read_bytes())
    expected = manifest[filename]["sha256"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    from io import BytesIO

    return pd.read_csv(BytesIO(raw), **kwargs)


def preprocessing(frame):
    """Fit imputers, scaling, and categories on the caller's training rows."""
    cats = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    nums = [c for c in frame.columns if c not in cats]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), nums),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), cats),
    ])


def model_pipeline(frame, estimator):
    return Pipeline([("preprocess", preprocessing(frame)), ("model", estimator)])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "random_state": SEED,
    }
