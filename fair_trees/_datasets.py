from importlib.resources import files
import pandas as pd


def _load_single(base, name):
    df = pd.read_csv(base.joinpath(f"{name}.csv"))

    # Separate columns by prefix convention
    y_cols = [c for c in df.columns if c.startswith("__y__")]
    z_cols = [c for c in df.columns if c.startswith("__Z__")]
    meta_cols = ["__split_groups_id__", "__Z_intersectional__"]
    x_cols = [c for c in df.columns if c not in y_cols + z_cols + meta_cols]

    X = df[x_cols]
    y = df[y_cols].rename(columns=lambda c: c.removeprefix("__y__"))
    Z = df[z_cols].rename(columns=lambda c: c.removeprefix("__Z__"))
    split_groups_id = df["__split_groups_id__"]
    Z_intersectional = df["__Z_intersectional__"]

    return {
        "X": X,
        "y": y,
        "split_groups_id": split_groups_id,
        "Z": Z,
        "Z_intersectional": Z_intersectional,
    }


def load_datasets():
    base = files("fair_trees")
    return {
        "adult": _load_single(base, "adult"),
        "bank_marketing": _load_single(base, "bank_marketing"),
    }