# ============================================================
# train_models.py — offline training step for app.py
#
# Reads the enriched observations workbook, fits one entropy
# decision tree per site, and writes everything the app needs
# into models.joblib (a few KB).
#
# Run this locally whenever the data or the model settings
# change, then commit the refreshed models.joblib. The app
# itself never reads the workbook, so the raw sighting records
# stay off the deployed repo entirely.
#
# pip install pandas numpy scikit-learn joblib openpyxl
# Run:  python train_models.py
# ============================================================

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_data

OUTPUT_FILE = "models.joblib"

SITES = ["Palmachim", "Hadera", "Achziv"]

# Day_Night is only used where it actually carries signal. Hadera has 3
# night sightings out of 945 and Achziv 18 out of 1624 — there the feature
# is unusable, and dropping it also saves the rows an "unknown time"
# filter would otherwise have discarded.
SITE_FEATURES = {
    "Palmachim": ["Season", "Day_Night", "Group_Solo"],
    "Hadera":    ["Season", "Group_Solo"],
    "Achziv":    ["Season", "Group_Solo"],
}

TEMP_FEATURE = "Temp_C"

SPECIES_MIN_OBS = 30    # rarer species at a site fold into "Other"
MAX_DEPTH = 4           # bounded for legibility / animation length
MIN_SAMPLES_LEAF = 20
TEST_SIZE = 0.25
RANDOM_STATE = 42


def _site_group(site_standardized):
    """Fold 'Achziv canyon', 'Hadera power plant', ... into a site key."""
    s = str(site_standardized).lower()
    for site in SITES:
        if site.lower() in s:
            return site
    return None


def prepare_data():
    """
    Load the enriched observations, keep only the three modelled sites,
    and fill missing water temperature.

    Hadera's Temp_C is only ~7% populated (the power-plant outflow is
    poorly represented in the Copernicus grid), so dropping those rows
    would leave far too little to train on. Missing values are filled
    from real readings only: site+season mean -> site mean ->
    season mean -> global mean.

    Temp_C_observed keeps the pre-imputation values so the app's
    temperature slider can be bounded by what was really measured.
    """
    df = load_data()
    df["SiteGroup"] = df["site_standardized"].apply(_site_group)
    df = df[df["SiteGroup"].notna()].copy()

    df["Temp_C_observed"] = df["Temp_C"]
    site_season = df.groupby(["SiteGroup", "Season"])["Temp_C"].transform("mean")
    site_mean = df.groupby("SiteGroup")["Temp_C"].transform("mean")
    season_mean = df.groupby("Season")["Temp_C"].transform("mean")
    df["Temp_C"] = (df["Temp_C"]
                    .fillna(site_season)
                    .fillna(site_mean)
                    .fillna(season_mean)
                    .fillna(df["Temp_C"].mean()))
    return df


def _site_frame(df, site):
    """Rows for one site, with the target column attached."""
    d = df[(df["SiteGroup"] == site) & (df["Season"] != "Unknown")].copy()
    if "Day_Night" in SITE_FEATURES[site]:
        d = d[d["Day_Night"] != "Unknown"]

    # Species seen rarely at this site get pooled so the tree isn't
    # trying to carve out classes it has almost no evidence for.
    counts = d["Common_EN"].value_counts()
    common = set(counts[counts >= SPECIES_MIN_OBS].index)
    d["Target"] = d["Common_EN"].where(d["Common_EN"].isin(common), "Other")
    return d


def _encode(d, site):
    """One-hot the site's categorical features, then append temperature."""
    X = pd.get_dummies(d[SITE_FEATURES[site]]).astype(int)
    X[TEMP_FEATURE] = d[TEMP_FEATURE].to_numpy()
    return X


def train_models(df):
    """
    Fit one entropy decision tree per site.

    No class_weight="balanced": it looks right for imbalanced classes
    but measurably wrecks accuracy here (at Achziv it drops top-1 from
    0.47 to ~0.15, well below the majority baseline).

    max_depth is fixed rather than searched — every alternative tested
    landed within noise on accuracy, so the cap is really about keeping
    the tree legible and the animation a few steps long.
    """
    models = {}
    for site in SITES:
        d = _site_frame(df, site)
        X, y = _encode(d, site), d["Target"]

        # Still held out rather than fitting on everything, so the tree
        # keeps the shape it was validated with.
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

        tree = DecisionTreeClassifier(
            criterion="entropy", max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE,
        ).fit(X_train, y_train)

        models[site] = {
            "tree": tree,
            "columns": list(X.columns),
            "classes": list(tree.classes_),
            "n_rows": len(d),
            "temp_min": float(d["Temp_C_observed"].min()),
            "temp_max": float(d["Temp_C_observed"].max()),
            "temp_default": float(d["Temp_C_observed"].median()),
        }
    return models


def main():
    df = prepare_data()
    print(f"Loaded {len(df)} sightings across {df['SiteGroup'].nunique()} sites")

    models = train_models(df)
    for site, m in models.items():
        tree = m["tree"]
        print(f"  {site:<10} {m['n_rows']:>5} rows  "
              f"depth {tree.get_depth()}, {tree.tree_.node_count} nodes, "
              f"{len(m['classes'])} classes")

    # SITES order is preserved by dict insertion, so the app can take its
    # site list straight from the file.
    joblib.dump(models, OUTPUT_FILE, compress=3)
    print(f"\nWrote {OUTPUT_FILE} — commit this; the workbook stays local.")


if __name__ == "__main__":
    main()
