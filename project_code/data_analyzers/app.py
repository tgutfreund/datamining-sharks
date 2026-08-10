# ============================================================
# app.py — "I saw something in the water. What could it be?"
#
# Interactive per-site decision tree predictor. Pick a site
# (Palmachim / Hadera / Achziv), describe the sighting
# (season, temperature, alone or in a group, day or night),
# and watch the path light up down that site's decision tree.
#
# Each site gets its own tree because the sites are ecologically
# very different: Palmachim and Achziv are almost purely rays,
# while Hadera's power-plant outflow is a winter shark
# aggregation. Splitting criterion is entropy (information gain).
#
# pip install pandas numpy scikit-learn plotly matplotlib streamlit
# Run:  streamlit run app.py
# ============================================================

import textwrap
import time

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from data_loader import load_data, CB_PALETTE

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SITES = ["Palmachim", "Hadera", "Achziv"]

# Day_Night is only offered where it actually carries signal.
# Hadera has 3 night sightings out of 945 and Achziv 18 out of
# 1624 — there the feature is unusable, and dropping it also
# saves the rows an "unknown time" filter would have discarded.
SITE_FEATURES = {
    "Palmachim": ["Season", "Day_Night", "Group_Solo"],
    "Hadera":    ["Season", "Group_Solo"],
    "Achziv":    ["Season", "Group_Solo"],
}

# Temp_C is appended as a numeric feature for every site.
TEMP_FEATURE = "Temp_C"

SPECIES_MIN_OBS = 30    # rarer species at a site fold into "Other"
MAX_DEPTH = 4           # bounded for legibility / animation length
MIN_SAMPLES_LEAF = 20
TEST_SIZE = 0.25
RANDOM_STATE = 42

SEASONS = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]

PATH_COLOR = CB_PALETTE[3]     # orange-red, highlights the taken path
LEAF_COLOR = CB_PALETTE[2]     # green, the final answer
MUTED_COLOR = "#EAEAEA"
LABEL_WRAP = 14                # wrap node text so boxes don't overlap
FRAME_DELAY = 0.55             # seconds between animation frames

SITE_BLURBS = {
    "Palmachim": "A ray site — stingrays and whiprays dominate, with a "
                 "real night-time presence (116 night sightings).",
    "Hadera": "The power-plant outflow: a winter shark aggregation. "
              "Dusky and sandbar sharks make up most sightings.",
    "Achziv": "The most ray-dominated site of the three, led by the "
              "round ribbontail ray.",
}


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------
def _site_group(site_standardized):
    """Fold 'Achziv canyon', 'Hadera power plant', ... into a site key."""
    s = str(site_standardized).lower()
    for site in SITES:
        if site.lower() in s:
            return site
    return None


@st.cache_data(show_spinner="Loading sightings…")
def prepare_data():
    """
    Load the enriched observations, keep only the three modelled sites,
    and fill missing water temperature.

    Hadera's Temp_C is only ~7% populated (the power-plant outflow is
    poorly represented in the Copernicus grid), so dropping those rows
    would leave far too little to train on. Missing values are filled
    from real readings only: site+season mean -> site mean ->
    season mean -> global mean.

    Temp_C_observed keeps the pre-imputation values so the temperature
    slider can be bounded by what was really measured at each site.
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
    features = SITE_FEATURES[site]
    d = df[(df["SiteGroup"] == site) & (df["Season"] != "Unknown")].copy()
    if "Day_Night" in features:
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


# ------------------------------------------------------------
# Model training
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Growing decision trees…")
def train_models(_df):
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
        d = _site_frame(_df, site)
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


def build_input_row(model, season, day_night, temp_c, group_solo):
    """Turn the sidebar selections into a single one-hot encoded row."""
    row = pd.DataFrame(0.0, index=[0], columns=model["columns"])
    for column, value in (("Season", season),
                          ("Day_Night", day_night),
                          ("Group_Solo", group_solo)):
        if value is None:
            continue
        name = f"{column}_{value}"
        if name in row.columns:
            row.loc[0, name] = 1
    row.loc[0, TEMP_FEATURE] = temp_c
    return row


# ------------------------------------------------------------
# Tree drawing & path animation
# ------------------------------------------------------------
def _node_annotations(tree, ax, feature_labels):
    """
    Draw the tree and return one annotation per node, in node-index order.

    plot_tree returns the node boxes plus the "True"/"False" edge labels;
    only the node boxes carry a bbox patch, so filtering on that recovers
    exactly the nodes in the order sklearn indexes them.
    """
    annotations = plot_tree(
        tree, ax=ax, filled=False, impurity=False, proportion=False,
        feature_names=feature_labels, class_names=list(tree.classes_),
        fontsize=8, rounded=True,
    )
    nodes = [a for a in annotations if a.get_bbox_patch() is not None]
    # Guard the index mapping: if a future sklearn changes what plot_tree
    # returns, fail loudly instead of highlighting the wrong boxes.
    assert len(nodes) == tree.tree_.node_count, (
        f"plot_tree returned {len(nodes)} node boxes for "
        f"{tree.tree_.node_count} nodes — the highlight mapping is unsafe."
    )
    edges = [a for a in annotations if a.get_bbox_patch() is None]
    return nodes, edges


def _label_root_branches(edges, font_size):
    """
    sklearn labels only the root's two edges, as True/False. Every question
    here is phrased so the left branch is the "no" case, so say that
    outright — it sets the convention for the rest of the tree.
    """
    for annotation in edges:
        text = annotation.get_text().strip()
        if text in ("True", "False"):
            annotation.set_text("No" if text == "True" else "Yes")
            annotation.set_fontsize(font_size)
            annotation.set_color("#444444")


# One-hot column -> the plain-English question that column asks.
_QUESTIONS = {
    "Day_Night_Night":   "At night?",
    "Day_Night_Day":     "In daylight?",
    "Group_Solo_Group":  "In a group?",
    "Group_Solo_Single": "On its own?",
}


def _node_text(model, index):
    """
    The one line a node shows: a question, or — at a leaf — a species.

    sklearn always sends the "<= threshold" case down the left branch. For
    a one-hot column that means left is "not that category", so temperature
    is phrased as "Warmer than X?" to match. Every node then reads the same
    way: left is No, right is Yes.
    """
    t = model["tree"].tree_
    if t.children_left[index] == -1:
        text = model["classes"][int(t.value[index][0].argmax())]
    else:
        column = model["columns"][t.feature[index]]
        if column == TEMP_FEATURE:
            text = f"Warmer than {t.threshold[index]:.1f}°C?"
        elif column in _QUESTIONS:
            text = _QUESTIONS[column]
        elif column.startswith("Season_"):
            text = column.replace("Season_", "").split(" (")[0] + "?"
        else:
            text = column
    # Long species names are wider than their slot in the layout and
    # would overlap their neighbours, so wrap rather than clip.
    return textwrap.fill(text, LABEL_WRAP)


def draw_tree(model, highlight=(), leaf=None):
    """Render the tree with the visited nodes highlighted."""
    tree = model["tree"]

    # plot_tree gives every leaf an equal horizontal slot, so a wide tree
    # needs a wider canvas. Kept as narrow as the labels allow: Streamlit
    # scales the image down to the container, shrinking the text with it.
    n_leaves = tree.get_n_leaves()
    fig_width = max(11.0, min(18.0, n_leaves * 1.15))
    font_size = 11 if n_leaves <= 8 else 9

    # Object-oriented Figure rather than plt.subplots: pyplot keeps every
    # figure it creates in a global registry, and this app redraws on each
    # interaction, so those would pile up until Agg fails to allocate.
    fig = Figure(figsize=(fig_width, 6.5))
    FigureCanvasAgg(fig)          # plot_tree needs a real renderer
    ax = fig.subplots()
    nodes, edges = _node_annotations(tree, ax, list(model["columns"]))

    # Strip sklearn's samples/value/class block down to a single line.
    # Done before the draw so each box shrinks to fit its new, shorter text.
    for i, annotation in enumerate(nodes):
        annotation.set_text(_node_text(model, i))
        annotation.set_fontsize(font_size)
    _label_root_branches(edges, font_size)
    fig.canvas.draw()   # bbox patches only exist once drawn

    highlight = set(highlight)
    for i, annotation in enumerate(nodes):
        patch = annotation.get_bbox_patch()
        if i == leaf:
            patch.set_facecolor(LEAF_COLOR)
            patch.set_edgecolor("black")
            annotation.set_color("white")
        elif i in highlight:
            patch.set_facecolor(PATH_COLOR)
            patch.set_edgecolor("black")
            annotation.set_color("white")
        else:
            patch.set_facecolor(MUTED_COLOR)
            patch.set_edgecolor("#BBBBBB")
            annotation.set_color("#555555")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def decision_path_nodes(model, row):
    """Ordered node indices the input visits, ending at its leaf."""
    return list(model["tree"].decision_path(row).indices)


# ------------------------------------------------------------
# Result rendering
# ------------------------------------------------------------
def shortlist_chart(model, row):
    """
    Horizontal bar of the leaf's class distribution, most likely first.

    Top-1 accuracy on these four features is modest, but the correct
    species lands in the top 3 roughly 70-84% of the time — so the
    ranked list, not a single guess, is the honest answer.
    """
    proba = model["tree"].predict_proba(row)[0]
    ranked = (pd.DataFrame({"Species": model["classes"], "Probability": proba})
              .query("Probability > 0")
              .sort_values("Probability", ascending=False)
              .reset_index(drop=True))

    # Plotly stacks the first horizontal bar at the bottom, so feed it in
    # ascending order to get the most likely species on top. Ordering this
    # way rather than reversing the axis keeps each bar with its own label.
    plot_order = ranked.iloc[::-1]
    species = plot_order["Species"].tolist()
    values = plot_order["Probability"].tolist()
    # Top three keep the accent colour; the tail is muted.
    colors = [PATH_COLOR if s in set(ranked["Species"].head(3)) else CB_PALETTE[5]
              for s in species]

    fig = go.Figure(go.Bar(
        x=values, y=species, orientation="h",
        marker_color=colors,
        text=[f"{v:.0%}" for v in values],
        # textangle=0 stops plotly rotating the label sideways when a bar
        # is too narrow for it; it moves the label outside the bar instead.
        textposition="auto", insidetextanchor="middle", textangle=0,
        cliponaxis=False, textfont={"size": 13},
        hoverinfo="skip", hovertemplate=None,   # the bar already states its %
    ))
    fig.update_layout(
        xaxis={"visible": False, "range": [0, max(values) * 1.18]},
        yaxis={"type": "category", "categoryorder": "array",
               "categoryarray": species, "ticksuffix": "  ",
               "automargin": True, "tickfont": {"size": 13}},
        height=60 + 40 * len(species),
        margin={"r": 10, "t": 10, "b": 10},
        showlegend=False, bargap=0.25,
    )
    return ranked, fig


def render_result(model, row):
    ranked, fig = shortlist_chart(model, row)
    top = ranked.iloc[0]

    st.subheader(f"Most likely: {top['Species']}")
    runners = ", ".join(ranked["Species"][1:3])
    if runners:
        st.caption(f"Then: {runners}")

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="What could it be?", page_icon="🦈",
                       layout="wide")
    st.title("🦈 I saw something in the water — what could it be?")
    st.caption(
        "Three decision trees, one per site, trained on real sightings from "
        "the Israeli Mediterranean coast. Describe what you saw and watch the "
        "tree work its way down to an answer."
    )

    df = prepare_data()
    models = train_models(df)

    site = st.radio("Where did you see it?", SITES, horizontal=True)
    model = models[site]
    features = SITE_FEATURES[site]
    st.caption(f"**{site}** — {SITE_BLURBS[site]} "
               f"Tree trained on {model['n_rows']:,} sightings.")

    st.markdown("### Describe the sighting")
    columns = st.columns(len(features) + 1)

    season = columns[0].selectbox("Season", SEASONS, index=2)

    next_col = 1
    day_night = None
    if "Day_Night" in features:
        day_night = columns[next_col].radio("Day or night?", ["Day", "Night"],
                                            horizontal=True)
        next_col += 1

    temp_c = columns[next_col].slider(
        "Water temperature (°C)",
        min_value=round(model["temp_min"], 1),
        max_value=round(model["temp_max"], 1),
        value=round(model["temp_default"], 1), step=0.1)
    next_col += 1

    group_solo = columns[next_col].radio("Alone or in a group?",
                                         ["Single", "Group"], horizontal=True)

    row = build_input_row(model, season, day_night, temp_c, group_solo)
    path = decision_path_nodes(model, row)
    leaf = path[-1]

    st.markdown("### The tree's reasoning")
    animate = st.button("▶ Animate the path", type="primary")
    canvas = st.empty()

    if animate:
        # Reveal the path one node at a time, ending on the leaf.
        for step in range(1, len(path) + 1):
            visited = path[:step]
            reached = leaf if step == len(path) else None
            canvas.pyplot(draw_tree(model, highlight=visited, leaf=reached))
            if step < len(path):
                time.sleep(FRAME_DELAY)
    else:
        canvas.pyplot(draw_tree(model, highlight=path, leaf=leaf))

    st.markdown("### The answer")
    render_result(model, row)


if __name__ == "__main__":
    main()
