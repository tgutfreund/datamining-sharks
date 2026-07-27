# ============================================================
# data_loader.py — shared data loader and derived columns
# Used by every viz_*/analysis_* script in this project.
#
# Single source of truth for column mapping so each analysis
# script can stay short and consistent.
#
# pip install pandas numpy
# ============================================================

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
# Site-fixed enriched workbook (corrected coastal coordinates).
INPUT_FILE = "observations_copernicus_enriched_sites_fixed.xlsx"
INPUT_SHEET = "ALL_ENRICHED"
OUTDIR = "outputs"

# Only these raw columns matter — everything else in the workbook
# is Copernicus QA / intermediate noise we deliberately skip.
RELEVANT_COLS = [
    "Time", "Species", "Count", "Report type", "Description",
    "Length (cm)", "Max Depth [m]", "Distance [m]", "Temp [C]",
    "water_temp_copernicus",
    "site_standardized", "site_lat", "site_lon",
    "obs_datetime", "year", "month",
]

# Colorblind-friendly qualitative palette (Okabe-Ito + extension).
# Reused across scripts to maximize perception consistency.
CB_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
    "#56B4E9", "#F0E442", "#000000", "#999999", "#117733",
    "#882255", "#44AA99", "#332288", "#AA4499", "#88CCEE",
    "#DDCC77", "#661100", "#6699CC", "#888888", "#AA7744",
]

# ------------------------------------------------------------
# Scientific name -> English common name
# (kept in sync with network_analysis.py)
# ------------------------------------------------------------
COMMON_NAMES = {
    "Taeniurops grabatus":        "Round Ribbontail Ray",
    "Torpedo torpedo":            "Common Torpedo Ray",
    "Glaucostegus cemiculus":     "Blackchin Guitarfish",
    "Aetomylaeus bovinus":        "Banded Eagle Ray",
    "Carcharhinus obscurus":      "Dusky Shark",
    "Dasyatis pastinaca":         "Common Stingray",
    "Dasyatis spp.":              "Stingray (spp.)",
    "Himantura uarnak":           "Honeycomb Stingray",
    "Dasyatis marmorata":         "Marbled Stingray",
    "Himantura leoparda":         "Leopard Whipray",
    "Torpedo marmorata":          "Marbled Torpedo Ray",
    "Gymnura altavela":           "Spiny Butterfly Ray",
    "Rhinoptera marginata":       "Lusitanian Cownose Ray",
    "Guitarfish indet.":          "Guitarfish (unidentified)",
    "Carcharhinus plumbeus":      "Sandbar Shark",
    "Carcharhinus spp.":          "Requiem Shark (spp.)",
    "Himantura spp.":             "Whipray (spp.)",
    "Isurus oxyrinchus":          "Shortfin Mako Shark",
    "Hexanchus griseus":          "Bluntnose Sixgill Shark",
    "Alopias superciliosus":      "Bigeye Thresher Shark",
    "Sphyrna zygaena":            "Smooth Hammerhead Shark",
    "Cetorhinus maximus":         "Basking Shark",
    "Selachii indet.":            "Shark (unidentified)",
    "Galeus melastomus":          "Blackmouth Catshark",
    "Mobula mobular":             "Giant Devil Ray",
    "Centrophorus granulosus":    "Gulper Shark",
    "Dipturus oxyrinchus":        "Longnosed Skate",
    "Rhinobatos rhinobatos":      "Common Guitarfish",
    "Pteroplatytrygon violacea":  "Pelagic Stingray",
    "Squatina aculeata":          "Sawback Angelshark",
    "Raja clavata":               "Thornback Ray",
    "Mustelus mustelus":          "Smooth-hound Shark",
    "Prionace glauca":            "Blue Shark",
    "Raja miraletus":             "Brown Ray",
    "Torpedo spp.":               "Torpedo Ray (spp.)",
    "Carcharhinus brevipinna":    "Spinner Shark",
    "Squalus blainville":         "Longnose Spurdog",
    "Centrophoridae indet.":      "Gulper Shark (unidentified)",
    "Etmopterus spinax":          "Velvet Belly Lanternshark",
    "Scyliorhinus canicula":      "Small-spotted Catshark",
    "Alopias vulpinus":           "Common Thresher Shark",
    "Rhincodon typus":            "Whale Shark",
    "Raja spp.":                  "Skate (spp.)",
    "Lusitanian cownose ray":     "Lusitanian Cownose Ray",
    "Spiny butterfly ray":        "Spiny Butterfly Ray",
    "Raja spp":                   "Skate (spp.)",
}

# Genus prefixes that are sharks. Everything elasmobranch that is
# not in this set is treated as a ray/skate/guitarfish.
SHARK_GENERA = (
    "Carcharhinus", "Isurus", "Sphyrna", "Alopias", "Hexanchus",
    "Prionace", "Mustelus", "Squalus", "Squatina", "Scyliorhinus",
    "Galeus", "Etmopterus", "Centrophorus", "Centrophoridae",
    "Cetorhinus", "Rhincodon", "Selachii",
)


def to_common(scientific_name):
    """Scientific name -> English common name, fallback to the original."""
    return COMMON_NAMES.get(str(scientific_name).strip(), str(scientific_name).strip())


def classify_shark_ray(scientific_name):
    """Return 1 if the species is a shark, 0 if it is a ray/skate/guitarfish."""
    s = str(scientific_name).strip()
    return 1 if s.startswith(SHARK_GENERA) else 0


def month_to_season(m):
    """Meteorological season label from a month number."""
    if pd.isna(m):
        return "Unknown"
    m = int(m)
    if m in (12, 1, 2):
        return "Winter (DJF)"
    if m in (3, 4, 5):
        return "Spring (MAM)"
    if m in (6, 7, 8):
        return "Summer (JJA)"
    return "Autumn (SON)"


# Raw Time labels -> coarse time-of-day buckets.
_TIME_OF_DAY_MAP = {
    "morning": "Morning",
    "dawn": "Morning",
    "light": "Daytime",
    "afternoon": "Afternoon",
    "evening": "Evening",
    "dusk": "Evening",
    "night": "Night",
    "dark": "Night",
    "unspecified": "Unknown",
}


def normalize_time_of_day(value):
    if pd.isna(value):
        return "Unknown"
    return _TIME_OF_DAY_MAP.get(str(value).strip().lower(), "Unknown")


def to_day_night(time_of_day):
    """Collapse the time-of-day bucket to a strict Day / Night split."""
    if time_of_day == "Night":
        return "Night"
    if time_of_day == "Unknown":
        return "Unknown"
    return "Day"


def to_group_solo(count_n):
    """Single animal vs. a group of two or more."""
    return "Group" if count_n > 1 else "Single"


# Report type -> coarse observation source group.
_SOURCE_MAP = {
    "scuba diving": "Diving",
    "freediving": "Diving",
    "snorkeling": "Diving",
    "spearfishing (freediving)": "Fishing",
    "anglers": "Fishing",
    "longlining": "Fishing",
    "rods on boat": "Fishing",
    "fishing - other": "Fishing",
    "swimming": "Beach & Swim",
    "beach goer": "Beach & Swim",
    "drone": "Aerial",
    "boat": "Boat & Port",
    "port": "Boat & Port",
    "other - non-fishing": "Other",
}


def normalize_source(value):
    if pd.isna(value):
        return "Unknown"
    return _SOURCE_MAP.get(str(value).strip().lower(), "Other")


def ensure_outdir(sub=None):
    """
    Create the shared outputs directory (optionally a per-question
    sub-folder like 'q1') and return its path.
    """
    path = os.path.join(OUTDIR, sub) if sub else OUTDIR
    os.makedirs(path, exist_ok=True)
    return path


def load_data(path=INPUT_FILE, sheet=INPUT_SHEET):
    """
    Load the enriched observations and attach normalized, analysis-ready
    columns. All scripts call this so column naming stays consistent.

    Returns a DataFrame with these derived columns added:
        Temp_C, Depth_m, Length_cm, Distance_m,
        Latitude, Longitude,
        Season, Time_of_Day, Day_Night, Group_Solo, Source,
        Common_EN, Is_Shark, Animal_Type
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Run 01_clean_sites_and_geocode.py, "
            "02_enrich_copernicus_temperature.py, and 03_merge_duplicate_sites.py first."
        )

    is_excel = str(path).lower().endswith((".xlsx", ".xls"))

    # Peek at the header so we can request only the columns that exist,
    # skipping all the Copernicus QA / intermediate noise.
    if is_excel:
        header = pd.read_excel(path, sheet_name=sheet, nrows=0)
    else:
        header = pd.read_csv(path, nrows=0)
    header.columns = [str(c).strip() for c in header.columns]
    usecols = [c for c in RELEVANT_COLS if c in header.columns]

    if is_excel:
        df = pd.read_excel(path, sheet_name=sheet, usecols=usecols)
    else:
        df = pd.read_csv(path, usecols=usecols)
    df.columns = [str(c).strip() for c in df.columns]

    # --- datetime / temporal ---
    df["obs_datetime"] = pd.to_datetime(df.get("obs_datetime"), errors="coerce")
    if "month" not in df.columns:
        df["month"] = df["obs_datetime"].dt.month
    if "year" not in df.columns:
        df["year"] = df["obs_datetime"].dt.year
    df["Season"] = df["month"].apply(month_to_season)
    df["Time_of_Day"] = df.get("Time").apply(normalize_time_of_day) \
        if "Time" in df.columns else "Unknown"
    df["Day_Night"] = df["Time_of_Day"].apply(to_day_night)

    # --- environmental (numeric) ---
    field_temp = pd.to_numeric(df.get("Temp [C]"), errors="coerce")
    cop_temp = pd.to_numeric(df.get("water_temp_copernicus"), errors="coerce")
    # Prefer the reliable Copernicus temperature, fall back to field reading.
    df["Temp_C"] = cop_temp.fillna(field_temp)
    df["Depth_m"] = pd.to_numeric(df.get("Max Depth [m]"), errors="coerce")
    df["Length_cm"] = pd.to_numeric(df.get("Length (cm)"), errors="coerce")
    df["Distance_m"] = pd.to_numeric(df.get("Distance [m]"), errors="coerce")
    df["Count_n"] = pd.to_numeric(df.get("Count"), errors="coerce").fillna(1).clip(lower=1)
    df["Group_Solo"] = df["Count_n"].apply(to_group_solo)

    # --- geography ---
    df["Latitude"] = pd.to_numeric(df.get("site_lat"), errors="coerce")
    df["Longitude"] = pd.to_numeric(df.get("site_lon"), errors="coerce")

    # --- categorical ---
    df["Source"] = df.get("Report type").apply(normalize_source) \
        if "Report type" in df.columns else "Unknown"
    df["Common_EN"] = df.get("Species").apply(to_common)
    df["Is_Shark"] = df.get("Species").apply(classify_shark_ray)
    df["Animal_Type"] = np.where(df["Is_Shark"] == 1, "Shark", "Ray")

    return df


if __name__ == "__main__":
    # Quick self-test / data summary.
    d = load_data()
    print(f"Loaded {len(d)} rows")
    print("Sharks vs rays:\n", d["Animal_Type"].value_counts())
    print("\nSeasons:\n", d["Season"].value_counts())
    print("\nTime of day:\n", d["Time_of_Day"].value_counts())
    print("\nDay vs night:\n", d["Day_Night"].value_counts())
    print("\nSingle vs group:\n", d["Group_Solo"].value_counts())
    print("\nSources:\n", d["Source"].value_counts())
