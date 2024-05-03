import geopandas as gpd
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", None)

from PIL import Image, ImageOps
from plotnine import (
    ggplot,
    aes,
    geom_map,
    geom_text,
    geom_label,
    ggtitle,
    element_blank,
    element_rect,
    scale_fill_manual,
    theme_minimal,
    theme,
)
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    PULP_CBC_CMD,
    GLPK_CMD,
    LpStatus,
    value,
)

# Arbitrary County IDs
county_id = np.arange(0, 21)

# County Names
county_names = [
    "Atlantic",
    "Bergen",
    "Burlington",
    "Camden",
    "Cape May",
    "Cumberland",
    "Essex",
    "Gloucester",
    "Hudson",
    "Hunterdon",
    "Mercer",
    "Middlesex",
    "Monmouth",
    "Morris",
    "Ocean",
    "Passaic",
    "Salem",
    "Somerset",
    "Sussex",
    "Union",
    "Warren",
]

# County population 2024
county_populations = [
    275044,
    961932,
    472233,
    529743,
    93815,
    153305,
    852510,
    310079,
    708563,
    130561,
    382563,
    866152,
    641370,
    517627,
    662731,
    513156,
    65519,
    350637,
    146689,
    575035,
    111601,
]

# Latitude and Longitude Coordinates for Each County
county_lat_long = [
    (39.4688295112, -74.6337304316),
    (40.9596717781, -74.0742255725),
    (39.8777172962, -74.6680674072),
    (39.8035392391, -74.9597543553),
    (39.0850981086, -74.8499770137),
    (39.3280985807, -75.1293182687),
    (40.7872147845, -74.2470601526),
    (39.7172646082, -75.1414130536),
    (40.7309049975, -74.0759554422),
    (40.5672969004, -74.9122477682),
    (40.2834374242, -74.7017524053),
    (40.4400836596, -74.4088356547),
    (40.2875173476, -74.1582023922),
    (40.8620042241, -74.5445106674),
    (39.8659324961, -74.2499159319),
    (41.0343060095, -74.3007756821),
    (39.5767751714, -75.3579856904),
    (40.5635069447, -74.6163366246),
    (41.1392969335, -74.6908006832),
    (40.6598994131, -74.3081414246),
    (40.8571485251, -74.9973498515),
]

# % White ethnicity by county
county_percent_white = [
    57.01,
    56.54,
    64.17,
    55.35,
    86.91,
    48.67,
    30.67,
    74.28,
    35.08,
    81.79,
    46.65,
    41.76,
    74.22,
    68.55,
    80.58,
    44.74,
    71.12,
    52.90,
    83.09,
    41.14,
    77.84,
]

# % Voting for Joe Biden (D) 2020 Election
county_2020_dem_voting = [
    52.7,
    57.7,
    59.1,
    66.2,
    41.5,
    52.5,
    77.3,
    50.2,
    72.6,
    46.8,
    69.4,
    60.4,
    47.9,
    51.4,
    35,
    57.7,
    42.7,
    59.8,
    39.2,
    67.3,
    41,
]

# % Voting for Donald Trump (R) 2020 Election
county_2020_rep_voting = [
    46,
    41.2,
    39.5,
    32.6,
    75.5,
    46.4,
    21.9,
    48.3,
    26.3,
    51.2,
    29.2,
    38.3,
    50.7,
    47.2,
    63.8,
    41.1,
    55.5,
    38.7,
    58.8,
    31.6,
    57.2,
]

county_info = pd.DataFrame(
    {
        "County_ID": county_id,
        "County_Name": county_names,
        "Population": county_populations,
        "White_Percentage": county_percent_white,
        "Democratic_Voting": county_2020_dem_voting,
        "Republican_Voting": county_2020_rep_voting,
        "Latitude": [county[0] for county in county_lat_long],
        "Longitude": [county[1] for county in county_lat_long],
    }
)
