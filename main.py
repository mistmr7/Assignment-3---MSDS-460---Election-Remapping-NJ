import geopandas as gpd
import numpy as np
import pandas as pd
import glpk

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
    listSolvers,
)

print(listSolvers(onlyAvailable=True))
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

# Create initial DataFrame of County Information
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

print(county_info)

df_county_names = pd.DataFrame(county_names, columns=["County"])
df = pd.DataFrame()
df["County"] = county_names
df["CountySort"] = county_id

shapefile_new_jersey = gpd.read_file("us-county-boundaries.shp")

map_population_by_county_data = shapefile_new_jersey.merge(
    county_info,
    left_on="name",
    right_on="County_Name",
    suffixes=("_left", "_right"),
)
county_populations = np.array(county_info.Population)
state_population = sum(county_populations)
print(county_info)

n_counties = 21
n_districts = 12
variable_names = [
    str(i) + "A" + str(j)
    for j in range(1, n_districts + 1)
    for i in range(1, n_counties + 1)
]
variable_names.sort()

population_split = 388369
min_dist = 388369
max_dist = 1165108

# Create the model and choose whether to minimize or maximize
model = LpProblem("Supply-Demand-Problem", LpMinimize)

# Declare Variables
# The Decision Variable is 1 if the county is assigned to the district.
DV_variable_y = LpVariable.matrix("Y", variable_names, cat="Binary")
assignment = np.array(DV_variable_y).reshape(21, 12)

# The Decision Variable is the population allocated to the district.
DV_variable_x = LpVariable.matrix("X", variable_names, cat="Integer", lowBound=0)
allocation = np.array(DV_variable_x).reshape(21, 12)

# Write the objective
objective_function = lpSum(assignment)
model += objective_function

# Constraints

# Allocate 100% of the population from each county.
for i in range(n_counties):
    model += (
        lpSum(allocation[i][j] for j in range(n_districts)) == county_populations[i],
        "Allocate All " + str(i),
    )

# This constraint makes assignment required for allocation.
# sum(county_populations) is the "big M"
for i in range(n_counties):
    for j in range(n_districts):
        model += (
            allocation[i][j] <= sum(county_populations) * assignment[i][j],
            "Allocation assignment " + str(i) + " " + str(j),
        )

for j in range(n_districts):
    # Atlantic County borders Burlington[2], Camden[3], Cape May[4], Cumberland[5], Gloucester[7], and Ocean[14]
    model += (
        assignment[0][j]
        <= assignment[2][j]
        + assignment[3][j]
        + assignment[4][j]
        + assignment[5][j]
        + assignment[7][j]
        + assignment[14][j]
    )
    # Bergen County borders Essex[6], Hudson[8], and Passaic[15]
    model += assignment[1][j] <= assignment[6][j] + assignment[8][j] + assignment[15][j]
    # Burlington County borders Atlantic[0], Camden[3], Mercer[10], Monmouth[12], and Ocean[14]
    model += (
        assignment[2][j]
        <= assignment[0][j]
        + assignment[3][j]
        + assignment[10][j]
        + assignment[12][j]
        + assignment[14][j]
    )
    # Camden County borders Atlantic[0], Burlington[2], and Gloucester[7]
    model += assignment[3][j] <= assignment[0][j] + assignment[2][j] + assignment[7][j]
    # Cape May County borders Atlantic[0] and Cumberland[5]
    model += assignment[4][j] <= assignment[0][j] + assignment[5][j]
    # Cumberland County borders Atlantic[0], Cape May[4], Gloucester[7], and Salem[16]
    model += (
        assignment[5][j]
        <= assignment[0][j] + assignment[4][j] + assignment[7][j] + assignment[16][j]
    )
    # Essex County borders Bergen[1], Hudson[8], Morris[13], Passaic[15], and Union[19]
    model += (
        assignment[6][j]
        <= assignment[1][j]
        + assignment[8][j]
        + assignment[13][j]
        + assignment[15][j]
        + assignment[19][j]
    )
    # Gloucester County borders Atlantic[0], Camden[3], Cumberland[5], and Salem[16]
    model += (
        assignment[7][j]
        <= assignment[0][j] + assignment[3][j] + assignment[5][j] + assignment[16][j]
    )
    # Hudson County borders Bergen[1], Essex[6], and Union[19]
    model += assignment[8][j] <= assignment[1][j] + assignment[6][j] + assignment[19][j]
    # Hunterdon County borders Mercer[10], Morris[13], Somerset[17], and Warren[20]
    model += (
        assignment[9][j]
        <= assignment[10][j] + assignment[13][j] + assignment[17][j] + assignment[20][j]
    )
    # Mercer county borders Burlington[2], Hunterdon[9], Middlesex[11], Monmouth[12], and Somerset[17]
    model += (
        assignment[10][j]
        <= assignment[2][j]
        + assignment[9][j]
        + assignment[11][j]
        + assignment[12][j]
        + assignment[17][j]
    )
    # Middlesex county borders Mercer[10], Monmouth[12], Somerset[17], and Union[19]
    model += (
        assignment[11][j]
        <= assignment[10][j] + assignment[12][j] + assignment[17][j] + assignment[19][j]
    )
    # Monmouth County borders Burlington[2], Mercer[10], Middlesex[11], and Ocean[14]
    model += (
        assignment[12][j]
        <= assignment[2][j] + assignment[10][j] + assignment[11][j] + assignment[14][j]
    )
    # Morris County borders Essex[6], Hunterdon[9], Passaic[15], Somerset[17], Sussex[18], Union[19], and Warren[20]
    model += (
        assignment[13][j]
        <= assignment[6][j]
        + assignment[9][j]
        + assignment[15][j]
        + assignment[17][j]
        + assignment[18][j]
        + assignment[19][j]
        + assignment[20][j]
    )
    # Ocean County borders Atlantic[0], Burlington[2], and Monmouth[12]
    model += (
        assignment[14][j] <= assignment[0][j] + assignment[2][j] + assignment[12][j]
    )
    # Passaic County borders Bergen[1], Essex[6], Morris[13], and Sussex[18]
    model += (
        assignment[15][j]
        <= assignment[1][j] + assignment[6][j] + assignment[13][j] + assignment[18][j]
    )
    # Salem County borders Cumberland[5] and Gloucester[7]
    model += assignment[16][j] <= assignment[5][j] + assignment[7][j]
    # Somerset County borders Hunterdon[9], Mercer[10], Middlesex[11], Morris[13], and Union[19]
    model += (
        assignment[17][j]
        <= assignment[9][j]
        + assignment[10][j]
        + assignment[11][j]
        + assignment[13][j]
        + assignment[19][j]
    )
    # Sussex County borders Morris[13], Passaic[15], and Warren[20]
    model += (
        assignment[18][j] <= assignment[13][j] + assignment[15][j] + assignment[20][j]
    )
    # Union County borders Essex[6], Hudson[8], Middlesex[11], Morris[13], and Somerset[17]
    model += (
        assignment[19][j]
        <= assignment[6][j]
        + assignment[8][j]
        + assignment[11][j]
        + assignment[13][j]
        + assignment[17][j]
    )
    # Warren County borders Hunterdon[9], Morris[13], and Sussex[18]
    model += (
        assignment[20][j] <= assignment[9][j] + assignment[13][j] + assignment[18][j]
    )


# District size constraints, in order to keep the size of districts by population similar
for j in range(n_districts):
    model += (
        lpSum(allocation[i][j] for i in range(n_counties)) <= max_dist,
        "District Size Maximum " + str(j),
    )
    model += (
        lpSum(allocation[i][j] for i in range(n_counties)) >= min_dist,
        "District Size Minimum " + str(j),
    )

# Only allow counties that meet certain critera to be split among multiple districts
# A county must have population > population_split to be split among up to two districts
for i in range(n_counties):  # added
    if county_populations[i] <= population_split:
        model += (
            lpSum(assignment[i][j] for j in range(n_districts)) <= 1,
            "Unique Assignment " + str(i),
        )
    else:
        model += (
            lpSum(assignment[i][j] for j in range(n_districts)) <= 2,
            "Up-to-two Assignments " + str(i),
        )

model.solve(GLPK_CMD(options=["--mipgap", "0.055", "--gomory"]))
print("The model status is: ", LpStatus[model.status])
print("The objective value is: ", value(objective_function))

# Access the results
for i in range(n_counties):
    for j in range(n_districts):
        if allocation[i][j].value() > 0:
            print(
                "County %d assigned to district %d: " % (i, j), allocation[i][j].value()
            )

# Prepare data for visualizing the results
result_value = []
for i in range(n_counties):
    for j in range(n_districts):
        var_output = {
            "County": i,
            "District": j + 1,
            "Assignment": int(assignment[i][j].value() * (j + 1)),
            "Allocation": allocation[i][j].value(),
        }
        result_value.append(var_output)

results = pd.DataFrame(result_value)
results = results[results["Assignment"] != 0]
results = results.sort_values(["County", "District"])
results = results.merge(
    df, left_on="County", right_on="CountySort", suffixes=("_ID", "_Name")
)
results["Multiple_County_Name"] = results["County_Name"].shift(periods=1)
results["Multiple_District"] = (
    results["District"].shift(periods=1).fillna(99).astype(int)
)

# Edit the assignment for the case when a county has multiple assignments
for i in range(0, len(results)):
    if results["County_Name"].loc[i] == results["Multiple_County_Name"].loc[i]:
        results.loc[i, "Assignment"] = int(
            str(results["District"].loc[i]) + str(results["Multiple_District"].loc[i])
        )
results = results.sort_values(["District", "County_Name"])
results.index = results["County_ID"]

color_dict = {
    1: "khaki",
    2: "pink",
    3: "mediumaquamarine",
    4: "plum",
    5: "paleturquoise",
    6: "lightcoral",
    7: "orange",
    8: "yellow",
    9: "red",
    10: "blue",
    11: "green",
    12: "grey",
}


def nj_map(map_data):
    # Create three maps to visualize the results.
    # (1) A map with population labels
    # (2) A map with county labels
    # (3) A map with county IDs

    plot_map_population_labels = (
        ggplot(map_data)
        + geom_map(aes(fill=str("Assignment")))
        + geom_label(
            aes(x="Longitude", y="Latitude", label="Population2020e", size=2),
            show_legend=False,
        )
        + theme_minimal()
        + theme(
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_title_x=element_blank(),
            axis_title_y=element_blank(),
            axis_ticks=element_blank(),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            figure_size=(7, 4),
        )
        + scale_fill_manual(values=color_dict)
    )

    plot_map_county_labels = (
        ggplot(map_data)
        + geom_map(aes(fill=str("Assignment")))
        + geom_label(
            aes(x="Longitude", y="Latitude", label="County_Name_left", size=2),
            show_legend=False,
        )
        + theme_minimal()
        + theme(
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_title_x=element_blank(),
            axis_title_y=element_blank(),
            axis_ticks=element_blank(),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            figure_size=(7, 4),
        )
        + scale_fill_manual(values=color_dict)
    )

    plot_map_county_ids = (
        ggplot(map_data)
        + geom_map(aes(fill=str("Assignment")))
        + geom_label(
            aes(x="Longitude", y="Latitude", label="County_ID", size=5),
            show_legend=False,
        )
        + theme_minimal()
        + theme(
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_title_x=element_blank(),
            axis_title_y=element_blank(),
            axis_ticks=element_blank(),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            figure_size=(7, 4),
        )
        + scale_fill_manual(values=color_dict)
    )

    return plot_map_population_labels, plot_map_county_labels, plot_map_county_ids


map_first_pass = shapefile_new_jersey.merge(
    results, left_on="name", right_on="County_Name", suffixes=("_left", "_right")
)
map_first_pass["District"] = map_first_pass["District"] + 1
map_first_pass_labels = map_first_pass.merge(
    county_info,
    left_on="County_ID",
    right_on="County_ID",
    suffixes=("_left", "_right"),
)
map_first_pass_labels["District"] = map_first_pass_labels["District"].astype("category")
map_first_pass_labels["Assignment"] = map_first_pass_labels["Assignment"].astype(
    "category"
)

nj_map(map_first_pass_labels)
