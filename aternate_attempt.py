# Imports
import pulp
import numpy as np
import pandas as pd
import geopandas as gpd

from plotnine import (
    ggplot,
    aes,
    geom_map,
    geom_text,
    geom_label,
    ggtitle,
    element_blank,
    scale_fill_manual,
    theme_minimal,
    theme,
)
from icecream import ic

# Assign arbitrary county IDs
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
print(county_info)
num_counties = len(county_populations)
num_districts = 12
population_split = 348787
mean_population = sum(county_populations) / num_districts
shapefile_new_jersey = gpd.read_file("us-county-boundaries.shp")

map_population_by_county_data = shapefile_new_jersey.merge(
    county_info,
    left_on="name",
    right_on="County_Name",
    suffixes=("_left", "_right"),
)

# Create county adjacency matrix
adjacency_matrix = {
    "0": {
        "0": 0,
        "1": 0,
        "2": 1,
        "3": 1,
        "4": 1,
        "5": 1,
        "6": 0,
        "7": 1,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 0,
        "14": 1,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "1": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 1,
        "7": 0,
        "8": 1,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 0,
        "14": 0,
        "15": 1,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "2": {
        "0": 1,
        "1": 0,
        "2": 0,
        "3": 1,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 1,
        "11": 0,
        "12": 1,
        "13": 0,
        "14": 1,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "3": {
        "0": 1,
        "1": 0,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 1,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 0,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "4": {
        "0": 1,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 1,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 0,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "5": {
        "0": 1,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 1,
        "5": 0,
        "6": 0,
        "7": 1,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 0,
        "14": 0,
        "15": 0,
        "16": 1,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "6": {
        "0": 0,
        "1": 1,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 1,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 1,
        "14": 0,
        "15": 1,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 1,
        "20": 0,
    },
    "7": {
        "0": 1,
        "1": 0,
        "2": 0,
        "3": 1,
        "4": 0,
        "5": 1,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 0,
        "14": 0,
        "15": 0,
        "16": 1,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "8": {
        "0": 0,
        "1": 1,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 1,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 0,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 1,
        "20": 0,
    },
    "9": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 1,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 1,
        "11": 0,
        "12": 0,
        "13": 1,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 1,
        "18": 0,
        "19": 0,
        "20": 1,
    },
    "10": {
        "0": 0,
        "1": 0,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 1,
        "10": 0,
        "11": 1,
        "12": 1,
        "13": 0,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 1,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "11": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 1,
        "11": 0,
        "12": 1,
        "13": 0,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 1,
        "18": 0,
        "19": 1,
        "20": 0,
    },
    "12": {
        "0": 0,
        "1": 0,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 1,
        "11": 1,
        "12": 0,
        "13": 0,
        "14": 1,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "13": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 1,
        "7": 0,
        "8": 0,
        "9": 1,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 0,
        "14": 0,
        "15": 1,
        "16": 0,
        "17": 1,
        "18": 1,
        "19": 1,
        "20": 1,
    },
    "14": {
        "0": 1,
        "1": 0,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 1,
        "13": 0,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "15": {
        "0": 0,
        "1": 1,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 1,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 1,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 1,
        "19": 0,
        "20": 0,
    },
    "16": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 1,
        "6": 0,
        "7": 1,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 0,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "17": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 1,
        "10": 1,
        "11": 1,
        "12": 0,
        "13": 1,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 1,
        "20": 0,
    },
    "18": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 1,
        "14": 0,
        "15": 1,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 1,
    },
    "19": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 1,
        "7": 0,
        "8": 1,
        "9": 0,
        "10": 0,
        "11": 1,
        "12": 0,
        "13": 1,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 1,
        "18": 0,
        "19": 0,
        "20": 0,
    },
    "20": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 1,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 1,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 1,
        "19": 0,
        "20": 0,
    },
}

# Create LP model
model = pulp.LpProblem("Districting_Problem", pulp.LpMinimize)

# Decision Variables
# The Decision Variable is 1 if the county is assigned to the district.
x = pulp.LpVariable.dicts(
    "x",
    ((i, j) for i in range(num_counties) for j in range(num_districts)),
    cat="Binary",
)

# The Decision Variable is 1 if the boundary between counties a and b is cut
x_0 = pulp.LpVariable.dicts(
    "x_0",
    ((a, b) for a in range(num_counties) for b in range(num_counties)),
    cat="Binary",
)
ic(x)
# ic(x_0)

# The Decision Variable is the deviation from mean population
pop_deviation = pulp.LpVariable("d", lowBound=0)
print(pop_deviation)
model += pulp.lpSum(
    pop_deviation
)  # Objective function: minimize deviation from population mean of all districts (total population / 12)

# Constraints
# Ensure that each county is only assigned to one district
for i in range(num_counties):
    model += pulp.lpSum([x[(i, j)] for j in range(num_districts)]) == 1

# Ensure that each district is assigned at least one county
for j in range(num_districts):
    model += pulp.lpSum([x[(i, j)] for i in range(num_counties)]) >= 1

# Add boundary assignment constraints
for a in range(num_counties):
    for b in range(num_counties):
        if (
            adjacency_matrix[str(a)][str(b)] == 1
        ):  # Check if counties a and b are adjacent
            print((str(a), str(b)))
            for j in range(num_districts):
                model += (
                    x[(a, b)] >= x[(a, j)] - x[(b, j)]
                )  # Constraint: x_0ab >= xa_j - xb_j
                model += (
                    x_0[(a, b)] >= x[(b, j)] - x[(a, j)]
                )  # Constraint: x_0ab >= xb_j - xa_j
            model += x_0[(a, b)] <= pulp.lpSum(
                x[(a, k)] for k in range(num_districts)
            )  # Constraint: x_0ab <= sum(xa_j)
            model += x_0[(a, b)] <= pulp.lpSum(
                x[(b, k)] for k in range(num_districts)
            )  # Constraint: x_0ab <= sum(xb_j)
min_dist = 65000
max_dist = 6828819
# District size constraints, in order to keep the size of districts within a specific range from the mean population
for j in range(num_districts):
    model += (
        pulp.lpSum([county_populations[i] * x[(i, j)] for i in range(num_counties)])
        + pop_deviation
        >= min_dist
    )
    model += (
        pulp.lpSum([county_populations[i] * x[(i, j)] for i in range(num_counties)])
        - pop_deviation
        <= max_dist
    )

# Model solution
model.solve()

print("Optimal districting plan:")
for j in range(num_districts):
    print(f"District {j+1}:")
    for i in range(num_counties):
        if pulp.value(x[(i, j)]) == 1:
            print(f"  {county_names[i]}")
    district_population = sum(
        county_populations[i] for i in range(num_counties) if pulp.value(x[(i, j)]) == 1
    )
    print(f"Population: {district_population}")

result_value = []
for i in range(num_counties):
    for j in range(num_districts):
        var_output = {
            "County": i,
            "District": j + 1,
            "Assignment": pulp.value(x[(i, j)]) * (j + 1),
            "Allocation": county_populations[i],
        }
        result_value.append(var_output)


df = pd.DataFrame()
df["County"] = county_names
df["CountySort"] = county_id

results = pd.DataFrame(result_value)
results = results[results["Assignment"] != 0]
results = results.sort_values(["County", "District"])
results = results.merge(
    df, left_on="County", right_on="CountySort", suffixes=("_ID", "_Name")
)
results["Multiple_County_Name"] = results["County_Name"].shift(periods=1)
results = results.sort_values(["District", "County_Name"])
results.index = results["County_ID"]

color_dict = {
    1: "lightpink",
    2: "darkorchid",
    3: "lavender",
    4: "chartreuse",
    5: "dodgerblue",
    6: "paleturquoise",
    7: "tan",
    8: "yellow",
    9: "red",
    10: "blue",
    11: "green",
    12: "slategrey",
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
            aes(x="Longitude", y="Latitude", label="Population", size=2),
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

plot1, plot2, plot3 = nj_map(map_first_pass_labels)
plot1.show()
plot2.show()
plot3.show()
