# Assignment-3---MSDS-460---Election-Remapping-NJ
This repository contains files associated with the project management assignment for MSDS 460, which focuses on applying linear programming techniques to a network model for critical path analysis in project management. An outline of the contents of our paper is included below.

## Introduction and Data Sources
The goal of this project is to develop mathematical models to assist in the redistricting process for New Jersey. We rely on two primary sources of data: county population information sourced from the World Population Review and county adjacency data obtained from the United States Census Bureau in 2023.

## Model Specification
We present two distinct models for redistricting: a primary model based on an existing redistricting model developed for Oregon, and an alternate model integrating concepts from two political redistricting articles. The standard form for our primary model is as follows:

**Index Sets:**
- i: County index (21 counties)
- j: District index (12 districts)
- k: Additional county index for adjacency matrix

**Data:**
- Pi: Population in county i
- L: Lower population limit for each district
- U: Upper population limit for each district
- Aik: Adjacency matrix indicating whether county i borders county k (binary)
- M: Large arbitrary number

**Decision Variables:**
- Yij: Binary variable representing whether county i is assigned to district j
- Xij: Integer variable representing the population allocated from county i to district j

**Objective Function:**
Minimize Σ Σ Yij (minimize the total number of county-to-district assignments)

**Constraints:**
1. Σj Xij = Pi ∀ i (allocate 100% of county populations to districts)
2. Xij ≤ M ⋅ Yij ∀ i (assignment required for population allocation)
3. Σj Yij ≤ 1 ∀ i (each county must be assigned to one district)
4. L ≤ Σi Xij ≤ U ∀ j (each district must be within population limits)
5. Yij ≤ Σ Aik ⋅ Ykj ∀ i, j, k (adjacency constraint)
6. Xij, Yij ≥ 0 ∀ i, j (non-negativity constraint)

## Programming
We implemented our models using Python's PuLP package, leveraging various solvers including GLPK, CBC, and PuLP's built-in solver. The code files and output files are available in the repository, along with image files depicting mapped distributions of counties to districts based on model solutions.

## Solution
Our initial attempts to solve the primary model encountered obstacles with both GLPK and CBC solvers, marked by prolonged execution times and infeasible solutions. After eliminating adjacency constraints and re-executing the model using PuLP's solver, we ultimately derived an optimal solution but the districts are not geographically compact. Similarly, the alternate model produced an optimal solution but it suffered from inadequate integration of adjacency constraints. The resulting redistricting map for this model is also not geographically compact. 

## Maps and Discussion
Color-coded maps showcasing the algorithmic redistricting for both models can be accessed in the repository. Furthermore, the paper features a map displaying the current redistricting of New Jersey. Within the discussion section, an analysis is conducted on the model results, focusing on their implications regarding fairness, equity, and representation in voting. This analysis encompasses an examination of the distribution of district populations, an evaluation of the geographical compactness of the proposed districts, and an assessment of how well the redistricting plans adhere to the principle of "one person, one vote".
