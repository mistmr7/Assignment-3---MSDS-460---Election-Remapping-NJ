# Assignment-3---MSDS-460---Election-Remapping-NJ
This repository contains files associated with the project management assignment for MSDS 460, which focuses on applying linear programming techniques to a network model for critical path analysis in project management. An outline of the contents of our paper is included below.

## Introduction and Data Sources
The goal of this project is to develop mathematical models to assist in the redistricting process for New Jersey. We rely on two primary sources of data: county population information sourced from the World Population Review and county adjacency data obtained from the United States Census Bureau in 2023.

## Model Specification
We present two distinct models for redistricting: a primary model based on an existing redistricting model developed for Oregon, and an alternate model integrating concepts from two political redistricting articles. The standard form for our primary model is as follows:

**Index Sets:**
- \( i \): County index (21 counties)
- \( j \): District index (12 districts)
- \( k \): Additional county index for adjacency matrix

**Data:**
- \( P_i \): Population in county \( i \)
- \( L \): Lower population limit for each district
- \( U \): Upper population limit for each district
- \( A_{ik} \): Adjacency matrix indicating whether county \( i \) borders county \( k \) (binary)
- \( M \): Large arbitrary number

**Decision Variables:**
- \( Y_{ij} \): Binary variable representing whether county \( i \) is assigned to district \( j \)
- \( X_{ij} \): Integer variable representing the population allocated from county \( i \) to district \( j \)

**Objective Function:**
\[ \text{Minimize} \sum_{i,j} Y_{ij} \]

**Subject to:**
1. **Population Allocation:**
\[ \sum_j X_{ij} = P_i \quad \forall i \]
2. **Assignment Requirement:**
\[ X_{ij} \leq M \cdot Y_{ij} \quad \forall i \]
3. **Single Assignment:**
\[ \sum_j Y_{ij} \leq 1 \quad \forall i \]
4. **Population Limits:**
\[ L \leq \sum_i X_{ij} \leq U \quad \forall j \]
5. **Adjacency Constraint:**
\[ Y_{ij} \leq \sum_k A_{ik} \cdot Y_{kj} \quad \forall i, j, k \]
6. **Non-negativity Constraint:**
\[ X_{ij}, Y_{ij} \geq 0 \quad \forall i, j \]

## Programming
We implemented our models using Python's PuLP package, leveraging various solvers including GLPK, CBC, and PuLP's built-in solver. The code files and output files are available in the repository, along with image files depicting mapped distributions of counties to districts based on model solutions.

## Solution
Our initial attempts to solve the primary model encountered obstacles with both GLPK and CBC solvers, marked by prolonged execution times and infeasible solutions. After eliminating adjacency constraints and re-executing the model using PuLP's solver, we ultimately derived an optimal solution but the districts are not geographically compact. Similarly, the alternate model produced an optimal solution but it suffered from inadequate integration of adjacency constraints. The resulting redistricting map for this model is also not geographically compact. 

## Maps and Discussion
Color-coded maps showcasing the algorithmic redistricting for both models can be accessed in the repository. Furthermore, the paper features a map displaying the current redistricting of New Jersey. Within the discussion section, an analysis is conducted on the model results, focusing on their implications regarding fairness, equity, and representation in voting. This analysis encompasses an examination of the distribution of district populations, an evaluation of the geographical compactness of the proposed districts, and an assessment of how well the redistricting plans adhere to the principle of "one person, one vote".
