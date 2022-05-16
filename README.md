# Hanalog - MOPTA_2022

## ESP solver

This solver works in multiple steps. Phase 1 allows to generate a schedule based on simulated OR costs, patients data that allows to account for different surgery priorities, and an expected daily emergencies arrival rate. Phase 2 allows to reoptimize that schedule dynamically when an emergency arrives ; different priorities of emergencies are considered. Simulated OR costs and patients data based on literature are provided but can be easily modified or recomputed depending on the real situation and the user preferences.

### Phase 1

#### AIMMS

The AIMMS program provided can be used by opening and running the project. The data is loaded automatically from the different xlsx data files.

#### Python 

This also be done using python, with which the rest of the tests have been implemented. Using a conda environment makes the packages installation easier.

```shell
cd Python
conda create --name hanalog_mopta --file requirements.txt
``` 

A schedule is created and automatically saved by running solve_week.py.


```shell
conda activate hanalog_mopta
cp OR_cost_reference.txt OR_cost.txt
python solve_week.py
``` 

Files schedule.csv and assignment.csv containing the schedule are produced, with the number of patients n = {70,100,140,200} defined in parameters.xlsx. Analysis of the quality of those results are provided in the report.

### Phase 2

Based on schedule.csv and assignment.csv, emergencies can dynamically be added to the schedule. This part is partially completed (all the logic is coded but the code crashes).

```shell
python re-opt.py
``` 


## Data

Data for simulated OR costs and patients is provided as described in the report, and those can be modified.

### OR costs

Costs can be recomputed with changed idle, wait and overtime costs. The following command takes around 1 hour to complete.

```shell
python OR_costs_generation.py
``` 

A OR_cost.txt file is produced and will then be used by the solver in phase 1. OR_cost.xlsx is used for readability and for the AIMMS program.


### Patients Data

data_generator.py produces the data used by the solver based on historical/public data. Modifying Patient_data{70..200}.csv allow to solve for the desired situation, with patients having different costs and priorities.


### Phase 1 data analysis

The results from phase 1 analysed in the report are replicable by running

```shell
python emergencies_cost_estimation.py
``` 

## Files/Directories

data/ - contains the code and references to produce the patients data along with a literature review (Survey.txt).

AIMMS/ - contains the AIMMS program that produces a schedule.

Python/ - contains the code described above for computing OR costs, generating a week schedule and analyse the results.
