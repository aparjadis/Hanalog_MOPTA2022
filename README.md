# Hanalog - MOPTA_2022

## ESP solver

This solver works in multiple steps. Phase 1 allows to generate a schedule based on simulated OR costs, patients data that allows to account for different surgery priorities, and an expected daily emergencies arrival rate. Phase 2 allows to reoptimize that schedule dynamically when an emergency arrives ; different priorities of emergencies are considered. Simulated OR costs and patients data based on literature are provided but can be easily modified or recomputed depending on the real situation and the user preferences.

### Phase 1

#### AIMMS

The AIMMS program provided can be used by opening and running the project. The data is loaded automatically from the different xlsx data files.

#### Python 

Using a conda environment makes the packages installation easier.

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

A schedule_phase1_n.csv file containing the schedule is produced, with n = {70,100,140,200} the number of patients defined in parameters.xlsx. Analysis of the quality of those results are provided in the report.

### Phase 2 ---TODO

Based on schedule_phase1.csv, emergencies can dynamically be added to the schedule.
--User shoud be able to enter the emergency urgency and time of arrival, and a new schedule is produced for this.

```shell
--CODE HERE--
python solve_emergencies.py #user interaction with the code?
``` 

A schedule_phase2.csv file containing the new updated schedule is produced.


## Data

Data for simulated OR costs and patients is provided as described in the report, and those can be modified.

### OR costs

Costs can be recomputed if the costs considered change. The following code takes around 1 hour to complete.

```shell
python OR_costs_generation.py
``` 

A OR_cost.txt file is produced and will then be used by the solver in phase 1. OR_cost.xlsx is used for readability and for AIMMS.


### Patients Data ---TODO

Launching data_generator.py? Modifying Patient_dataXXX.csv?


### Phase 1 data analysis

The results from phase 1 analysed in the report are replicable by running

```shell
python emergencies_cost_estimation.py
``` 

## Files/Directories
Survey.txt - contains Literature review
Report.pdf - submitted report ---TO ADD
data/ - contains ...

