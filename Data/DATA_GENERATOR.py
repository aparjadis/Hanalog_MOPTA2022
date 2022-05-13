#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:54:38 2022

@author: gawpra
"""

import pandas as pd
import random as rd
import numpy as np
np.random.seed(seed=10)
rd.seed(10)

class DATA_GENERATOR():
    
    def __init__(self, instance = 1, emergencies = 1):
        
        """Initialise:
            reference_list: All articles with name and _id
            cost_file: All cost settings from different articles, select one index from here to set the cost
            patient_count: Instance patient counts from MOPTA 2022 Doc
            instance: The four instances asked with patient count
            
            urgency_status: least value most urgent
            urgency_treatment_threshold: max time for treating a patient for resp urgency
            urgency_percentage: rsp percentage of the population with a given urgency
            avg_wt_fraction: mean waiting time as a fraction of threshold treatment time 
            
            emergency_status: least value most urgent
            emergency_percentage: percentage for a given acuity
            
            doh: hours in day
            dow: dasy in week
            
            
            This function additionally generates the set of patients
            """
        self.reference_list = pd.read_excel("../Data/Reference.xlsx")
        self.running_cost = pd.read_excel("../Data/Running_Cost.xlsx")
        
        self.patient_count = {1:70, 2: 100, 3:140, 4:200}
        self.emer_patient_count = emergencies
        self.dow = 5
        self.doh = 8
        self.instance = instance
        
        #elective surgery
        self.urgency_status = [1,2,3,4]
        self.urgency_treatment_threshold = {1: 8, 2: 30, 3: 90, 4: 270}
        self.urgency_percentage = {1: 0.08, 2: 0.4, 3: 0.4, 4:0.12}
        self.avg_wt_fraction = 0.5
        
        #emergency surgery

        self.emergency_status = [1,2,3]
        self.emergency_response = {1:30, 2: 40, 3: 480}
        self.emergency_percentage = {1: 0.1, 2: 0.4, 3: 0.5}

        self.specialty_code = [1,2,3,4,5,6]
        self.specialty_type = {1:'Card', 2: 'Gastro', 3:'Gyn',  4:'Med', 5: 'Orth', 6: 'Uro'}
        self.specialty_mean = {1:99, 2: 132, 3:78,  4:75, 5: 142, 6: 72}
        self.specialty_std = {1:53, 2: 76, 3:52,  4:72, 5: 58, 6: 38}
        
        self.specialty_distribution = {1:0.14, 2:0.18, 3:0.28, 4:0.05 ,5:0.17, 6:0.18}
        
    def read_surgery_data(self):
        self.surgery_times = pd.read_excel("../Data/mssadjustssurgerydata.xls")
        self.surgery_times = self.surgery_times[self.surgery_times['Actual Surgery TIME'] > 0 ]
        self.emer_surgery_times = self.surgery_times[self.surgery_times['Emergency'] == 'Yes']
        self.elective_surgery_times = self.surgery_times[self.surgery_times['Emergency'] == 'No']
        self.durations = {}
        for i in self.specialty_type.values():
            self.durations[i] = list(self.elective_surgery_times[self.elective_surgery_times['Surgery Team'] == i]['Actual Surgery TIME'])
        
    def generate_data(self, index =  1):
        self.set_fixed_cost(index =  1)
        self.read_surgery_data()
        self.generate_patients()
        self.generate_emergencies()

        
    def set_fixed_cost(self, index =  1 ):
        
        """
        Parameter setter function
        index: index of the queried setting
        sub_id: sub setting for that code
        C_over: overtime cost
        C_idle: idle time cost
        C_wait: wait time cost
        C_wait_emer : emergency cost per time for delayed treatment
        unit: type of cost parameters in the srticles
        info: additional information
        
        perf_vs_postp :cost ratio for perform against postpone
        perf_vs_canc : cost ratio for perform against cancel, here cancel is additional to initial perform cost
        """
        self.index = index
        self.id = self.running_cost.loc[index, 'sub_id']
        self.sub_id = self.running_cost.loc[index, 'sub_id']
        self.C_over = self.running_cost.loc[index, 'C_over']
        self.C_idle = self.running_cost.loc[index, 'C_idle']
        self.C_wait = self.running_cost.loc[index, 'C_wait']
        self.C_wait_emer = self.running_cost.loc[index, 'C_wait_emer']
        
        self.cost_type = self.running_cost.loc[index, 'type']
        if self.running_cost.loc[index, 'unit'] == "-":
            self.unit = None
        else:
            self.unit = self.running_cost.loc[index, 'unit']
        self.info = self.running_cost.loc[index, 'info']
        
        self.base_vs_postp = 10
        self.perf_vs_postp = 1.5*20
        self.perf_vs_canc = 1.5*20
        
        
    
        
    def generate_patients(self):
        """
        Generates patient set
        patients: dataframe of patients with current waiting times, priorities and therie respective perform and postpone cost.
            columns - 
            _id - patient id
            waiting_time - time in days
            priority - from 3 category
            specialty - type of patient
            perform_cost - cost of perform a surgery
            postpone_cost - cost of postponing a surgery
            cancel_cost - cost of cancel surgery
            #duration - final duration for that patient
        
        note: The seed has been set
        """
        
        mean_wt = {j :  self.urgency_treatment_threshold[j] * self.avg_wt_fraction for j in self.urgency_status}
        max_status = max(self.urgency_status) + 1
        
        self.patients = pd.DataFrame(columns = ['_id', 'waiting_time', 'priority', 'specialty', 'perform_cost', 'postpone_cost', 'cancel_cost'])
        self.patients['_id'] = range(0, self.patient_count[self.instance])
        self.patients['priority'] = rd.choices(self.urgency_status, self.urgency_percentage.values(), k = self.patient_count[self.instance])
        specialty = rd.choices(self.specialty_code ,self.specialty_distribution.values() ,k = self.patient_count[self.instance])
        self.patients['specialty'] = [self.specialty_type[j] for j in specialty]
        
        wt = []
        perf_cost = []
        #dur = []
        
        for i in range(self.patient_count[self.instance]):
            wt.append(min(np.random.exponential(mean_wt[self.patients['priority'].iloc[i]]),self.urgency_treatment_threshold[self.patients['priority'].iloc[i]]))   
            perf_cost.append((max_status - self.patients['priority'].iloc[i]) * (1 + wt[-1]/self.urgency_treatment_threshold[self.patients['priority'].iloc[i]]) * self.C_over )   
            #dur.append(rd.sample(self.durations[self.patients['specialty'].iloc[i]], k = 1)[0])
        self.patients['waiting_time'] = wt
        perf_cost = [ round(elem, 2) for elem in perf_cost ]
#        self.patients['perform_cost'] = perf_cost
        self.patients['perform_cost'] = [self.base_vs_postp * i for i in  perf_cost]
        self.patients['postpone_cost'] = [self.perf_vs_postp * i for i in  perf_cost]
        self.patients['cancel_cost'] = [self.perf_vs_canc * i for i in  perf_cost ]
        #self.patients['duration'] = dur
        self.patients.to_csv(f"Patient_data{self.patient_count[self.instance]}.csv")
        
    def generate_emergencies(self, _id = 1):
        """
        Generate given number of emergencies for a day alongwith arrival times
        emergency_patients: dataframe of patients with category and arrival times
        """
        emer_id = _id + 1
        acuity = rd.choices(self.emergency_status, self.emergency_percentage.values(), k = 1)
        arrival_time  = self.doh  * np.random.uniform(size = 1)
        duration = rd.sample(list(self.emer_surgery_times['Actual Surgery TIME']), k = 1)

        return {'_id':emer_id, 'acuity':acuity, 'arrival_time':arrival_time, "duration":duration}


for i in range(1,5):
    param = DATA_GENERATOR(instance = i)
    param.generate_data()
