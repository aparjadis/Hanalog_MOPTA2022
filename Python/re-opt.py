#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:48:15 2022

@author: Prakash
"""

import numpy as np
import pandas as pd
import copy
import sys
sys.path.append('../')
from Data.data_generator import DATA_GENERATOR
from minizinc import Instance, Model, Solver
import random as rd

import time

start_time = time.time()
import ast
import nest_asyncio
nest_asyncio.apply()

df = pd.read_excel('parameters.xlsx')
n_patients = int(df['n_patients'][0])

instances = {1:70, 2:100}
inst = 2
param = DATA_GENERATOR()
param.generate_data()

specialties = {'CAR': 'Card', 'GAS': 'Gastro', 'GYN': 'Gyn', 'MED': 'Med', 'ORT': 'Orth', 'URO': 'Uro'}

def update_surgery_end_time(time, a, today):
    for i in range(today.elective_complete.at[a],len(today["patients"].loc[a])):
        end = today["Start_times"].loc[a][i] +  today["durations"].loc[a][i]
        if end <= time:
            today.elective_end_times.loc[a].append(end)
            today.elective_complete.at[a] += 1
    for i in range(today.emer_complete.at[a],len(today["emer_patients"].loc[a])):
        end = today["emer_start_times"].loc[a][i] +  today["emer_durations"].loc[a][i]
        if end <= time:
            today.emer_end_times.loc[a].append(end)
            today.emer_complete.at[a] += 1
    return today

def current_surgery_exp_end( time,k):
    if today.elective_complete.at[k] != today.Electives.at[k]:
        for i in range(today.elective_complete.at[k], today.Electives.at[k] ):
            if  today.Start_times.at[k][i] <= time and today.Start_times.at[k][i] +  today["durations"].loc[k][i] > time:
                return today.Start_times.at[k][i] +  today["durations"].loc[k][i]
        for i in range(len(today.emer_patients.at[k])):
            if  today.emer_start_times.at[k][i] <= time and today.emer_start_times.at[k][i] +  today["emer_durations"].loc[k][i] > time:
                return today.emer_start_times.at[k][i] +  today["emer_durations"].loc[k][i]
        
    return 0 

def get_rem_elective_info(time,k):
    n_electives, elective_start, exp_elective_dur,elective_patients, C_canc = 0, [], [],[] , []
    for i in range(today.elective_complete.at[k], today.Electives.at[k]):
        if  today.Start_times.at[k][i] >= time:
           n_electives += 1
           elective_start.append(today.Start_times.at[k][i])
           exp_elective_dur.append(param.specialty_mean[param.specialty_code[today.Specialty.at[k]]])
           elective_patients.append(today.patients.loc[k][i])
           C_canc.append(patient_assignment.Cancel_cost.iloc[today.patients.loc[k][i]])
    return n_electives, elective_start, exp_elective_dur,elective_patients, C_canc

def get_rem_emer_info(time, k):
    n_nu_emer,nu_emer_start,nu_emer_duration = 0, [], []
    for i in range(len(today.emer_patients.at[k])):
        if  today.emer_start_times.at[k][i] >= time:
           n_nu_emer += 1
           
           nu_emer_duration.append(param.emer_exp_dur)
            
    return  n_nu_emer,nu_emer_start,nu_emer_duration

def update_elective_start_times(k, time, new_elective_start_action):
    l = 0
    for i in range(today.elective_complete.at[k] + today.status.at[k], today.Electives.at[k]):
        
        if  today.Start_times.at[k][i] >= time:
            if new_elective_start_action[l] == None:
                today.Start_times.at[k][i] = 1000
                today.elective_start_times.at[k][i] = 1000
                
            else:
                diff = new_elective_start_action[l] - today.elective_start_times.at[k][i]
                if diff > 0:
                    today.wait_time.at[k].append(diff)
                    today.Start_times.at[k][i] = new_elective_start_action[l]
                    today.elective_start_times.at[k][i] = new_elective_start_action[l]
        l += 1
        
def get_next_surgery_start(time, k):
    
    times = np.concatenate([np.array(today.elective_start_times.at[k]), np.array(today.emer_start_times.at[k])])
    surgery_list = np.concatenate([np.array(today.patients.at[k]), np.array(today.emer_patients.at[k])])
 
    mask = times > time
    
    if sum(mask) > 0:
    
        min_pos = np.argmin(times[mask])
        min_time = min(times[mask])
        surgery = int(surgery_list[mask][min_pos])
        flag = 'EL' if surgery in today.patients.at[k] else 'EM'
        if flag == 'EM':
            min_time = time
        #surgery_list = np.delete(surgery_list, np.argwhere(surgery_list==surgery))
            
        return [flag, surgery, min_time]
        
    else:
        return None
    
    
def select_block(emer_blocks, candidates):
    reduced_set = {}
    for i in emer_blocks:
        if i in candidates and emer_blocks[i] > 0:
            reduced_set[i] = emer_blocks[i]
    if len(reduced_set) > 0:
        return max(reduced_set, key=reduced_set.get)
    else:
        return rd.choices(candidates)[0]
    
def max_planned_emer_end(k):
    if len(today.emer_start_times.loc[k]) > 0:
        max_start = max(today.emer_start_times.loc[k])
        index = today.emer_start_times.loc[k].index(max_start)
        end = max_start + today.emer_durations.loc[k][index]
        return end
    else:
        return 0
        

emergencies_per_day = 3
allowedOT = 240

#cost_calculations = {}
cost_calculations = pd.DataFrame(columns = [ 'iter','C_over_time','C_idle_time','C_wait_time','C_canc', 'num_canc'])#Emer_wait

for itr in range(10):
    schedule  = pd.read_csv("results-" + str(n_patients) + "/schedule.csv")
    
    final_schedule = pd.DataFrame()
    
    for i in range(len(schedule)):
        schedule.Specialty.iat[i] = specialties[schedule.Specialty.iat[i]]
        
    schedule["Start_times"] = schedule.Start_times.apply(ast.literal_eval)
    schedule["patients"] = 0
    schedule["patients"] = schedule["patients"].astype('object')
    schedule["durations"] = 0
    schedule["status"] = 1 #free
    schedule["durations"] = schedule["durations"].astype('object')
    schedule["elective_end_times"] = [list() for x in range(len(schedule.index))]
    schedule["elective_start_times"] = [schedule.Start_times.loc[x][0:schedule.Electives.loc[x]] for x in range(len(schedule.index))]
    
    schedule["elective_complete"] = 0
    schedule["emer_patients"] = [list() for x in range(len(schedule.index))]
    schedule["emer_start_times"] = [list() for x in range(len(schedule.index))]
    schedule["emer_durations"] = [list() for x in range(len(schedule.index))]
    schedule["emer_acuity"] = [list() for x in range(len(schedule.index))]
    schedule["emer_end_times"] = [list() for x in range(len(schedule.index))]
    schedule["emer_complete"] = 0
    schedule["emer_wait"] = [list() for x in range(len(schedule.index))]
    schedule["idle_time"] = 0
    schedule["over_time"] = 0
    schedule["wait_time"] = [list() for x in range(len(schedule.index))]
    schedule["C_canc"] = [list() for x in range(len(schedule.index))]
    schedule["canc_patients"] = [list() for x in range(len(schedule.index))]
    schedule['emer_arrival'] = [list() for x in range(len(schedule.index))]
    
    
    patient_assignment = pd.read_csv("results-" + str(n_patients) +"/assignment.csv")
    patient_assignment = patient_assignment.astype({"Patient":"int","Block":"int"})
    patient_assignment['duration'] = 0
    for i in range(len(schedule)):
        temp  = patient_assignment[patient_assignment.Block == schedule.Block.iloc[i]]
        schedule["patients"].iat[i] = list(temp.Patient.values)
        times = np.array(rd.sample(param.durations[param.patients['specialty'].iloc[i]], k = len(temp.Patient.values))).astype(int)
        schedule["durations"].iat[i] = list(times)
        for j in range(len(schedule.patients.iloc[i])):
            patient_assignment['duration'].at[schedule.patients.iloc[i][j]] = times[j]
        
    
    
    days = [1,2,3,4,5]
    
    for i in days:
        print("ITR - ", itr," Running Day ", i)
        today = copy.deepcopy(schedule[schedule["Day"] == i])
        #today = today.reset_index()
        emer_blocks = {}
        emer_patients = param.generate_emergencies(count = emergencies_per_day, _id = n_patients + (i - 1) * emergencies_per_day)
        block_start_events = {}
        
        for j in range(len(today)):
            block_start_events[today['Block'].iloc[j]] = [] 
            if today['Emergencies'].iloc[j] > 0:
                emer_blocks[today['Block'].iloc[j]] = today['Emergencies'].iloc[j]
            t = 0
            for k in range(len(today['patients'].iloc[j])):
                block_start_events[today['Block'].iloc[j]].append(max(t,today['Start_times'].iloc[j][k]))
                t = t + today['durations'].iloc[j][k]
        
        #simulate
        time = 0
        #change this
        #emer_patients[0]['acuity'] = 3
        last_finish = {}
        event_order = {('A',emer_patients[k]['acuity'] , 'EM',emer_patients[k]['_id'], k ):  emer_patients[k]['arrival_time'] for k in emer_patients}
        for k in today.index:
            if len(today.patients.loc[k]) > 0:
                next_e = ('S' , today.patients.loc[k][0], 'EL', k)
                print("Started: ",next_e , "at ", 0)
                today["status"].at[k] = 0
                event_order[('F' , today.patients.loc[k][0], 'EL',k)] = today.durations.loc[k][0]
            
            last_finish[k] = 0
        event_order = {k: v for k, v in sorted(event_order.items(), key=lambda item: item[1])}
     
        
        
        while(True):
            
            
            if len(event_order) == 0:
                for k in today.index:
                    today['over_time'].at[k] = max(0, last_finish[k] - 480)
                final_schedule = pd.concat([final_schedule,today])
                break
            next_e = list(event_order)[0]
            
            emer_start = {}
            if next_e[0] == 'A':
                time = event_order[next_e]
                #today = update_surgery_end_time(time, today)
                print("Arrival: ", next_e, " at ", time)
                
                if next_e[1] == 3:
                    candidates = [k for k in today.index if today.status.loc[k] == 1]
                    emer_blocks = {k: v for k, v in sorted(emer_blocks.items(), key=lambda item: item[1])}
                    
                    
                    if len(candidates) == 0:
                        block_select = max(emer_blocks, key=emer_blocks.get)
                    else:
                        block_select = select_block(emer_blocks, candidates)
                        
                    if today.loc[block_select].status == 1:
                        emer_start[block_select]  = time
                        event_order['S' , next_e[3] , 'EM', block_select] = emer_start[block_select]
                        event_order = {k: v for k, v in sorted(event_order.items(), key=lambda item: item[1])}
                        
                    else:
                        emer_start[block_select] = max(current_surgery_exp_end(time, block_select),max_planned_emer_end(block_select))
                    
                else:
                    
                    if sum(today["status"]) == 0:
                        print("No Block Free")
                        
                        for z in list(event_order):
                            if z[0] == 'F':
                                break
                        #emer_wait_per_week = event_order[z] + 1 - event_order[next_e]
                        event_order[next_e] = event_order[z] + 1
                        
                        event_order = {k: v for k, v in sorted(event_order.items(), key=lambda item: item[1])}
                        
                        continue
                        
                    costs = {}
                    new_elective_start_action = {}
                    current_emer_start = {}
                    
                    canc_cost = {}
                    patients = {}
                    
                    #find best block to place
                    for k in today.index:
                        #end = current_surgery_exp_end(k, time)
                        if today["status"].at[k] == 0:
                            costs[k] = 1000000
                            continue
                        
                        time_rem = 480 - time #max([time, end])
                        
                        
                        n_electives, elective_start, exp_elective_dur, elective_patients, C_canc = get_rem_elective_info(time,k)
                        n_nu_emer,nu_emer_start,nu_emer_duration = get_rem_emer_info(time,k)
                        if n_electives == 0 and n_nu_emer == 0:
                            emer_start[k] = time
                            new_elective_start_action[k] = []
                            costs[k] = max(param.emer_exp_dur - time_rem ,0) * param.C_over
                        else:
                            actions = [0,1]
                        
                            model = Model("./RE_OPT.mzn")
                            solver = Solver.lookup("gecode")
                            instance = Instance(solver, model)
                            if n_nu_emer > 0:
                                print("cheeck")
                            
                            
                            instance["num_electives"] = n_electives
                            instance["num_emer"] = 1
                            instance["num_n_emer"] = n_nu_emer

                            
                            instance["ACTION"] = [0,1]
                
                            instance["duration"] = [exp_elective_dur[i] if j == 0 else 0 for i in range(n_electives) for j in actions]
                            instance["start_elective_current"] = [int(elective_start[i] - time) if j == 0 else 0 for i in range(n_electives) for j in actions]
                            
                            instance["emer_duration"] = [param.emer_exp_dur]
                            instance["n_emer_duration"] = nu_emer_duration
                            
                            instance["C_over"] = 5
                            instance["C_wait"] = 2
                            instance["C_idle"] = 1
                            instance["C_canc"] = C_canc
                
                            instance["allowed_OT"] = allowedOT
                            instance["cap_rem"] = int(time_rem)
                            
                            instance["acuity"] = next_e[1]
                            
                            result = instance.solve()
                
                            new_elective_start_action[k] = [ time + result['new_e_start_action'][l][0] if result['new_e_start_action'][l][0] != None else result['new_e_start_action'][l][1] for l in range(n_electives) ]
                            canc_cost[k] = C_canc
                            current_emer_start[k] =  [time + l for l in result['n_emer_start']]
                            patients[k] = elective_patients
                            emer_start[k] = time + result['emer_start'][0]
                            costs[k] = result['obj']
                            
                    min_cost = min(costs.values())
                    candidates = [k for k in costs if abs(min_cost - costs[k]) < 5]
                    
                    block_select = select_block(emer_blocks, candidates)
                    today.C_canc.loc[block_select].extend([canc_cost[block_select][l] for l in new_elective_start_action[block_select] if l == None])
                    today.canc_patients.loc[block_select].extend([patients[block_select][l] for l in new_elective_start_action[block_select] if l == None])
                    
                    if len(new_elective_start_action[block_select]) > 0 :
                        update_elective_start_times(block_select,time, new_elective_start_action[block_select])
                    
                today.emer_patients.loc[block_select].append(next_e[3] )
                today.emer_durations.loc[block_select].append(emer_patients[next_e[4]]['duration'])
                today.emer_acuity.loc[block_select].append(emer_patients[next_e[4]]['acuity'])
                today.emer_start_times[block_select].append(emer_start[block_select])
                block_start_events[block_select].append( emer_start[block_select])
                if block_select in emer_blocks:
                    emer_blocks[block_select] = emer_blocks[block_select] - 1 
                event_order.pop(next_e)
                event_order['S' , next_e[3] , 'EM', block_select] = emer_start[block_select]
                event_order = {k: v for k, v in sorted(event_order.items(), key=lambda item: item[1])}
                
                patient_assignment = patient_assignment.append({'Patient':int(next_e[3]), 'Block': block_select, 'Cancel_cost':100000, 'duration' : emer_patients[next_e[4]]['duration']}, ignore_index=True)
                today.emer_arrival.loc[block_select].append(emer_patients[next_e[4]]['arrival_time'])
                
            elif  next_e[0] == 'F':
                time = event_order[next_e]
                today = update_surgery_end_time(time, next_e[3], today)
                next_surgery = get_next_surgery_start(time, next_e[3])
                if next_surgery != None:
                    event_order['S' , next_surgery[1], next_surgery[0], next_e[3]] = next_surgery[2]
                today["status"].at[next_e[3]] = 1
                event_order.pop(next_e)
                event_order = {k: v for k, v in sorted(event_order.items(), key=lambda item: item[1])}
                print("Finished: ",next_e, " at ", time)
                last_finish[next_e[3]] = time
            elif next_e[0] == 'S':
                today["status"].at[next_e[3]] = 0
                time = event_order[next_e]
                if time - last_finish[next_e[3]] < 0:
                    print("here")
                today["idle_time"].at[next_e[3]] += time - last_finish[next_e[3]]
                event_order['F' , next_e[1], next_e[2], next_e[3]] = time + patient_assignment.duration.loc[next_e[1]]
                
                event_order.pop(next_e)
                event_order = {k: v for k, v in sorted(event_order.items(), key=lambda item: item[1])}
                print("Started: ",next_e, " at ", time)
                
    final_schedule['Cum_C_wait_time'] = 0
    final_schedule['Cum_C_canc'] = 0
    final_schedule['Cum_Emer_wait'] = 0
    final_schedule['num_canc'] = 0
    for z in range(len(final_schedule)):
        final_schedule['Cum_C_wait_time'].iat[z] = sum(final_schedule['wait_time'].iloc[z])
        final_schedule['Cum_C_canc'].iat[z] = sum(final_schedule['C_canc'].iloc[z])
        #final_schedule['Cum_Emer_wait'].iat[z] = sum(final_schedule['emer_wait'].iloc[z])
        final_schedule['num_canc'].iat[z] = len(final_schedule['canc_patients'].iloc[z])
    
    cost_calculations.loc[itr] = [itr,sum(final_schedule['over_time']) * param.C_over, sum(final_schedule['idle_time']) * param.C_idle,
                                 sum(final_schedule['Cum_C_wait_time']) * param.C_wait,sum(final_schedule['Cum_C_canc']), 
                                 np.mean(final_schedule['num_canc'])]#,sum(final_schedule['Cum_Emer_wait'])]
    print("Week Completed")
    # ({'iter':itr,
    #                           'C_over_time': sum(final_schedule['over_time']),
    #                           'C_idle_time': sum(final_schedule['idle_time']),
    #                           'C_wait_time': sum(final_schedule['Cum_C_wait_time']),
    #                           'C_canc': sum(final_schedule['Cum_C_canc']), 
    #                           'num_canc':np.mean(final_schedule['num_canc']),
    #                           'Emer_wait': sum(final_schedule['Cum_Emer_wait'])}, ignore_index = True)
    
                
cost_calculations.to_csv("Cost_output.csv")
