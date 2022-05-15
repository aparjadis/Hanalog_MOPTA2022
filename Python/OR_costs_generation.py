#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:10:56 2022

@author: Augustin Parjadis
"""

import numpy as np
from pandas import DataFrame

import sys
sys.path.append('../')
from Data.data_generator import DATA_GENERATOR
import random as rd
param = DATA_GENERATOR()
param.generate_data()

#costs
C_WAITING = 2 #cost of waiting for an OR, per minute
C_OVERTIME = 5 #cost of working overtime, per minute
C_IDLE = 1 #cost of waiting for a surgery, per minute

N_ITER = 100000

#surgeries
specialties = ['CAR','GAS','GYN','MED','ORT','URO']
specialties_ratio = [0.14,0.18,0.28,0.05,0.17,0.18]
specialties_slots = [[1,1,3,5,5],[1,2,3,3,4,5],[1,2,2,3,3,4,4,5],[3],[1,2,2,3,4,5],[1,1,2,3,4,5]]
specialties_mean = [99,132,78,75,142,72]
specialties_std = [53,76,52,72,58,38]
emergency_mean = 120
emergency_std = 60

specialties_count = dict()
specialties_days = dict()
specialties_means = dict()
specialties_stdev = dict()
N_SURGERIES = 100
for i,s in enumerate(specialties):
    specialties_count[s] = int(N_SURGERIES*specialties_ratio[i])
    specialties_days[s] = specialties_slots[i]
    specialties_means[s] = specialties_mean[i]
    specialties_stdev[s] = specialties_std[i]
    



class OR:
    def __init__(self, day, type_surgery, t_start):
        self.day = day
        self.type = type_surgery
        self.t_start = t_start #scheduled
        self.t_start_real = list() #random
        self.t_end = list() #random
        
    def cost(self,t):
        return C_WAITING*(t-self.t_start)
    
    def cancellation(self):
        return C_CANCELLED
    
    
def greedy_start(days):#builds optimistic start times with mean duration
    for d in range(1,6):
        for s in specialties_days:
            while d in specialties_days[s]:#create an OR of type s
                t = 0
                start_times = []
                while t < 8*60-specialties_means[s] and specialties_count[s] > 0:#populate it
                    start_times.append(t)
                    t += specialties_means[s]
                    specialties_count[s] -= 1
                days[d-1].append(OR(d,s,start_times))
                specialties_days[s].remove(d)
                
                
def day_simulation(days):#adds random end times
    for d in days:
        for OR in d:           
            if(len(OR.t_start) > 0):
                OR.t_start_real.append(OR.t_start[0])#real start is equal to the scheduled one
                for i in range(len(OR.t_start)):
                    t = OR.t_start_real[i]
                    OR.t_end.append(t+max(-t,int(np.random.normal(specialties_means[OR.type],specialties_stdev[OR.type]))))#random end
                    if i < len(OR.t_start)-1:
                        if OR.t_start[i+1] < OR.t_end[i]:#waiting occurs
                            OR.t_start_real.append(OR.t_end[i])
                        else:#real start time equal to the scheduled one
                            OR.t_start_real.append(OR.t_start[i+1])
                            
def day_simulation_times(days,dur):#computes end times with dur
    for d in days:
        for OR in d:           
            if(len(OR.t_start) > 0):
                OR.t_start_real.append(OR.t_start[0])#real start is equal to the scheduled one
                for i in range(len(OR.t_start)):
                    t = OR.t_start_real[i]
                    OR.t_end.append(t+dur[i])#given duration
                    if i < len(OR.t_start)-1:
                        if OR.t_start[i+1] < OR.t_end[i]:#waiting occurs
                            OR.t_start_real.append(OR.t_end[i])
                        else:#real start time equal to the scheduled one
                            OR.t_start_real.append(OR.t_start[i+1])
                

                
def costs(days):#takes the simulated day and computes cost
    ocost, wcost, icost = 0,0,0
    
    for d in days:
        for OR in d:
            
#            print(f'\nOR: Day {OR.day}, type {OR.type}')
#            print(f'scheduled start times {OR.t_start}')
#            print(f'start times {OR.t_start_real}')
#            print(f'end times {OR.t_end}')
            
            for i in range(len(OR.t_start)-1):
                if OR.t_end[i] > OR.t_start[i+1]:
                    wcost += (OR.t_end[i]-OR.t_start[i+1])*C_WAITING
#                    print(f'Waiting cost of {c}')
                elif OR.t_end[i] < OR.t_start[i+1]:
                    icost += (OR.t_start[i+1]-OR.t_end[i])*C_IDLE
#                    print(f'Idle cost of {c}')
                
            if len(OR.t_end) > 0:
                if OR.t_end[-1] > 60*8:
                    ocost += (OR.t_end[-1]-60*8)*C_OVERTIME
                else:
                    icost += (60*8-OR.t_end[-1])*C_IDLE
#                    print(f'Overtime cost of {(OR.t_end[-1]-60*8)*C_OVERTIME}')
                    
    return (ocost+wcost+icost,icost,wcost,ocost)


def rand_times(p,e,s):#return a list of durations of size e+s
    times = []
    for i in range(p):
        times.append( max(1,int(np.random.normal(specialties_means[s],specialties_stdev[s]))) )
    for i in range(e):
        times.append( max(1,int(np.random.normal(emergency_mean,emergency_std))) )
    return times

def rand_data_times(p,e,s):#return a list of durations of size e+s

    times = []
    for i in range(p):
        times.append( rd.sample(param.durations[param.patients['specialty'].iloc[i]], k = 1)[0])
    for i in range(e):
        emer = param.generate_emergencies()
        times.append( emer['duration'][0] )
        
    return times
  
o_file = "OR_cost.txt"
##WRITE HEADER
def estimate_best_cost(p,e,s):#patients, emergencies, specialty
    
    #if empty
    if p == 0 and e == 0:
        with open(o_file, 'a') as f_out:
            f_out.write('t '+s+' '+str(p)+' '+str(e)+' - '+ str(int(8*60*C_IDLE))+f' () {int(8*60*C_IDLE)} 0 0\n')
        return (int(8*60*C_IDLE),[])
    
    to_evaluate = []
    
    #generates random start times close to mean
    for i in range(1000):
        t = []
        for j in range(p):
            if t == []:
                t.append(0)
            else:
                t.append( t[-1] + specialties_means[s] + int(np.random.normal(0,10)))
        if e > 0:
            for j in range(e):
                if t == []:
                    t.append(0)
                else:
                    t.append( t[-1] + emergency_mean + int(np.random.normal(0,10)))
        to_evaluate.append(tuple(t))
    
    #generates random start times in [MEAN-STD,MEAN+STD]
    for i in range(1000):
        t = []
        for j in range(p):
            if t == []:
                t.append(0)
            else:
                t.append( t[-1] + min( max( int(np.random.normal(specialties_means[s],specialties_stdev[s])) , specialties_means[s]-specialties_stdev[s] ) , specialties_means[s]+specialties_stdev[s] ) )
        if e > 0:
            for j in range(e):
                if t == []:
                    t.append(0)
                else:
                    t.append( t[-1] + min( max( int(np.random.normal(emergency_mean,emergency_std)) , emergency_mean-emergency_std ) , emergency_mean+emergency_std ) )
        to_evaluate.append(tuple(t))
        
    #Do len(N) rounds, and keep N best start times based on M simulations
    N = [50, 10, 1]
    M = [N_ITER//1000, N_ITER//10, N_ITER]
    
    for it in range(len(N)):
        times = [rand_times(p,e,s) for i in range(M[it])]
        results = dict()
        for t in to_evaluate:
            res = []
            ri,ro,rw = [],[],[]
            or_test = [[[OR(1,s,t)]] for j in range(M[it])]#create M[it] simulations
            for i,d in enumerate(or_test):#tests several random realizations
                dur = times[i]
                day_simulation_times(d,dur)
                r = costs(d)#(4)
                res.append(r[0])
                ri.append(r[1])
                rw.append(r[2])
                ro.append(r[3])
            results[t] = (sum(res)/len(res),sum(ri)/len(ri),sum(rw)/len(rw),sum(ro)/len(ro))
            
            
        if len(results) > 1:
            to_evaluate = []
    #        print("\n")
            for n in range(N[it]):#select n best
#                to_evaluate.append(min(results, key=results.get))
                to_evaluate.append(min(results.items(), key = lambda t: t[1][0])[0])
    #            print(f'Best ({it}): {min(results, key=results.get)}: {results[min(results, key=results.get)]}')
#                del results[min(results, key=results.get)]
                del results[min(results.items(), key = lambda t: t[1][0])[0]]
        else:
            to_evaluate = [min(results.items(), key = lambda t: t[1][0])[0]]
    
    start_t = min(results.items(), key = lambda t: t[1][0])[0]
    cost,id_cost,wa_cost,ov_cost = min(results.items(), key = lambda t: t[1][0])[1]
    
    print(f'Best simulated {start_t}: {cost}')
    
    #should cancellations be made
#    n_cancel = 0
#    for i in range(1,p+1):
#        if len(C[(p-i,e,s)][1]) == p-i+e:
#            cost_cancel = C[(p-i,e,s)][0]+C_CANCELLED*i
#            if cost_cancel < cost:
#                print(f'removing {i} patient')
#                cost = cost_cancel
#                n_cancel = i
#                start_t = C[(p-i,e,s)][1]
#            
#    cancel_str = " "
#    if n_cancel == 1:
#        cancel_str += str(n_cancel)+" canceled surgery"
#    elif n_cancel > 1:
#        cancel_str += str(n_cancel)+" canceled surgeries"
#        
#    print(f'Cost {start_t}: {cost}')
#    
#    with open("OR_cost.txt", 'a') as f_out:
#        f_out.write('t '+s+' '+str(p)+' '+str(e)+' - '+ str(int(cost))+" "+str(start_t)+cancel_str+'\n')
        
    with open(o_file, 'a') as f_out:
        f_out.write('t '+s+' '+str(p)+' '+str(e)+' - '+ str(int(cost))+" "+str(start_t)+" "+str(int(id_cost))+" "+str(int(wa_cost))+" "+str(int(ov_cost))+'\n')
        
    return (int(cost),start_t)


if __name__ == "__main__":
    
    with open(o_file, 'w') as f_out:
        f_out.write('#costs\n')
        f_out.write(f'#C_WAITING = {C_WAITING} #cost of waiting for an OR, per minute\n')
        f_out.write(f'#C_OVERTIME = {C_OVERTIME} #cost of working overtime, per minute\n')
        f_out.write(f'#C_IDLE = {C_IDLE} #cost of waiting for a surgery, per minute\n\n')
            
    C = dict() #C[(p,e,s)]
    for sp in specialties:
        for pa in range(9):#number of elective surgeries
            for em in range(4):#number of emergencies
                C[(pa,em,sp)] = estimate_best_cost(pa,em,sp)
                
    sp_l = [[pa for em in range(4) for pa in range(9)]]
    sp_l += [[em for em in range(4) for pa in range(9)]]
    sp_l += [[C[(pa,em,sp)][0] for em in range(4) for pa in range(9)] for sp in specialties]
    
    to_write = dict()
    col = ['Electives','Emergencies\Specialities','1','2','3','4','5','6']
    for i,c in enumerate(col):
        to_write[c] = sp_l[i]
    df = DataFrame(to_write)
    df.to_excel('OR_cost.xlsx', sheet_name='sheet1', index=False)
