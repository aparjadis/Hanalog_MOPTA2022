#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This file contains the code to produce a schedule for the week, with the parameters and data 
from the files parameters.xlsx, OR_cost.txt, Patient_data{n_patients}.csv.
'''

import pandas as pd
import re
from minizinc import Instance, Model, Solver
import nest_asyncio
nest_asyncio.apply()


spec = {'Card': 0, 'Gastro': 1, 'Gyn': 2, 'Med': 3, 'Orth': 4, 'Uro': 5}
n_day = 5

df = pd.read_excel('parameters.xlsx')
n_patients = int(df['n_patients'][0])
n_daily_emergencies = int(df['n_daily_emergencies'][0])
n_pat,n_em = n_patients,n_daily_emergencies
day = [int(df['day'][i]) for i in range(32)]#day of a block
spec_b = [int(df['spec_b'][i]) for i in range(32)]#speciality of a block

df = pd.read_csv(f'Patient_data{n_pat}.csv')
spec_i = [spec[df['specialty'][i]]+1 for i in range(n_patients)]#speciality of a patient
performing_cost = [float(df['perform_cost'][i]) for i in range(n_patients)]
postponing_cost = [float(df['postpone_cost'][i]) for i in range(n_patients)]
cancel_cost = [float(df['postpone_cost'][i]) for i in range(n_patients)]


#Load OR_cost. or_cost[s][l][m] is the cost of l electives and m emergencies for speciality s
spec = {'CAR': 0, 'GAS': 1, 'GYN': 2, 'MED': 3, 'ORT': 4, 'URO': 5}
spec_str = {0:'CAR', 1:'GAS',2:'GYN',3:'MED',4:'ORT',5:'URO'}
or_cost = [[[0 for m in range(4)] for l in range(9)] for s in range(6)]
or_cost_t = [[["" for m in range(4)] for l in range(9)] for s in range(6)]
f = open("OR_cost.txt", "r")
for line in f:
    l = line.split()
    if l != []:
        if l[0] == 't':
            el,em = int(l[2]), int(l[3])
            or_cost[spec[l[1]]][el][em] = int(l[5])
            t,c = "",0
            if el+em > 0:
                temp = l[6]
                while temp[-1] != ")":
                    t += temp
                    c += 1
                    temp = l[6+c]
                t += temp
            or_cost_t[spec[l[1]]][el][em] = t

model = Model("./mip_solve_week.mzn")
solver = Solver.lookup("coin-bc")
instance = Instance(solver, model)

instance["I"] = n_patients
instance["M"] = n_daily_emergencies
instance["day"] = day
instance["spec_b"] = spec_b
instance["spec_i"] = spec_i
instance["cost"] = or_cost
instance["performing_cost"] = performing_cost
instance["postponing_cost"] = postponing_cost

result = instance.solve()
#print('Results ',result)

z = result['z']
x = result['x']
y = result['y']
w = result['w']

specialties = ['CAR','GAS','GYN','MED','ORT','URO']
specialties_mean = [99,132,78,75,142,72]
specialties_std = [53,76,52,72,58,38]
emergency_mean = 120
emergency_std = 60
cost_emer = 500
C_WAITING = 2 #cost of waiting for an OR, per minute
C_OVERTIME = 5 #cost of working overtime, per minute
C_IDLE = 1 #cost of waiting for a surgery, per minute

#idling, waiting, overtime, performing, postponing 
C = [0 for i in range(5)]

def print_cost(C):
    L = ['idling', 'waiting', 'overtime', 'performing', 'postponing']
    for i in range(len(C)):
        print(f'{L[i]} costs = {C[i]}')

def costs_noEmergency(s,t,l):
    
    id_c, wa_c, ov_c = 0,0,0
    T = [t[i] for i in range(l)]
    dur = [specialties_mean[s] for i in range(l)]
    for i in range(len(T)-1):
        
        if T[i]+dur[i] > T[i+1]:
            wa_c += (T[i]+dur[i]-T[i+1])*C_WAITING
        elif T[i]+dur[i] < T[i+1]:
            id_c += (T[i+1]-(T[i]+dur[i]))*C_IDLE
        T[i+1] = max(T[i+1],T[i]+dur[i])
        
    if l > 0:
        if T[-1]+specialties_mean[s] > 60*8:
            ov_c += (T[-1]+dur[-1]-60*8)*C_OVERTIME
        else:
            id_c += (60*8-(T[-1]+dur[-1]))*C_IDLE
    else:
        return (60*8)*C_IDLE,wa_c,ov_c
                
    return id_c,wa_c,ov_c


def costs_Emergency(s,t,l,m,added_m):#m spots reserved, added_m spots actually used
    
    if added_m == 0:
        return costs_noEmergency(s,t,l)
    
    id_c, wa_c, ov_c = 0,0,0
    
    if added_m <= m:
        T = t[:l+added_m]
        dur = [specialties_mean[s] for i in range(l)]+[emergency_mean for i in range(added_m)]
    else:
        T = t.copy()
        dur = [specialties_mean[s] for i in range(l)]+[emergency_mean for i in range(m)]
        for i in range(added_m-m):
            dur.append(emergency_mean)
            T.append(T[-1]+emergency_mean)
            
    for i in range(len(T)-1):
        
        if T[i]+dur[i] > T[i+1]:
            wa_c += (T[i]+dur[i]-T[i+1])*C_WAITING
        elif T[i]+dur[i] < T[i+1]:
            id_c += (T[i+1]-(T[i]+dur[i]))*C_IDLE
        T[i+1] = max(T[i+1],T[i]+dur[i])
        

    if T[-1]+dur[-1] > 60*8:
        ov_c += (T[-1]+dur[-1]-60*8)*C_OVERTIME
    else:
        id_c += (60*8-(T[-1]+dur[-1]))*C_IDLE
                
    return id_c,wa_c,ov_c

def find_em_block(em,n_daily_emergencies,B):
    #forall k<=mm_day, add to reserved spot
    #forall k>mm_day, add at the end of block and find cost ; keep best cost, add em to block for next k
    #return tuple of blocks
    em_blocks = [[] for n in range(n_day)]
    
    for i in range(len(B)):
        B[i][3] = 0
    
    if n_daily_emergencies == em:
        for i,b in enumerate(B):
            for j in range(b[1]):
                B[i][3] += 1
#                print(f'em to block {i}')
                em_blocks[day[i]-1].append(i)
                
    elif n_daily_emergencies > em:
        candidates = [[] for n in range(n_day)]
        for i,b in enumerate(B):
            for j in range(b[1]):
                candidates[day[i]-1].append(i)
        for n in range(n_day):
            assert(len(candidates[n]) == n_daily_emergencies)
            while(len(em_blocks[n]) < em):
                res = []
                for c in candidates[n]:
                    res.append( (c,sum(costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]+1))-sum(costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]))) )
                min_c, c = 10000,0
                for r in res:
                    if r[1] < min_c:
                        min_c = r[1]
                        c = r[0]
                em_blocks[n].append(c)
                candidates[n].remove(c)
                B[c][3] += 1
#                print(f'em to block {c}')
                
    elif n_daily_emergencies < em:
        for i,b in enumerate(B):
            for j in range(b[1]):
                B[i][3] += 1
#                print(f'em to block {i}')
                em_blocks[day[i]-1].append(i)
        candidates = [[i for i in range(32)]*(em-n_daily_emergencies) for n in range(n_day)]
        for n in range(n_day):
            res = []
            while(len(em_blocks[n]) < em):
                for c in candidates[n]:
                    if day[c]-1 == n:
#                        print(f'candidate {c},{B[c]}')
                        res.append( (c,sum(costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]+1))-sum(costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]))) )
#                        print(f'cost {res[-1]}')
                min_c, c = 10000,0
                for r in res:
#                    print(r[0],r[1],costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]+1),costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]))
                    if r[1] < min_c:
                        min_c = r[1]
                        c = r[0]
                em_blocks[n].append(c)
                candidates[n].remove(c)
                B[c][3] += 1
#                print(f'em to block {c}')
                
    return B,em_blocks

Blocks = []
schedule = pd.DataFrame(columns = ['Block','Specialty','Day','Electives','Emergencies','Start_times'])
loc = 0
c = 0
#with open(f"schedule_phase1_{n_pat}.csv", 'w') as f_out:
#    f_out.write(f'Block,Specialty,Day,Electives,Emergencies,,Start_times\n')
for b in range(32):
    for l in range(9):
        for m in range(4):
            if z[b][l][m]:
#                    f_out.write(f'{b},{spec_str[spec_b[b]-1]},{day[b]},{l},{m},,{or_cost_t[spec_b[b]-1][l][m][1:-1]}\n')
                schedule.loc[loc] = [b,spec_str[spec_b[b]-1],day[b],l,m,list(eval(or_cost_t[spec_b[b]-1][l][m]))]
                print(f'Block {b} with {l} electives, {m} emergencies (cost {or_cost[spec_b[b]-1][l][m]} ({spec_str[spec_b[b]-1]}), start times {or_cost_t[spec_b[b]-1][l][m]} day {day[b]})')
                c += or_cost[spec_b[b]-1][l][m]
                loc += 1
                times = [int(s) for s in re.findall(r'\d+', or_cost_t[spec_b[b]-1][l][m])]
                temp_c = costs_noEmergency(spec_b[b]-1,times,l)
#                    print(temp_c)
                for i,j in enumerate(temp_c):
                    C[i] += j
                Blocks.append([l,m,times,0])
                
#    f_out.write(f'\nPatient,Block,Cancel_cost\n') 
loc = 0
assign = pd.DataFrame(columns = ['Patient','Block','Cancel_cost'])
for i in range(n_patients):
    if w[i]:
#            f_out.write(f'{i},x\n')   
        assign.loc[loc] = [i,-1, 0]
        print(f'Patient {i} postponed (cost {postponing_cost[i]})')
        c += postponing_cost[i]
        C[4] += int(postponing_cost[i])
    else:
        block = -1
        for b in range(32):
            if x[b][i]:
                block = b
        assign.loc[loc] = [int(i), block, cancel_cost[i]]
#            f_out.write(f'{i},{block},{cancel_cost[i]}\n')  
        c += performing_cost[i]
        C[3] += int(performing_cost[i])
    loc +=1
     
schedule.to_csv(f'results-{n_patients}/schedule.csv')
assign.to_csv(f'results-{n_patients}/assignment.csv')
#print(f'\n cost = {sum(C)}')
#print_cost(C)
            
print(f'\n\nSchedule with {n_patients} electives, {n_daily_emergencies} emergency spots per day')

#for em in range(4):
#    C_em = [0 for i in range(5)]
#    C_em[3],C_em[4] = C[3],C[4]
#    for i in range(em):
#        C_em[3] += 5*cost_emer
#    b,em_blocks = find_em_block(em,n_daily_emergencies,Blocks.copy())
##    print("-- ",em,em_blocks)
##    print(b)
#    
#    for i in range(32):
#        l,m = b[i][0],b[i][1]
#        times = b[i][2]
#        temp_c = costs_Emergency(spec_b[i]-1,times,l,m,b[i][3])
##        print(temp_c)
#        for i,j in enumerate(temp_c):
#            C_em[i] += j
#    print(f'\n{em} emergencies per day - cost = {sum(C_em)}')
#    print_cost(C_em)