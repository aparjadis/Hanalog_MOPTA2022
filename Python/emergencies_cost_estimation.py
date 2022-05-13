#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from minizinc import Instance, Model, Solver
import nest_asyncio
nest_asyncio.apply()


for n_pat in [70,100,140,200]:
#for n_pat in [140]:
    print(f'\n---{n_pat} patients----------------------------------------\n')
    for n_emer in range(4):
        
        spec = {'Card': 0, 'Gastro': 1, 'Gyn': 2, 'Med': 3, 'Orth': 4, 'Uro': 5}
        n_day = 5
    
        df = pd.read_excel('parameters.xlsx')
    #    n_patients = int(df['n_patients'][0])
        n_patients = n_pat
        #n_daily_emergencies = int(df['n_daily_emergencies'][0])
        n_daily_emergencies = n_emer
        day = [int(df['day'][i]) for i in range(32)]#day of a block
        spec_b = [int(df['spec_b'][i]) for i in range(32)]#speciality of a block
        
        df = pd.read_csv(f'Patient_data{n_pat}.csv')
        spec_i = [spec[df['specialty'][i]]+1 for i in range(n_patients)]#speciality of a patient
        performing_cost = [float(df['perform_cost'][i]) for i in range(n_patients)]
        postponing_cost = [float(df['postpone_cost'][i]) for i in range(n_patients)]
        cancel_cost = [float(df['postpone_cost'][i])-float(df['perform_cost'][i]) for i in range(n_patients)]
        
        
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
        cancel_cost_g = 800
        C_WAITING = 2 #cost of waiting for an OR, per minute
        C_OVERTIME = 5 #cost of working overtime, per minute
        C_IDLE = 1 #cost of waiting for a surgery, per minute
        
        #idling, waiting, overtime, performing, postponing 
        C = [0 for i in range(5)]
        
        def print_cost(C):
            L = ['idling', 'waiting', 'overtime', 'performing', 'postponing','cancellation']
            for i in range(len(C)):
                print(f'{L[i]} costs = {C[i]}')
            print('\n---------------------------------------------')
        
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
        
        
        def costs_Emergency(s,t,l,m,added_m,b_cancel_c):#m spots reserved, added_m spots actually used
            
            cancel = False
            
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
                    if l > 1:
                        T.append(T[-1]+emergency_mean)
                    else:
                        T.append(emergency_mean*i)
                cancel = True
                    
            for i in range(len(T)-1):
                
                if T[i]+dur[i] > T[i+1]:
                    wa_c += (T[i]+dur[i]-T[i+1])*C_WAITING
                elif T[i]+dur[i] < T[i+1]:
                    id_c += (T[i+1]-(T[i]+dur[i]))*C_IDLE
                T[i+1] = max(T[i+1],T[i]+dur[i])
                
        
            if T[-1]+dur[-1] > 60*8:
                ov_c += (T[-1]+dur[-1]-60*8)*C_OVERTIME
#                if added_m > m:
#                    ov_c += int((T[-1]+dur[-1]-60*8)*1.5*C_OVERTIME)
#                else:
#                    ov_c += (T[-1]+dur[-1]-60*8)*C_OVERTIME
            else:
                id_c += (60*8-(T[-1]+dur[-1]))*C_IDLE
            
            if cancel and l > 0:
                fc = costs_Emergency_cancel(s,t,l,m,added_m)
                cancel_c = sum(fc)+b_cancel_c
#                print(f'cancel {sum(fc)} {id_c+wa_c+ov_c} must have cost < {id_c+wa_c+ov_c-sum(fc)}')
                if cancel_c<id_c+wa_c+ov_c:
                    return fc[0],fc[1],fc[2],cancel_c<id_c+wa_c+ov_c
                
                
            return id_c,wa_c,ov_c,False
        
        
        def costs_Emergency_cancel(s,t,l,m,added_m):#m spots reserved, added_m spots actually used
            

            
            id_c, wa_c, ov_c = 0,0,0

            T = t.copy()
            T = T[:l-1]+[temp-specialties_mean[s] for temp in T[l:]]
            
            dur = [specialties_mean[s] for i in range(l-1)]+[emergency_mean for i in range(m)]
            for i in range(added_m-m):
                dur.append(emergency_mean)
                if l > 1:
                    T.append(T[-1]+emergency_mean)
                else:
                    T.append(emergency_mean*i)
#            print(f'l {l}, m {m} added_m {added_m}, t {t} -> {T}')
            
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
        
        
        def find_em_block(em,n_daily_emergencies,B,b_cancel_cost):
            #forall k<=mm_day, add to reserved spot
            #forall k>mm_day, add at the end of block and find cost ; keep best cost, add em to block for next k
            #return tuple of blocks
            em_blocks = [[] for n in range(n_day)]
            cancelled = []
            
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
                            temp_sum = sum(costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]+1,b_cancel_cost[c])[:3])-sum(costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3],b_cancel_cost[c])[:3])
                            temp_cond = costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]+1,b_cancel_cost[c])[-1]
#                            print(f'{temp_sum} {temp_cond}')
                            res.append( (c,temp_sum,temp_cond) )
                        min_c, c, cancel_cond = 10000,0,False
                        for r in res:
                            if r[1] < min_c:
                                min_c = r[1]
                                c = r[0]
                                cancel_cond = r[2]
#                                print(cancel_cond)
                        em_blocks[n].append(c)
                        candidates[n].remove(c)
                        if cancel_cond:
#                            print(f'patient cancelled!! (cancel cost {b_cancel_cost[c]})')
                            cancelled.append(c)
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
                                temp_sum = sum(costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]+1,b_cancel_cost[c])[:3])-sum(costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3],b_cancel_cost[c])[:3])
                                temp_cond = costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]+1,b_cancel_cost[c])[-1]
#                                print(f'{temp_sum} {temp_cond}')
                                res.append( (c,temp_sum,temp_cond) )
        #                        print(f'candidate {c},{B[c]}')
#                                res.append( (c,sum(costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]+1))-sum(costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]))) )
        #                        print(f'cost {res[-1]}')
                        min_c, c, cancel_cond = 10000,0,False
                        for r in res:
        #                    print(r[0],r[1],costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]+1),costs_Emergency(spec_b[c]-1,B[c][2],B[c][0],B[c][1],B[c][3]))
                            if r[1] < min_c:
                                min_c = r[1]
                                c = r[0]
                                cancel_cond = r[2]
#                                print(cancel_cond)
                        em_blocks[n].append(c)
                        candidates[n].remove(c)
                        if cancel_cond:
#                            print(f'patient cancelled!! (cancel cost {b_cancel_cost[c]})')
                            cancelled.append(c)
                        B[c][3] += 1
        #                print(f'em to block {c}')
                        
            return B,em_blocks,cancelled
        
        Blocks = []
        b_cancel_cost = [10000 for i in range(32)]
        b_perform_cost = [10000 for i in range(32)]
#        with open("schedule_phase1.csv", 'w') as f_out:
#            f_out.write(f'Block,Specialty,Day,Electives,Emergencies,,Start_times\n')
        c = 0
        for b in range(32):
            for l in range(9):
                for m in range(4):
                    if z[b][l][m]:
#                            f_out.write(f'{b},{spec_str[spec_b[b]-1]},{day[b]},{l},{m},,{or_cost_t[spec_b[b]-1][l][m][1:-1]}\n')
                        print(f'Block {b} with {l} electives, {m} emergencies (cost {or_cost[spec_b[b]-1][l][m]} ({spec_str[spec_b[b]-1]}), start times {or_cost_t[spec_b[b]-1][l][m]} day {day[b]})')
                        c += or_cost[spec_b[b]-1][l][m]
                        
                        times = [int(s) for s in re.findall(r'\d+', or_cost_t[spec_b[b]-1][l][m])]
                        temp_c = costs_noEmergency(spec_b[b]-1,times,l)
    #                    print(temp_c)
                        for i,j in enumerate(temp_c):
                            C[i] += j
                        Blocks.append([l,m,times,0])
                        
#            f_out.write(f'\nPatient,Block,Cancel_cost\n')        
        postp = 0
        for i in range(n_patients):
            if w[i]:
#                    f_out.write(f'{i},x\n')   
                print(f'Patient {i} postponed ({spec_str[spec_i[i]-1]},cost {int(postponing_cost[i])})')
                c += postponing_cost[i]
                C[4] += int(postponing_cost[i])
                postp += 1
            else:
                block = -1
                for b in range(32):
                    if x[b][i]:
                        block = b
                        if cancel_cost[i]<b_cancel_cost[b]:
                            b_cancel_cost[b]=cancel_cost[i]
                            b_perform_cost[b]=performing_cost[i]
#                f_out.write(f'{i},{block},{cancel_cost[i]}\n')  
                c += performing_cost[i]
                C[3] += int(performing_cost[i])
                    
                    
        print(f'{postp}/{n_pat} patients postponed')
                
        #print(f'\n cost = {sum(C)}')
        #print_cost(C)
                    
        print(f'\n\nSchedule with {n_patients} electives, {n_daily_emergencies} emergency spots per day')
        print('\n---------------------------------------------')
        
        for em in range(4):
            C_em = [0 for i in range(6)]
            C_em[3],C_em[4] = C[3],C[4]
            b,em_blocks,cancelled_b = find_em_block(em,n_daily_emergencies,Blocks.copy(),b_cancel_cost)
            C_em[3] += 5*cost_emer*em
            for ca in cancelled_b:
                C_em[3] -= int(b_perform_cost[ca])
                C_em[5] += int(b_cancel_cost[ca])+int(b_perform_cost[ca])
        #    print("-- ",em,em_blocks)
        #    print(b)
#            print(f'CANCELLED {cancelled_b}')
            
            for i in range(32):
                l,m = b[i][0],b[i][1]
                times = b[i][2]
                temp_c = costs_Emergency(spec_b[i]-1,times,l,m,b[i][3],b_cancel_cost[i])
        #        print(temp_c)
                for i,j in enumerate(temp_c):
                    C_em[i] += j
            print(f'\n{em} emergencies per day - cost = {sum(C_em)}')
            print_cost(C_em)
        print('---------------------------------------------')
    print('---------------------------------------------')