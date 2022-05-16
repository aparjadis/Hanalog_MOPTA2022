% Preparing a banquet in the least time
%enum ELECTIVE_id; % the set of electives
%enum EMER_id; % the set of emergencies
int: num_electives; % total number of electives
int: num_emer; % total number of emergencies
int: num_n_emer; % total number of non emergent emergencies
int: acuity;
set of int: ELECTIVES = 1..num_electives;
set of int: EMER = 1..num_emer;
set of int: N_EMER = 1..num_n_emer ;
set of int: ACTION; %two possible action, to perform or to postpone
%output ["s = ",show(ELECTIVES)];
array[ELECTIVES, ACTION] of int: duration; %expec durations  of electives
array[ELECTIVES, ACTION] of int: start_elective_current; %current start times of electives

array[EMER] of int:  emer_duration; %expec durations of current emergencies which we want to place in the schdule
array[N_EMER] of int:  n_emer_duration; %expec durations of current emergencies already in the schdule

float :C_over;
float :C_wait;
float :C_idle;
array[ELECTIVES] of float: C_canc;

int: allowed_OT;% allowed ot for the hospital
int: cap_rem;%remaining capapcity for work time
int: maxt = cap_rem + allowed_OT; %max time capacity available to the hospital
set of int: TIME = 0..maxt;

array[ELECTIVES,ACTION] of var opt TIME:    new_e_start_action;  %new start time for electives for given action. This is dummy variable since only one action will be executed
array[ELECTIVES] of var TIME:    new_e_start;%actual start times of electives
array[ELECTIVES] of var TIME:    new_e_duration;%dummy variable for duration. I dthe surgery stands cancelled then duration is zero else it is given the actual expected durations
array[N_EMER] of var TIME: n_emer_start;% new start times for scheduled emergencies
array[EMER] of var TIME:    emer_start;  %new start time for the emergency to be scheduled


var TIME: end; %expected end of all surgeries
var 0..allowed_OT: ot; %variable to record overtime
array[ELECTIVES] of var 0..maxt: wt;%variable to store waiting time for each elective

array[ELECTIVES] of var bool:    cancel;%indicator if an electie is cancelled
var float : obj;%objective function


include "globals.mzn";
include "gecode.mzn";


constraint disjunctive([new_e_start_action[i,k] | i in ELECTIVES, k in ACTION] ++ [emer_start[i] | i in EMER] ++ [n_emer_start[i] | i in N_EMER],
                       [duration[i,k] | i in ELECTIVES, k in ACTION] ++ [emer_duration[i]  | i in EMER] ++ [n_emer_duration[i] | i in N_EMER]);
                       %no two surgeries run simultaneously of any type
                       % disjunctive([new_e_start_action, emer_start, n_emer_start] [duration, emer_duration, n_emer_duration])
                       
constraint forall(i in ELECTIVES)(alternative(new_e_start[i],new_e_duration[i],
                 [new_e_start_action[i,k]|k in ACTION],[duration[i,k]|k in ACTION]));
                 %either a surgery is postponed or rescheduled, the action 0 is performing the surgery on the same day, action 1 is cancelling
                 
                 %alternative(new_e_start, new_e_duration, [new_e_start_action, duration]) for all electives 
                 %Alternative constraint for optional tasks. Task ( s0 , d0 ) spans the optional tasks ( s [ i ], d [ i ]) in the array arguments and at most one can occur

constraint ot >= (end - cap_rem);
%overtime is time when exceed the remaing nornal hour capacity, end being the  expected end time for last scheduled surgery

constraint forall(i in ELECTIVES)(cancel[i] = absent(new_e_start_action[i,0]));
%if a surgery is cancelled, the indicator cancel is set to 1, absent returns true iff option type variable v takes the value None, here if we postpone, the cancel value will be none and vice versa.

constraint forall(i in ELECTIVES)(wt[i] >= new_e_start_action[i,0] -  start_elective_current[i,0]);
%waiting time calculation for a elective surgery

constraint forall(i in EMER, j in ELECTIVES where acuity < 3)(emer_start[i] <= new_e_start[j]);
%the most urgent emrgency has to start next before all electives

constraint end = max([new_e_start[i] + new_e_duration[i] | i in ELECTIVES] ++ 
                     [emer_start[i] + emer_duration[i]| i in EMER] ++ 
                     [n_emer_start[i] + n_emer_duration[i]| i in N_EMER]) ;
                     %end of the day i sthe max expected end, end of any surgery is start + durations

constraint forall(i in ELECTIVES,k in ACTION)(new_e_start_action[i,k] >=  start_elective_current[i,k]);
%no elective surgery starts before the original starting time

constraint obj = C_over * ot + C_wait * sum(i in ELECTIVES)(wt[i]) +   sum(i in ELECTIVES)(C_canc[i] * cancel[i]) ;
%objective of overtime, waiting time, cancelation

%solve satisfy;
solve   minimize obj;

 
%output ["s = ",show(new_start_action[1,0])]