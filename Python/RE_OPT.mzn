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
set of int: ACTION;
%output ["s = ",show(ELECTIVES)];
array[ELECTIVES, ACTION] of int: duration; 
array[ELECTIVES, ACTION] of int: start_elective_current;

array[EMER] of int:  emer_duration;
array[N_EMER] of int:  n_emer_duration;

float :C_over;
float :C_wait;
float :C_idle;
array[ELECTIVES] of float: C_canc;

int: allowed_OT;
int: cap_rem;
int: maxt = cap_rem + allowed_OT;
set of int: TIME = 0..maxt;

array[ELECTIVES,ACTION] of var opt TIME:    new_e_start_action;  
array[ELECTIVES] of var TIME:    new_e_start;
array[ELECTIVES] of var TIME:    new_e_duration;
array[N_EMER] of var TIME: n_emer_start;
array[EMER] of var TIME:    emer_start;  


var TIME: end; 
var 0..allowed_OT: ot; 
array[ELECTIVES] of var 0..maxt: wt;

array[ELECTIVES] of var bool:    cancel;
var float : obj;


include "globals.mzn";
include "gecode.mzn";


constraint disjunctive([new_e_start_action[i,k] | i in ELECTIVES, k in ACTION] ++ [emer_start[i] | i in EMER] ++ [n_emer_start[i] | i in N_EMER],
                       [duration[i,k] | i in ELECTIVES, k in ACTION] ++ [emer_duration[i]  | i in EMER] ++ [n_emer_duration[i] | i in N_EMER]);
                       %no two surgeries run simultaneously
                       
constraint forall(i in ELECTIVES)(alternative(new_e_start[i],new_e_duration[i],
                 [new_e_start_action[i,k]|k in ACTION],[duration[i,k]|k in ACTION]));
                 %either a surgery is postponed or rescheduled

constraint ot >= (end - cap_rem);
%overtime

constraint forall(i in ELECTIVES)(cancel[i] = absent(new_e_start_action[i,0]));
%if a surgery is cancelled

constraint forall(i in ELECTIVES)(wt[i] >= new_e_start_action[i,0] -  start_elective_current[i,0]);
%waiting time calculation for a elective surgery

constraint forall(i in EMER, j in ELECTIVES where acuity < 3)(emer_start[i] <= new_e_start[j]);
%the most urgent emrgency has to start next

constraint end = max([new_e_start[i] + new_e_duration[i] | i in ELECTIVES] ++ 
                     [emer_start[i] + emer_duration[i]| i in EMER] ++ 
                     [n_emer_start[i] + n_emer_duration[i]| i in N_EMER]) ;
                     %end of the day

constraint forall(i in ELECTIVES,k in ACTION)(new_e_start_action[i,k] >=  start_elective_current[i,k]);
%no surgery starts befiore the original starting time

constraint obj = C_over * ot + C_wait * sum(i in ELECTIVES)(wt[i]) +   sum(i in ELECTIVES)(C_canc[i] * cancel[i]) ;
%objective of overtime, waiting time, canceleation

%solve satisfy;
solve   minimize obj;

 
%output ["s = ",show(new_start_action[1,0])]