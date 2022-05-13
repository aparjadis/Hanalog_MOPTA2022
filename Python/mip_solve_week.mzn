int: B = 32;%blocks
int: D = 5;%days
int: S = 6;%specialties
int: I;%patients
int: EL = 8;%electives
int: EM = 3;%emergencies
int: M;%emergencies

array[1..B] of 1..D: day;
array[1..B] of 1..S: spec_b;
array[1..I] of 1..S: spec_i;
array[1..S,0..EL,0..EM] of int: cost;
array[1..I] of float: postponing_cost;
array[1..I] of float: performing_cost;

array[1..B,1..I] of var 0..1: x;%electives, patient i to block b
array[1..B] of var 0..EM: y;%emergency per block
array[1..I] of var 0..1: w;%patient postponed
array[1..B,0..EL,0..EM] of var 0..1: z;%constitution of blocks

var float: total_cost;

%total_cost
constraint total_cost = sum(b in 1..B,el in 0..EL,em in 0..EM)(cost[spec_b[b],el,em]*z[b,el,em]) + sum(i in 1..I)((w[i]*(postponing_cost[i]-performing_cost[i]))+performing_cost[i]);

%link patient attribution to block constitution
constraint forall(b in 1..B)( sum(i in 1..I)(x[b,i]) = sum(el in 0..EL,em in 0..EM)(el*z[b,el,em]) );

%link emergency attribution to block constitution
constraint forall(b in 1..B)( y[b] = sum(el in 0..EL,em in 0..EM)(em*z[b,el,em]) );

%only one z positive
constraint forall(b in 1..B)(sum(el in 0..EL,em in 0..EM)(z[b,el,em]) = 1);

%patient in only one block with the corresponding speciality, or postponed
constraint forall(i in 1..I)( sum(b in 1..B where spec_b[b]=spec_i[i])(x[b,i]) + w[i] = 1 );
% constraint forall(i in 1..I)( sum(b in 1..B where spec_b[b]!=spec_i[i])(x[b,i]) = 0 );
constraint forall(i in 1..I,b in 1..B where spec_b[b]!=spec_i[i])( x[b,i] = 0 );

%emergency per day
constraint forall(d in 1..D)(sum(b in 1..B where day[b]=d)(y[b]) = M);

solve minimize total_cost;%10311 with data patient

output
["Specialties are: 1-CARD, 2-GASTRO,3-GYN,4-MED,5-ORTH,6-URO\n\nTotal cost: " ++ show(total_cost) ++ "\n"]
++
[
    if b = 1 /\ el = 0 /\ em = 0 then "" else "," endif ++ show(z[b,el,em])
      | b in 1..B, el in 0..EL, em in 0..EM
]
% ++
% [
%     if em = 0 then "\nSpeciality " ++ show(spec_b[b]) ++ " (day " ++ show(day[b]) ++ "), " ++ show(el) ++ " electives: " ++ show(z[b,el,em]) else " " ++ show(z[b,el,em]) endif
%       | b in 1..B, el in 0..EL, em in 0..EM
% ]


% array[1..B] of 1..D: day = [1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5];
% array[1..B] of 1..S: spec_b = [2,1,5,3,6,1,6,2,5,5,3,3,6,2,2,1,4,3,3,6,5,2,5,3,3,6,2,1,5,3,6,1];
% array[1..I] of 1..S: spec_i = [2, 3, 3, 6, 3, 1, 3, 4, 3, 1, 5, 2, 1, 4, 5, 4, 1, 1, 5, 3, 1, 3, 1, 3, 5, 2, 3, 5, 2, 4, 1, 5, 2, 3, 6, 5, 2, 5, 3, 3, 2, 2, 6, 3, 2, 5, 1, 3, 1, 1, 6, 3, 5, 2, 3, 1, 6, 1, 6, 6, 2, 2, 2, 1, 3, 3, 6, 3, 2, 3];
% array[1..S,0..EL,0..EM] of int: cost = array3d(1..S,0..EL,0..EM,[480, 380, 285, 220, 380, 285, 220, 344, 291, 220, 343, 644, 219, 345, 643, 944, 347, 645, 943, 1244, 647, 945, 1243, 1544, 947, 1245, 1543, 1844, 1247, 1545, 1843, 2144, 1547, 1845, 2143, 2444, 480, 347, 246, 379, 347, 247, 380, 679, 237, 379, 680, 979, 382, 679, 980, 1279, 682, 979, 1280, 1579, 982, 1279, 1580, 1879, 1282, 1579, 1880, 2179, 1582, 1879, 2180, 2479, 1882, 2179, 2480, 2779, 480, 400, 324, 258, 401, 324, 258, 269, 332, 260, 271, 449, 266, 271, 450, 749, 271, 451, 750, 1049, 453, 751, 1050, 1349, 753, 1051, 1350, 1649, 1053, 1351, 1650, 1949, 1353, 1651, 1950, 2249, 480, 400, 330, 297, 399, 330, 296, 390, 337, 297, 391, 657, 296, 393, 662, 957, 391, 667, 962, 1257, 673, 967, 1262, 1557, 973, 1267, 1562, 1857, 1273, 1567, 1862, 2157, 1573, 1867, 2162, 2457, 480, 338, 216, 307, 338, 216, 309, 607, 208, 308, 609, 907, 307, 608, 909, 1207, 607, 908, 1209, 1507, 907, 1208, 1509, 1807, 1207, 1508, 1809, 2107, 1507, 1808, 2109, 2407, 1807, 2108, 2409, 2707, 480, 408, 336, 266, 407, 336, 266, 218, 343, 271, 219, 266, 279, 218, 262, 491, 221, 262, 494, 791, 261, 479, 794, 1091, 473, 779, 1094, 1391, 773, 1079, 1394, 1691, 1073, 1379, 1694, 1991]);
% array[1..I] of int: postponing_cost = [60, 57, 42, 56, 48, 117, 60, 19, 48, 30, 73, 31, 65, 60, 48, 60, 90, 81, 77, 71, 77, 28, 39, 34, 90, 39, 49, 43, 24, 30, 90, 41, 16, 18, 71, 38, 73, 51, 46, 34, 36, 58, 44, 46, 65, 60, 15, 17, 55, 33, 58, 52, 90, 46, 60, 54, 15, 88, 90, 56, 71, 21, 19, 60, 37, 31, 32, 90, 32, 19];
% array[1..I] of int: perfoming_cost = [40, 38, 28, 37, 32, 78, 40, 12, 32, 20, 49, 21, 43, 40, 32, 40, 60, 54, 51, 47, 51, 18, 26, 22, 60, 26, 32, 28, 16, 20, 60, 27, 11, 12, 47, 25, 48, 34, 31, 23, 24, 39, 29, 30, 43, 40, 10, 11, 36, 22, 38, 34, 60, 30, 40, 36, 10, 59, 60, 37, 47, 14, 12, 40, 25, 20, 21, 60, 21, 13];
