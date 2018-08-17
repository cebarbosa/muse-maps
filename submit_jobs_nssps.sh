#!/bin/bash
#PBS -lwalltime=12:00:00
## create 6 array jobs, with $PBS_ARRAYID set for each job
#PBS -t 1-104
#PBS -l nodes=1:ppn=24
#PBS -j oe
  
# ------------------------------------------------------------------------
# Start of the input; change these as required
# the start and end array-job id, taken from the -t option in array jobs.
T_START=1
T_END=104
  
# start and end parameter for all array jobs; these will be used as input
# to your program
I_START=1
I_END=830

# end of the parameters; do not change below, until where you run your 
# program
# ------------------------------------------------------------------------

# compute the number of tasks from start and end. The start and end are
# inclusive; the actual number if tasks is therefore one more than the
# difference between end and start.
N_I=$(( $I_END - $I_START + 1 ))

# the same goes for the number of array jobs
N_T=$(( $T_END - $T_START + 1 ))
  
# Compute the range of task ids for this job
IT_START=$(( $I_START + $N_I * ($PBS_ARRAYID - $T_START) / $N_T ))
IT_END=$(( $I_START + $N_I * ($PBS_ARRAYID - $T_START + 1) / $N_T - 1 ))


# ------------------------------------------------------------------------
# Now the start and end range for this job have been defined; use
# GNU Parallel to actually run the tasks within this job efficiently.
#
# Change this to make it run your program or script. 

# The example below will execute something like
#
# ./my_program $IT_START          &
# ./my_program $IT_START + 1      &
# ./my_program $IT_START + 2      &
# ...
# ./my_program $IT_END            &
#
# and so on, but it will keep the number of running tasks equal to the number
# of cpu cores, until all work is finished, so it will not overload the
# system.

# ------------------------------------------------------------------------
# Change "./my_program" to your script; it will actually be run
# with each parameter as command-line argument
module load parallel/20131122
cd /sto/home/cebarbosa/hydraimf/repo/hydraimf/
seq $IT_START $IT_END | parallel ./task_nssps.sh
