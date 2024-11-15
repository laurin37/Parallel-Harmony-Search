#!/bin/bash
#PBS -l select=1:ncpus=4:mem=2gb
# set max execution time
#PBS -l walltime=3:00:00
# imposta la coda di esecuzione
#PBS -q short_cpuQ

module load mpich-3.2

./Parallel-Harmony-Search/src/a.out 1000 100000
./Parallel-Harmony-Search/src/a.out 1000 1000000
./Parallel-Harmony-Search/src/a.out 1000 10000000 # 1.5 Minutes
#./Parallel-Harmony-Search/src/a.out 1000 100000000 # 14 Minutes
#./Parallel-Harmony-Search/src/a.out 1000 1000000000 # 2 Hours 20 Minutes