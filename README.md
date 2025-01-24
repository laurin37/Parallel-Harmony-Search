# Parallel-Harmony-Search
This is parallelization of Harmony Search Algorithm. 

# compile Serial
```
g++ -std=c++11 -g -Wall -o serial.out harmonySearchSerial.cpp
```

# compile MPI
```
mpic++ -std=c++11 -g -Wall harmonySearchParallelMPI.cpp -o mpi.out
```

# compile OpenMP
```
g++ -std=c++11 -g -Wall -o openmp.out harmonySearchParallelOpenMP.cpp
```

# compile Hybrid
```
mpic++ -std=c++11 -g -Wall harmonySearchParallelHybrid.cpp -o hybrid.out
```


# submit to hpc
```
qsub harmonySearch.sh
```

# analyze run
```
python analyze_hs.py ../data/harmony_search_results.csv --max_iter 100000000 --dim 35 --output my_analysis
```