#!/bin/bash

NUM_PROCESSES="2 4 8"

OUTPUT_FILE=time_mpi_bcast.txt

for n in $NUM_PROCESSES; do
    echo "Running with $n processes"
    output=$(mpiexec -n $n python3 mpi_bcast.py)
    echo "$output" >> $OUTPUT_FILE
done
