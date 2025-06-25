#!/bin/bash

NUM_PROCESSES="2 4 8 16 32"
OUTPUT_FILE=time_omp.txt

for n in $NUM_PROCESSES; do
    echo "Running with $n processes"
    
    # Measure the execution time using the 'time' command
    time_taken=$(time -p python3 omp.py 2>&1 | grep real | awk '{print $2}')

    echo "Completed in $time_taken seconds"
    echo "$time_taken   $n" >> $OUTPUT_FILE
done
