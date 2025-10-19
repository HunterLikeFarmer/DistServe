#!/bin/bash

export DATASET=/home/exouser/dataset

# Clear the output file or create it fresh
> sim_result.txt

# Define arrays for the parameter combinations
num_nodes=(4 5)
ngpus_per_node=(2 4)

# Loop through all combinations
for num_node in "${num_nodes[@]}"; do
    for ngpu_per_node in "${ngpus_per_node[@]}"; do
        python -m simdistserve.simulate_print_all \
            --num-node ${num_node} \
            --ngpu-per-node ${ngpu_per_node} \
            --model-type "facebook/opt-13b" \
            --workload sharegpt \
            --backend distserve \
            --prefill-target 200 \
            --decode-target 100 \
            --prefill-percentage 90 \
            --decode-percentage 90 \
            --max-per-gpu-rate 5 \
            --esp 0.25 \
            --N 300 >> sim_result.txt 2>&1
    done
done

echo "All simulations completed. Results saved to sim_result.txt"