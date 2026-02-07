#!/bin/bash

#chmod +x run_all.sh
#./run_all.sh

cd "$(dirname "$0")"

# Ensure all run sh files have execute permission
chmod +x run_*.sh 2>/dev/null
echo "Starting all causal discovery experiments (run version)..."

start_time=$(date)
echo "Start time: $start_time"

FAILED_SCRIPTS=0

# Function to run script with error handling
run_script() {
    script_name=$1
    echo "=== Running $script_name experiments ==="
    if [ -f "./$script_name" ]; then
        ./$script_name
        if [ $? -ne 0 ]; then
            echo "WARNING: $script_name failed!"
            FAILED_SCRIPTS=$((FAILED_SCRIPTS + 1))
        fi
    else
        echo "ERROR: $script_name not found!"
        FAILED_SCRIPTS=$((FAILED_SCRIPTS + 1))
    fi
}

# Run all method scripts
run_script "run_var.sh"
run_script "run_lgc.sh"
run_script "run_pcmci.sh"
run_script "run_dynotears.sh"
run_script "run_varlingam.sh"
run_script "run_tsci.sh"
run_script "run_ntsnotears.sh"
run_script "run_ngc.sh"
run_script "run_cuts.sh"
run_script "run_cutsplus.sh"

end_time=$(date)
echo ""
echo "All experiments completed!"
echo "Start time: $start_time"
echo "End time: $end_time"
echo "Failed scripts: $FAILED_SCRIPTS"

if [ $FAILED_SCRIPTS -gt 0 ]; then
    exit 1
else
    exit 0
fi