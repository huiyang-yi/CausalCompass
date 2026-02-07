#!/bin/bash

# CUTS experiment script
# Simple syntax, easy to understand and extend
# chmod +x run_cuts.sh
# ./run_cuts.sh

echo "Starting CUTS experiments..."

# Parameters
DEVICE="cuda:1"                    # Device to use for all experiments
P_VALUES="10"                      #"10 20"
T_VALUES="500 1000"                #"250 500 1000"
F_VALUES="10 40"                        #"10 40"
SEEDS="0,1,2,3,4"                         #"0,1,2,3,4"

# CUTS hyperparameters
INPUT_STEP_VALUES="1 3 5 10"              #1 3 5 10
BATCH_SIZE_VALUES="32 128"                #32 128
WEIGHT_DECAY_VALUES="0 0.001 0.003"       #0 0.001 0.003

NORMALIZATION_METHODS="zscore minmax"
RHO_VALUES="0.5"
GAMMA_VALUES="1.2"
MISSING_PROB_VALUES="0.4"
DISCRETE_RATIO_VALUES="0.5"
GAUSSIAN_RATIO_VALUES="0.5"
# Nonstationary specific noise_std values
NOISE_STD_VAR="1.0"
MEAN_LOG_SIGMA_VAR="1.0"
#MEAN_LOG_SIGMA_VAR="2.7"
NOISE_STD_LORENZ_F10="2.0"
MEAN_LOG_SIGMA_LORENZ_F10="2.5"
NOISE_STD_LORENZ_F40="2.0"
MEAN_LOG_SIGMA_LORENZ_F40="3.5"

FAILED_COUNT=0
TOTAL_COUNT=0

# Function to run experiments
run_experiment() {
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo "[$TOTAL_COUNT] Running: $*"

    python ../run.py "$@" --seeds $SEEDS --device $DEVICE
    if [ $? -ne 0 ]; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo "  *** FAILED ***"
        echo "FAILED: $*" >> failed_experiments.txt
    fi
}

echo "=== 1. vanilla dataset ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for input_step in $INPUT_STEP_VALUES; do
            for batch_size in $BATCH_SIZE_VALUES; do
                for weight_decay in $WEIGHT_DECAY_VALUES; do
                    # VAR
                    run_experiment --dataset_type vanilla --data_model VAR --method cuts --p $p --T $T --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type vanilla --data_model Lorenz --method cuts --p $p --T $T --F $F --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay
                    done
                done
            done
        done
    done
done

echo "=== 2. standardized dataset ==="
for norm in $NORMALIZATION_METHODS; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for input_step in $INPUT_STEP_VALUES; do
                for batch_size in $BATCH_SIZE_VALUES; do
                    for weight_decay in $WEIGHT_DECAY_VALUES; do
                        # VAR
                        run_experiment --dataset_type standardized --data_model VAR --method cuts --p $p --T $T --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay --normalization_method $norm
                        # Lorenz
                        for F in $F_VALUES; do
                            run_experiment --dataset_type standardized --data_model Lorenz --method cuts --p $p --T $T --F $F --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay --normalization_method $norm
                        done
                    done
                done
            done
        done
    done
done

echo "=== 3. trendseason dataset ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for input_step in $INPUT_STEP_VALUES; do
            for batch_size in $BATCH_SIZE_VALUES; do
                for weight_decay in $WEIGHT_DECAY_VALUES; do
                    # VAR
                    run_experiment --dataset_type trendseason --data_model VAR --method cuts --p $p --T $T --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type trendseason --data_model Lorenz --method cuts --p $p --T $T --F $F --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay
                    done
                done
            done
        done
    done
done

echo "=== 4. confounder dataset ==="
for rho in $RHO_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for input_step in $INPUT_STEP_VALUES; do
                for batch_size in $BATCH_SIZE_VALUES; do
                    for weight_decay in $WEIGHT_DECAY_VALUES; do
                        # VAR
                        run_experiment --dataset_type confounder --data_model VAR --method cuts --p $p --T $T --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay --rho $rho
                        # Lorenz
                        for F in $F_VALUES; do
                            run_experiment --dataset_type confounder --data_model Lorenz --method cuts --p $p --T $T --F $F --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay --rho $rho
                        done
                    done
                done
            done
        done
    done
done

echo "=== 6. measurement_error dataset ==="
for gamma in $GAMMA_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for input_step in $INPUT_STEP_VALUES; do
                for batch_size in $BATCH_SIZE_VALUES; do
                    for weight_decay in $WEIGHT_DECAY_VALUES; do
                        # VAR
                        run_experiment --dataset_type measurement_error --data_model VAR --method cuts --p $p --T $T --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay --gamma $gamma
                        # Lorenz
                        for F in $F_VALUES; do
                            run_experiment --dataset_type measurement_error --data_model Lorenz --method cuts --p $p --T $T --F $F --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay --gamma $gamma
                        done
                    done
                done
            done
        done
    done
done

echo "=== 8. missing dataset ==="
for prob in $MISSING_PROB_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for input_step in $INPUT_STEP_VALUES; do
                for batch_size in $BATCH_SIZE_VALUES; do
                    for weight_decay in $WEIGHT_DECAY_VALUES; do
                        # VAR
                        run_experiment --dataset_type missing --data_model VAR --method cuts --p $p --T $T --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay --missing_prob $prob
                        # Lorenz
                        for F in $F_VALUES; do
                            run_experiment --dataset_type missing --data_model Lorenz --method cuts --p $p --T $T --F $F --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay --missing_prob $prob
                        done
                    done
                done
            done
        done
    done
done

echo "=== 9. mixed_data dataset ==="
for ratio in $DISCRETE_RATIO_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for input_step in $INPUT_STEP_VALUES; do
                for batch_size in $BATCH_SIZE_VALUES; do
                    for weight_decay in $WEIGHT_DECAY_VALUES; do
                        # VAR
                        run_experiment --dataset_type mixed_data --data_model VAR --method cuts --p $p --T $T --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay --discrete_ratio $ratio
                        # Lorenz
                        for F in $F_VALUES; do
                            run_experiment --dataset_type mixed_data --data_model Lorenz --method cuts --p $p --T $T --F $F --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay --discrete_ratio $ratio
                        done
                    done
                done
            done
        done
    done
done


echo "=== 11. nonstationary dataset ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for input_step in $INPUT_STEP_VALUES; do
            for batch_size in $BATCH_SIZE_VALUES; do
                for weight_decay in $WEIGHT_DECAY_VALUES; do
                    # VAR
                    run_experiment --dataset_type nonstationary --data_model VAR --method cuts \
                        --p $p --T $T --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay \
                        --noise_std $NOISE_STD_VAR --mean_log_sigma $MEAN_LOG_SIGMA_VAR

                    # Lorenz F=10
                    run_experiment --dataset_type nonstationary --data_model Lorenz --method cuts \
                        --p $p --T $T --F 10 --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay \
                        --noise_std $NOISE_STD_LORENZ_F10 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F10

                    # Lorenz F=40
                    run_experiment --dataset_type nonstationary --data_model Lorenz --method cuts \
                        --p $p --T $T --F 40 --cuts_input_step $input_step --cuts_batch_size $batch_size --cuts_weight_decay $weight_decay \
                        --noise_std $NOISE_STD_LORENZ_F40 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F40
                done
            done
        done
    done
done


echo ""
echo "All CUTS experiments completed!"
echo ""
# Final summary
echo "================================="
echo "Experiment completion status:"
echo "- Total experiments: $TOTAL_COUNT"
echo "- Successful: $((TOTAL_COUNT - FAILED_COUNT))"
echo "- Failed: $FAILED_COUNT"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "- Failed experiment details: failed_experiments.txt"
fi
echo "================================="

if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi