#!/bin/bash

# VAR experiment script
# Simple syntax, easy to understand and extend
# chmod +x run_var.sh
# ./run_var.sh

echo "Starting VAR experiments..."

# Parameters
P_VALUES="10"                        #"10 20"
T_VALUES="500 1000"                #"250 500 1000"
F_VALUES="10 40"                        #"10 40"
SEEDS="0,1,2,3,4"                     #"0,1,2,3,4"

# VAR hyperparameters
TAU_MAX_VALUES="1 2 3 4 5"                    # "1 2 3 4 5"
THRESHOLD_VALUES="0 0.01 0.05 0.1 0.3"        # "0 0.01 0.05 0.1 0.3"

# ===== Data Parameter Configuration Section =====
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
# ===== End of Data Parameter Configuration =====

FAILED_COUNT=0
TOTAL_COUNT=0

# Function to run experiments
run_experiment() {
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo "[$TOTAL_COUNT] Running: $*"

    python ../run.py "$@" --seeds $SEEDS
    if [ $? -ne 0 ]; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo "  *** FAILED ***"
        echo "FAILED: $*" >> failed_experiments.txt
    fi
}

echo "=== 1. vanilla dataset ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for tau in $TAU_MAX_VALUES; do
            for threshold in $THRESHOLD_VALUES; do
                # VAR
                run_experiment --dataset_type vanilla --data_model VAR --method var --p $p --T $T --tau_max $tau --threshold $threshold
                # Lorenz
                for F in $F_VALUES; do
                    run_experiment --dataset_type vanilla --data_model Lorenz --method var --p $p --T $T --F $F --tau_max $tau --threshold $threshold
                done
            done
        done
    done
done

echo "=== 2. standardized dataset ==="
for norm in $NORMALIZATION_METHODS; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for tau in $TAU_MAX_VALUES; do
                for threshold in $THRESHOLD_VALUES; do
                    # VAR
                    run_experiment --dataset_type standardized --data_model VAR --method var --p $p --T $T --tau_max $tau --threshold $threshold --normalization_method $norm
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type standardized --data_model Lorenz --method var --p $p --T $T --F $F --tau_max $tau --threshold $threshold --normalization_method $norm
                    done
                done
            done
        done
    done
done

echo "=== 3. trendseason dataset ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for tau in $TAU_MAX_VALUES; do
            for threshold in $THRESHOLD_VALUES; do
                # VAR
                run_experiment --dataset_type trendseason --data_model VAR --method var --p $p --T $T --tau_max $tau --threshold $threshold
                # Lorenz
                for F in $F_VALUES; do
                    run_experiment --dataset_type trendseason --data_model Lorenz --method var --p $p --T $T --F $F --tau_max $tau --threshold $threshold
                done
            done
        done
    done
done

echo "=== 4. confounder dataset ==="
for rho in $RHO_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for tau in $TAU_MAX_VALUES; do
                for threshold in $THRESHOLD_VALUES; do
                    # VAR
                    run_experiment --dataset_type confounder --data_model VAR --method var --p $p --T $T --tau_max $tau --threshold $threshold --rho $rho
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type confounder --data_model Lorenz --method var --p $p --T $T --F $F --tau_max $tau --threshold $threshold --rho $rho
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
            for tau in $TAU_MAX_VALUES; do
                for threshold in $THRESHOLD_VALUES; do
                    # VAR
                    run_experiment --dataset_type measurement_error --data_model VAR --method var --p $p --T $T --tau_max $tau --threshold $threshold --gamma $gamma
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type measurement_error --data_model Lorenz --method var --p $p --T $T --F $F --tau_max $tau --threshold $threshold --gamma $gamma
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
            for tau in $TAU_MAX_VALUES; do
                for threshold in $THRESHOLD_VALUES; do
                    # VAR
                    run_experiment --dataset_type missing --data_model VAR --method var --p $p --T $T --tau_max $tau --threshold $threshold --missing_prob $prob
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type missing --data_model Lorenz --method var --p $p --T $T --F $F --tau_max $tau --threshold $threshold --missing_prob $prob
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
            for tau in $TAU_MAX_VALUES; do
                for threshold in $THRESHOLD_VALUES; do
                    # VAR
                    run_experiment --dataset_type mixed_data --data_model VAR --method var --p $p --T $T --tau_max $tau --threshold $threshold --discrete_ratio $ratio
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type mixed_data --data_model Lorenz --method var --p $p --T $T --F $F --tau_max $tau --threshold $threshold --discrete_ratio $ratio
                    done
                done
            done
        done
    done
done


echo "=== 11. nonstationary dataset ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for tau in $TAU_MAX_VALUES; do
            for threshold in $THRESHOLD_VALUES; do
                # VAR
                run_experiment --dataset_type nonstationary --data_model VAR --method var \
                    --p $p --T $T --tau_max $tau --threshold $threshold \
                    --noise_std $NOISE_STD_VAR --mean_log_sigma $MEAN_LOG_SIGMA_VAR

                # Lorenz F=10
                run_experiment --dataset_type nonstationary --data_model Lorenz --method var \
                    --p $p --T $T --F 10 --tau_max $tau --threshold $threshold \
                    --noise_std $NOISE_STD_LORENZ_F10 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F10

                # Lorenz F=40
                run_experiment --dataset_type nonstationary --data_model Lorenz --method var \
                    --p $p --T $T --F 40 --tau_max $tau --threshold $threshold \
                    --noise_std $NOISE_STD_LORENZ_F40 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F40
            done
        done
    done
done


echo ""
echo "All VAR experiments completed!"
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