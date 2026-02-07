#!/bin/bash

# NGC experiment script (cMLP and cLSTM)
# Simple syntax, easy to understand and extend
# chmod +x run_ngc.sh
# ./run_ngc.sh

echo "Starting NGC experiments..."

# Parameters
DEVICE="cuda:1"                    # Device to use for all experiments
P_VALUES="15"                      #"10 20"
T_VALUES="500 1000"                #"250 500 1000"
F_VALUES="10 40"                        #"10 40"
SEEDS="0,1,2,3,4"                         #"0,1,2,3,4"

# NGC hyperparameters
LAM_VALUES="0.0001 0.005 0.05"                 # "0.0001 0.005 0.01 0.02 0.05"
LR_VALUES_VAR="0.01 0.1"                            # "0.01 0.05 0.1"
LR_VALUES_LORENZ="0.0005 0.001"                     # "0.0002 0.0005 0.001 0.002"

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
        for lam in $LAM_VALUES; do
            # VAR experiments with VAR learning rates
            for lr in $LR_VALUES_VAR; do
                # cMLP VAR
                run_experiment --dataset_type vanilla --data_model VAR --method cmlp --p $p --T $T --lam $lam --learning_rate $lr
                # cLSTM VAR
                run_experiment --dataset_type vanilla --data_model VAR --method clstm --p $p --T $T --lam $lam --learning_rate $lr
            done

            # Lorenz experiments with Lorenz learning rates
            for lr in $LR_VALUES_LORENZ; do
                for F in $F_VALUES; do
                    # cMLP Lorenz
                    run_experiment --dataset_type vanilla --data_model Lorenz --method cmlp --p $p --T $T --F $F --lam $lam --learning_rate $lr
                    # cLSTM Lorenz
                    run_experiment --dataset_type vanilla --data_model Lorenz --method clstm --p $p --T $T --F $F --lam $lam --learning_rate $lr
                done
            done
        done
    done
done

echo "=== 2. standardized dataset ==="
for norm in $NORMALIZATION_METHODS; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for lam in $LAM_VALUES; do
                # VAR experiments with VAR learning rates
                for lr in $LR_VALUES_VAR; do
                    # cMLP VAR
                    run_experiment --dataset_type standardized --data_model VAR --method cmlp --p $p --T $T --lam $lam --learning_rate $lr --normalization_method $norm
                    # cLSTM VAR
                    run_experiment --dataset_type standardized --data_model VAR --method clstm --p $p --T $T --lam $lam --learning_rate $lr --normalization_method $norm
                done

                # Lorenz experiments with Lorenz learning rates
                for lr in $LR_VALUES_LORENZ; do
                    for F in $F_VALUES; do
                        # cMLP Lorenz
                        run_experiment --dataset_type standardized --data_model Lorenz --method cmlp --p $p --T $T --F $F --lam $lam --learning_rate $lr --normalization_method $norm
                        # cLSTM Lorenz
                        run_experiment --dataset_type standardized --data_model Lorenz --method clstm --p $p --T $T --F $F --lam $lam --learning_rate $lr --normalization_method $norm
                    done
                done
            done
        done
    done
done

echo "=== 3. trendseason dataset ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for lam in $LAM_VALUES; do
            # VAR experiments with VAR learning rates
            for lr in $LR_VALUES_VAR; do
                # cMLP VAR
                run_experiment --dataset_type trendseason --data_model VAR --method cmlp --p $p --T $T --lam $lam --learning_rate $lr
                # cLSTM VAR
                run_experiment --dataset_type trendseason --data_model VAR --method clstm --p $p --T $T --lam $lam --learning_rate $lr
            done

            # Lorenz experiments with Lorenz learning rates
            for lr in $LR_VALUES_LORENZ; do
                for F in $F_VALUES; do
                    # cMLP Lorenz
                    run_experiment --dataset_type trendseason --data_model Lorenz --method cmlp --p $p --T $T --F $F --lam $lam --learning_rate $lr
                    # cLSTM Lorenz
                    run_experiment --dataset_type trendseason --data_model Lorenz --method clstm --p $p --T $T --F $F --lam $lam --learning_rate $lr
                done
            done
        done
    done
done

echo "=== 4. confounder dataset ==="
for rho in $RHO_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for lam in $LAM_VALUES; do
                # VAR experiments with VAR learning rates
                for lr in $LR_VALUES_VAR; do
                    # cMLP VAR
                    run_experiment --dataset_type confounder --data_model VAR --method cmlp --p $p --T $T --lam $lam --learning_rate $lr --rho $rho
                    # cLSTM VAR
                    run_experiment --dataset_type confounder --data_model VAR --method clstm --p $p --T $T --lam $lam --learning_rate $lr --rho $rho
                done

                # Lorenz experiments with Lorenz learning rates
                for lr in $LR_VALUES_LORENZ; do
                    for F in $F_VALUES; do
                        # cMLP Lorenz
                        run_experiment --dataset_type confounder --data_model Lorenz --method cmlp --p $p --T $T --F $F --lam $lam --learning_rate $lr --rho $rho
                        # cLSTM Lorenz
                        run_experiment --dataset_type confounder --data_model Lorenz --method clstm --p $p --T $T --F $F --lam $lam --learning_rate $lr --rho $rho
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
            for lam in $LAM_VALUES; do
                # VAR experiments with VAR learning rates
                for lr in $LR_VALUES_VAR; do
                    # cMLP VAR
                    run_experiment --dataset_type measurement_error --data_model VAR --method cmlp --p $p --T $T --lam $lam --learning_rate $lr --gamma $gamma
                    # cLSTM VAR
                    run_experiment --dataset_type measurement_error --data_model VAR --method clstm --p $p --T $T --lam $lam --learning_rate $lr --gamma $gamma
                done

                # Lorenz experiments with Lorenz learning rates
                for lr in $LR_VALUES_LORENZ; do
                    for F in $F_VALUES; do
                        # cMLP Lorenz
                        run_experiment --dataset_type measurement_error --data_model Lorenz --method cmlp --p $p --T $T --F $F --lam $lam --learning_rate $lr --gamma $gamma
                        # cLSTM Lorenz
                        run_experiment --dataset_type measurement_error --data_model Lorenz --method clstm --p $p --T $T --F $F --lam $lam --learning_rate $lr --gamma $gamma
                    done
                done
            done
        done
    done
done

echo "=== 7. mechanism_violation dataset ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for lam in $LAM_VALUES; do
            # VAR experiments with VAR learning rates
            for lr in $LR_VALUES_VAR; do
                run_experiment --dataset_type mechanism_violation --data_model VAR --method cmlp --p $p --T $T --lam $lam --learning_rate $lr
                run_experiment --dataset_type mechanism_violation --data_model VAR --method clstm --p $p --T $T --lam $lam --learning_rate $lr
            done

            # Lorenz experiments with Lorenz learning rates
            for lr in $LR_VALUES_LORENZ; do
                run_experiment --dataset_type mechanism_violation --data_model Lorenz --method cmlp --p $p --T $T --lam $lam --learning_rate $lr
                run_experiment --dataset_type mechanism_violation --data_model Lorenz --method clstm --p $p --T $T --lam $lam --learning_rate $lr
            done
        done
    done
done

echo "=== 8. missing dataset ==="
for prob in $MISSING_PROB_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for lam in $LAM_VALUES; do
                # VAR experiments with VAR learning rates
                for lr in $LR_VALUES_VAR; do
                    # cMLP VAR
                    run_experiment --dataset_type missing --data_model VAR --method cmlp --p $p --T $T --lam $lam --learning_rate $lr --missing_prob $prob
                    # cLSTM VAR
                    run_experiment --dataset_type missing --data_model VAR --method clstm --p $p --T $T --lam $lam --learning_rate $lr --missing_prob $prob
                done

                # Lorenz experiments with Lorenz learning rates
                for lr in $LR_VALUES_LORENZ; do
                    for F in $F_VALUES; do
                        # cMLP Lorenz
                        run_experiment --dataset_type missing --data_model Lorenz --method cmlp --p $p --T $T --F $F --lam $lam --learning_rate $lr --missing_prob $prob
                        # cLSTM Lorenz
                        run_experiment --dataset_type missing --data_model Lorenz --method clstm --p $p --T $T --F $F --lam $lam --learning_rate $lr --missing_prob $prob
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
            for lam in $LAM_VALUES; do
                # VAR experiments with VAR learning rates
                for lr in $LR_VALUES_VAR; do
                    # cMLP VAR
                    run_experiment --dataset_type mixed_data --data_model VAR --method cmlp --p $p --T $T --lam $lam --learning_rate $lr --discrete_ratio $ratio
                    # cLSTM VAR
                    run_experiment --dataset_type mixed_data --data_model VAR --method clstm --p $p --T $T --lam $lam --learning_rate $lr --discrete_ratio $ratio
                done

                # Lorenz experiments with Lorenz learning rates
                for lr in $LR_VALUES_LORENZ; do
                    for F in $F_VALUES; do
                        # cMLP Lorenz
                        run_experiment --dataset_type mixed_data --data_model Lorenz --method cmlp --p $p --T $T --F $F --lam $lam --learning_rate $lr --discrete_ratio $ratio
                        # cLSTM Lorenz
                        run_experiment --dataset_type mixed_data --data_model Lorenz --method clstm --p $p --T $T --F $F --lam $lam --learning_rate $lr --discrete_ratio $ratio
                    done
                done
            done
        done
    done
done

echo "=== 11. nonstationary dataset ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for lam in $LAM_VALUES; do
            # VAR experiments with VAR learning rates
            for lr in $LR_VALUES_VAR; do
                # cMLP VAR
                run_experiment --dataset_type nonstationary --data_model VAR --method cmlp \
                    --p $p --T $T --lam $lam --learning_rate $lr \
                    --noise_std $NOISE_STD_VAR --mean_log_sigma $MEAN_LOG_SIGMA_VAR
                # cLSTM VAR
                run_experiment --dataset_type nonstationary --data_model VAR --method clstm \
                    --p $p --T $T --lam $lam --learning_rate $lr \
                    --noise_std $NOISE_STD_VAR --mean_log_sigma $MEAN_LOG_SIGMA_VAR
            done

            # Lorenz experiments with Lorenz learning rates
            for lr in $LR_VALUES_LORENZ; do
                # cMLP Lorenz F=10
                run_experiment --dataset_type nonstationary --data_model Lorenz --method cmlp \
                    --p $p --T $T --F 10 --lam $lam --learning_rate $lr \
                    --noise_std $NOISE_STD_LORENZ_F10 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F10
                # cLSTM Lorenz F=10
                run_experiment --dataset_type nonstationary --data_model Lorenz --method clstm \
                    --p $p --T $T --F 10 --lam $lam --learning_rate $lr \
                    --noise_std $NOISE_STD_LORENZ_F10 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F10

                # cMLP Lorenz F=40
                run_experiment --dataset_type nonstationary --data_model Lorenz --method cmlp \
                    --p $p --T $T --F 40 --lam $lam --learning_rate $lr \
                    --noise_std $NOISE_STD_LORENZ_F40 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F40
                # cLSTM Lorenz F=40
                run_experiment --dataset_type nonstationary --data_model Lorenz --method clstm \
                    --p $p --T $T --F 40 --lam $lam --learning_rate $lr \
                    --noise_std $NOISE_STD_LORENZ_F40 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F40
            done
        done
    done
done


echo ""
echo "All NGC experiments completed!"
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