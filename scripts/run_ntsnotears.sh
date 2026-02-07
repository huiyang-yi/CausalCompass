#!/bin/bash

# NTS-NOTEARS experiment script
# Simple syntax, easy to understand and extend
# chmod +x run_ntsnotears.sh
# ./run_ntsnotears.sh

echo "Starting NTS-NOTEARS experiments..."

# Parameters
DEVICE="cuda:0"                       # "cuda:1"
P_VALUES="15"                      #"10 20"
T_VALUES="500 1000"                #"250 500 1000"
F_VALUES="10 40"                   #"10 40"
SEEDS="0,1,2,3,4"                  #"0,1,2,3,4"

# VAR hyperparameters
TAU_MAX_VAR="3"                     #"1 2 3 4 5"
WTHRE_VAR="0.01 0.05 0.1 0.3 0.5"
LAMBDA_1_VAR="0.001 0.01 0.1"      # Single values for VAR
LAMBDA_2_VAR="0.001 0.005 0.01"

# Lorenz hyperparameters (following paper's configuration)
TAU_MAX_LORENZ="1"
WTHRE_LORENZ="0.1 0.3 0.5"
LAMBDA_1_LORENZ="0.001,0.1"        # List format for Lorenz
LAMBDA_2_LORENZ="0.01"

# ===== Data parameter configuration area =====
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
# ===== Data parameter configuration area end =====

FAILED_COUNT=0
TOTAL_COUNT=0

run_experiment() {
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo "[$TOTAL_COUNT] Running: $*"

    python run.py "$@" --seeds $SEEDS --device $DEVICE
    if [ $? -ne 0 ]; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo "  *** FAILED ***"
        echo "FAILED: $*" >> failed_experiments.txt
    fi
}

echo "=== 1. vanilla dataset ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        # VAR experiments with VAR hyperparameters
        for tau in $TAU_MAX_VAR; do
            for wthre in $WTHRE_VAR; do
                for lambda1 in $LAMBDA_1_VAR; do
                    for lambda2 in $LAMBDA_2_VAR; do
                        run_experiment --dataset_type vanilla --data_model VAR --method ntsnotears \
                            --p $p --T $T --tau_max $tau --wthre $wthre \
                            --lambda_1 "$lambda1" --lambda_2 $lambda2
                    done
                done
            done
        done

        # Lorenz experiments with Lorenz hyperparameters
        for tau in $TAU_MAX_LORENZ; do
            for wthre in $WTHRE_LORENZ; do
                for lambda1 in $LAMBDA_1_LORENZ; do
                    for lambda2 in $LAMBDA_2_LORENZ; do
                        for F in $F_VALUES; do
                            run_experiment --dataset_type vanilla --data_model Lorenz --method ntsnotears \
                                --p $p --T $T --F $F --tau_max $tau --wthre $wthre \
                                --lambda_1 "$lambda1" --lambda_2 $lambda2
                        done
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
            # VAR experiments with VAR hyperparameters
            for tau in $TAU_MAX_VAR; do
                for wthre in $WTHRE_VAR; do
                    for lambda1 in $LAMBDA_1_VAR; do
                        for lambda2 in $LAMBDA_2_VAR; do
                            run_experiment --dataset_type standardized --data_model VAR --method ntsnotears \
                                --p $p --T $T --tau_max $tau --wthre $wthre \
                                --lambda_1 "$lambda1" --lambda_2 $lambda2 --normalization_method $norm
                        done
                    done
                done
            done

            # Lorenz experiments with Lorenz hyperparameters
            for tau in $TAU_MAX_LORENZ; do
                for wthre in $WTHRE_LORENZ; do
                    for lambda1 in $LAMBDA_1_LORENZ; do
                        for lambda2 in $LAMBDA_2_LORENZ; do
                            for F in $F_VALUES; do
                                run_experiment --dataset_type standardized --data_model Lorenz --method ntsnotears \
                                    --p $p --T $T --F $F --tau_max $tau --wthre $wthre \
                                    --lambda_1 "$lambda1" --lambda_2 $lambda2 --normalization_method $norm
                            done
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
        # Lorenz experiments with Lorenz hyperparameters
        for tau in $TAU_MAX_LORENZ; do
            for wthre in $WTHRE_LORENZ; do
                for lambda1 in $LAMBDA_1_LORENZ; do
                    for lambda2 in $LAMBDA_2_LORENZ; do
                        for F in $F_VALUES; do
                            run_experiment --dataset_type trendseason --data_model Lorenz --method ntsnotears \
                                --p $p --T $T --F $F --tau_max $tau --wthre $wthre \
                                --lambda_1 "$lambda1" --lambda_2 $lambda2
                        done
                    done
                done
            done
        done

        # VAR experiments with VAR hyperparameters
        for tau in $TAU_MAX_VAR; do
            for wthre in $WTHRE_VAR; do
                for lambda1 in $LAMBDA_1_VAR; do
                    for lambda2 in $LAMBDA_2_VAR; do
                        run_experiment --dataset_type trendseason --data_model VAR --method ntsnotears \
                            --p $p --T $T --tau_max $tau --wthre $wthre \
                            --lambda_1 "$lambda1" --lambda_2 $lambda2
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
            # VAR experiments with VAR hyperparameters
            for tau in $TAU_MAX_VAR; do
                for wthre in $WTHRE_VAR; do
                    for lambda1 in $LAMBDA_1_VAR; do
                        for lambda2 in $LAMBDA_2_VAR; do
                            run_experiment --dataset_type confounder --data_model VAR --method ntsnotears \
                                --p $p --T $T --tau_max $tau --wthre $wthre \
                                --lambda_1 "$lambda1" --lambda_2 $lambda2 --rho $rho
                        done
                    done
                done
            done

            # Lorenz experiments with Lorenz hyperparameters
            for tau in $TAU_MAX_LORENZ; do
                for wthre in $WTHRE_LORENZ; do
                    for lambda1 in $LAMBDA_1_LORENZ; do
                        for lambda2 in $LAMBDA_2_LORENZ; do
                            for F in $F_VALUES; do
                                run_experiment --dataset_type confounder --data_model Lorenz --method ntsnotears \
                                    --p $p --T $T --F $F --tau_max $tau --wthre $wthre \
                                    --lambda_1 "$lambda1" --lambda_2 $lambda2 --rho $rho
                            done
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
            # VAR experiments with VAR hyperparameters
            for tau in $TAU_MAX_VAR; do
                for wthre in $WTHRE_VAR; do
                    for lambda1 in $LAMBDA_1_VAR; do
                        for lambda2 in $LAMBDA_2_VAR; do
                            run_experiment --dataset_type measurement_error --data_model VAR --method ntsnotears \
                                --p $p --T $T --tau_max $tau --wthre $wthre \
                                --lambda_1 "$lambda1" --lambda_2 $lambda2 --gamma $gamma
                        done
                    done
                done
            done

            # Lorenz experiments with Lorenz hyperparameters
            for tau in $TAU_MAX_LORENZ; do
                for wthre in $WTHRE_LORENZ; do
                    for lambda1 in $LAMBDA_1_LORENZ; do
                        for lambda2 in $LAMBDA_2_LORENZ; do
                            for F in $F_VALUES; do
                                run_experiment --dataset_type measurement_error --data_model Lorenz --method ntsnotears \
                                    --p $p --T $T --F $F --tau_max $tau --wthre $wthre \
                                    --lambda_1 "$lambda1" --lambda_2 $lambda2 --gamma $gamma
                            done
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
            # VAR experiments with VAR hyperparameters
            for tau in $TAU_MAX_VAR; do
                for wthre in $WTHRE_VAR; do
                    for lambda1 in $LAMBDA_1_VAR; do
                        for lambda2 in $LAMBDA_2_VAR; do
                            run_experiment --dataset_type missing --data_model VAR --method ntsnotears \
                                --p $p --T $T --tau_max $tau --wthre $wthre \
                                --lambda_1 "$lambda1" --lambda_2 $lambda2 --missing_prob $prob
                        done
                    done
                done
            done

            # Lorenz experiments with Lorenz hyperparameters
            for tau in $TAU_MAX_LORENZ; do
                for wthre in $WTHRE_LORENZ; do
                    for lambda1 in $LAMBDA_1_LORENZ; do
                        for lambda2 in $LAMBDA_2_LORENZ; do
                            for F in $F_VALUES; do
                                run_experiment --dataset_type missing --data_model Lorenz --method ntsnotears \
                                    --p $p --T $T --F $F --tau_max $tau --wthre $wthre \
                                    --lambda_1 "$lambda1" --lambda_2 $lambda2 --missing_prob $prob
                            done
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
            # VAR experiments with VAR hyperparameters
            for tau in $TAU_MAX_VAR; do
                for wthre in $WTHRE_VAR; do
                    for lambda1 in $LAMBDA_1_VAR; do
                        for lambda2 in $LAMBDA_2_VAR; do
                            run_experiment --dataset_type mixed_data --data_model VAR --method ntsnotears \
                                --p $p --T $T --tau_max $tau --wthre $wthre \
                                --lambda_1 "$lambda1" --lambda_2 $lambda2 --discrete_ratio $ratio
                        done
                    done
                done
            done

            # Lorenz experiments with Lorenz hyperparameters
            for tau in $TAU_MAX_LORENZ; do
                for wthre in $WTHRE_LORENZ; do
                    for lambda1 in $LAMBDA_1_LORENZ; do
                        for lambda2 in $LAMBDA_2_LORENZ; do
                            for F in $F_VALUES; do
                                run_experiment --dataset_type mixed_data --data_model Lorenz --method ntsnotears \
                                    --p $p --T $T --F $F --tau_max $tau --wthre $wthre \
                                    --lambda_1 "$lambda1" --lambda_2 $lambda2 --discrete_ratio $ratio
                            done
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
        # VAR experiments with VAR hyperparameters
        for tau in $TAU_MAX_VAR; do
            for wthre in $WTHRE_VAR; do
                for lambda1 in $LAMBDA_1_VAR; do
                    for lambda2 in $LAMBDA_2_VAR; do
                        run_experiment --dataset_type nonstationary --data_model VAR --method ntsnotears \
                            --p $p --T $T --tau_max $tau --wthre $wthre \
                            --lambda_1 "$lambda1" --lambda_2 $lambda2 \
                            --noise_std $NOISE_STD_VAR --mean_log_sigma $MEAN_LOG_SIGMA_VAR
                    done
                done
            done
        done

        # Lorenz experiments with Lorenz hyperparameters
        for tau in $TAU_MAX_LORENZ; do
            for wthre in $WTHRE_LORENZ; do
                for lambda1 in $LAMBDA_1_LORENZ; do
                    for lambda2 in $LAMBDA_2_LORENZ; do
                        # Lorenz F=10
                        run_experiment --dataset_type nonstationary --data_model Lorenz --method ntsnotears \
                            --p $p --T $T --F 10 --tau_max $tau --wthre $wthre \
                            --lambda_1 "$lambda1" --lambda_2 $lambda2 \
                            --noise_std $NOISE_STD_LORENZ_F10 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F10

                        # Lorenz F=40
                        run_experiment --dataset_type nonstationary --data_model Lorenz --method ntsnotears \
                            --p $p --T $T --F 40 --tau_max $tau --wthre $wthre \
                            --lambda_1 "$lambda1" --lambda_2 $lambda2 \
                            --noise_std $NOISE_STD_LORENZ_F40 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F40
                    done
                done
            done
        done
    done
done



echo ""
echo "All NTS-NOTEARS experiments completed!"
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