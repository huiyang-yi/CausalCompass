#!/bin/bash

# chmod +x run_varlingam.sh
# ./run_varlingam.sh

echo "Starting VARLiNGAM experiments..."

#todo:
P_VALUES="10"                      #"10 20"
T_VALUES="500 1000"                #"250 500 1000"
F_VALUES="10 40"                        #"10 40"
SEEDS="0,1,2,3,4"                         #"0,1,2,3,4"

TAU_MAX_VALUES="3 5"                      #"3 5"
VARLINGAM_ALPHA_VALUES="0 0.01 0.05 0.1 0.3"     #"0 0.01 0.05 0.1 0.3"

NORMALIZATION_METHODS="zscore minmax"
RHO_VALUES="0.5"
GAMMA_VALUES="1.2"
MISSING_PROB_VALUES="0.4"
DISCRETE_RATIO_VALUES="0.5"
GAUSSIAN_RATIO_VALUES="0.5"
# Nonstationary specific parameters
NOISE_STD_VAR="1.0"
MEAN_LOG_SIGMA_VAR="1.0"
#MEAN_LOG_SIGMA_VAR="2.7"
NOISE_STD_LORENZ_F10="2.0"
MEAN_LOG_SIGMA_LORENZ_F10="2.5"
NOISE_STD_LORENZ_F40="2.0"
MEAN_LOG_SIGMA_LORENZ_F40="3.5"

FAILED_COUNT=0
TOTAL_COUNT=0

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

echo "=== 1. vanilla ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for tau in $TAU_MAX_VALUES; do
            for alpha in $VARLINGAM_ALPHA_VALUES; do
                # VAR
                run_experiment --dataset_type vanilla --data_model VAR --method varlingam --p $p --T $T --tau_max $tau --varlingamalpha $alpha
                # Lorenz
                for F in $F_VALUES; do
                    run_experiment --dataset_type vanilla --data_model Lorenz --method varlingam --p $p --T $T --F $F --tau_max $tau --varlingamalpha $alpha
                done
            done
        done
    done
done

echo "=== 2. standardized ==="
for norm in $NORMALIZATION_METHODS; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for tau in $TAU_MAX_VALUES; do
                for alpha in $VARLINGAM_ALPHA_VALUES; do
                    # VAR
                    run_experiment --dataset_type standardized --data_model VAR --method varlingam --p $p --T $T --tau_max $tau --varlingamalpha $alpha --normalization_method $norm
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type standardized --data_model Lorenz --method varlingam --p $p --T $T --F $F --tau_max $tau --varlingamalpha $alpha --normalization_method $norm
                    done
                done
            done
        done
    done
done

echo "=== 3. trendseason ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for tau in $TAU_MAX_VALUES; do
            for alpha in $VARLINGAM_ALPHA_VALUES; do
                # VAR
                run_experiment --dataset_type trendseason --data_model VAR --method varlingam --p $p --T $T --tau_max $tau --varlingamalpha $alpha
                # Lorenz
                for F in $F_VALUES; do
                    run_experiment --dataset_type trendseason --data_model Lorenz --method varlingam --p $p --T $T --F $F --tau_max $tau --varlingamalpha $alpha
                done
            done
        done
    done
done

echo "=== 4. confounder ==="
for rho in $RHO_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for tau in $TAU_MAX_VALUES; do
                for alpha in $VARLINGAM_ALPHA_VALUES; do
                    # VAR
                    run_experiment --dataset_type confounder --data_model VAR --method varlingam --p $p --T $T --tau_max $tau --varlingamalpha $alpha --rho $rho
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type confounder --data_model Lorenz --method varlingam --p $p --T $T --F $F --tau_max $tau --varlingamalpha $alpha --rho $rho
                    done
                done
            done
        done
    done
done

echo "=== 6. measurement_error ==="
for gamma in $GAMMA_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for tau in $TAU_MAX_VALUES; do
                for alpha in $VARLINGAM_ALPHA_VALUES; do
                    # VAR
                    run_experiment --dataset_type measurement_error --data_model VAR --method varlingam --p $p --T $T --tau_max $tau --varlingamalpha $alpha --gamma $gamma
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type measurement_error --data_model Lorenz --method varlingam --p $p --T $T --F $F --tau_max $tau --varlingamalpha $alpha --gamma $gamma
                    done
                done
            done
        done
    done
done


echo "=== 8. missing ==="
for prob in $MISSING_PROB_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for tau in $TAU_MAX_VALUES; do
                for alpha in $VARLINGAM_ALPHA_VALUES; do
                    # VAR
                    run_experiment --dataset_type missing --data_model VAR --method varlingam --p $p --T $T --tau_max $tau --varlingamalpha $alpha --missing_prob $prob
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type missing --data_model Lorenz --method varlingam --p $p --T $T --F $F --tau_max $tau --varlingamalpha $alpha --missing_prob $prob
                    done
                done
            done
        done
    done
done

echo "=== 9. mixed_data ==="
for ratio in $DISCRETE_RATIO_VALUES; do
    for p in $P_VALUES; do
        for T in $T_VALUES; do
            for tau in $TAU_MAX_VALUES; do
                for alpha in $VARLINGAM_ALPHA_VALUES; do
                    # VAR
                    run_experiment --dataset_type mixed_data --data_model VAR --method varlingam --p $p --T $T --tau_max $tau --varlingamalpha $alpha --discrete_ratio $ratio
                    # Lorenz
                    for F in $F_VALUES; do
                        run_experiment --dataset_type mixed_data --data_model Lorenz --method varlingam --p $p --T $T --F $F --tau_max $tau --varlingamalpha $alpha --discrete_ratio $ratio
                    done
                done
            done
        done
    done
done


echo "=== 11. nonstationary ==="
for p in $P_VALUES; do
    for T in $T_VALUES; do
        for tau in $TAU_MAX_VALUES; do
            for alpha in $VARLINGAM_ALPHA_VALUES; do
                # VAR
                run_experiment --dataset_type nonstationary --data_model VAR --method varlingam \
                    --p $p --T $T --tau_max $tau --varlingamalpha $alpha \
                    --noise_std $NOISE_STD_VAR --mean_log_sigma $MEAN_LOG_SIGMA_VAR

                # Lorenz F=10
                run_experiment --dataset_type nonstationary --data_model Lorenz --method varlingam \
                    --p $p --T $T --F 10 --tau_max $tau --varlingamalpha $alpha \
                    --noise_std $NOISE_STD_LORENZ_F10 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F10

                # Lorenz F=40
                run_experiment --dataset_type nonstationary --data_model Lorenz --method varlingam \
                    --p $p --T $T --F 40 --tau_max $tau --varlingamalpha $alpha \
                    --noise_std $NOISE_STD_LORENZ_F40 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F40
            done
        done
    done
done


echo ""
echo "All VARLiNGAM experiments completed!"
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