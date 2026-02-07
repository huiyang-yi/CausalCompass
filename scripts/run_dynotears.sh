#!/bin/bash


# chmod +x run_dynotears.sh
# ./run_dynotears.sh

echo "Starting DYNOTEARS experiments..."

#todo:
P_VALUES="15"                      #"10"
T_VALUES="500 1000"                #"500 1000"
F_VALUES="10 40"                        #"10 40"
SEEDS="0,1,2,3,4"                         #"0,1,2,3,4"

TAU_MAX_VALUES="3 5"                      #"3 5"
WTHRE_VALUES="0.01 0.05 0.1 0.3"           #"0 0.01 0.05 0.1 0.3"
LAMBDA_A_VALUES="0.001 0.01 0.1"          #"0.001 0.01 0.1"
LAMBDA_W_VALUES="0.001 0.005 0.01"    #"0.001 0.005 0.01"

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
            for wthre in $WTHRE_VALUES; do
                for lambda_a in $LAMBDA_A_VALUES; do
                    for lambda_w in $LAMBDA_W_VALUES; do
                        # VAR
                        run_experiment --dataset_type vanilla --data_model VAR --method dynotears --p $p --T $T --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w
                        # Lorenz
                        for F in $F_VALUES; do
                            run_experiment --dataset_type vanilla --data_model Lorenz --method dynotears --p $p --T $T --F $F --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w
                        done
                    done
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
                for wthre in $WTHRE_VALUES; do
                    for lambda_a in $LAMBDA_A_VALUES; do
                        for lambda_w in $LAMBDA_W_VALUES; do
                            # VAR
                            run_experiment --dataset_type standardized --data_model VAR --method dynotears --p $p --T $T --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w --normalization_method $norm
                            # Lorenz
                            for F in $F_VALUES; do
                                run_experiment --dataset_type standardized --data_model Lorenz --method dynotears --p $p --T $T --F $F --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w --normalization_method $norm
                            done
                        done
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
            for wthre in $WTHRE_VALUES; do
                for lambda_a in $LAMBDA_A_VALUES; do
                    for lambda_w in $LAMBDA_W_VALUES; do
                        # VAR
                        run_experiment --dataset_type trendseason --data_model VAR --method dynotears --p $p --T $T --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w
                        # Lorenz
                        for F in $F_VALUES; do
                            run_experiment --dataset_type trendseason --data_model Lorenz --method dynotears --p $p --T $T --F $F --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w
                        done
                    done
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
                for wthre in $WTHRE_VALUES; do
                    for lambda_a in $LAMBDA_A_VALUES; do
                        for lambda_w in $LAMBDA_W_VALUES; do
                            # VAR
                            run_experiment --dataset_type confounder --data_model VAR --method dynotears --p $p --T $T --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w --rho $rho
                            # Lorenz
                            for F in $F_VALUES; do
                                run_experiment --dataset_type confounder --data_model Lorenz --method dynotears --p $p --T $T --F $F --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w --rho $rho
                            done
                        done
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
                for wthre in $WTHRE_VALUES; do
                    for lambda_a in $LAMBDA_A_VALUES; do
                        for lambda_w in $LAMBDA_W_VALUES; do
                            # VAR
                            run_experiment --dataset_type measurement_error --data_model VAR --method dynotears --p $p --T $T --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w --gamma $gamma
                            # Lorenz
                            for F in $F_VALUES; do
                                run_experiment --dataset_type measurement_error --data_model Lorenz --method dynotears --p $p --T $T --F $F --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w --gamma $gamma
                            done
                        done
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
                for wthre in $WTHRE_VALUES; do
                    for lambda_a in $LAMBDA_A_VALUES; do
                        for lambda_w in $LAMBDA_W_VALUES; do
                            # VAR
                            run_experiment --dataset_type missing --data_model VAR --method dynotears --p $p --T $T --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w --missing_prob $prob
                            # Lorenz
                            for F in $F_VALUES; do
                                run_experiment --dataset_type missing --data_model Lorenz --method dynotears --p $p --T $T --F $F --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w --missing_prob $prob
                            done
                        done
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
                for wthre in $WTHRE_VALUES; do
                    for lambda_a in $LAMBDA_A_VALUES; do
                        for lambda_w in $LAMBDA_W_VALUES; do
                            # VAR
                            run_experiment --dataset_type mixed_data --data_model VAR --method dynotears --p $p --T $T --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w --discrete_ratio $ratio
                            # Lorenz
                            for F in $F_VALUES; do
                                run_experiment --dataset_type mixed_data --data_model Lorenz --method dynotears --p $p --T $T --F $F --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w --discrete_ratio $ratio
                            done
                        done
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
            for wthre in $WTHRE_VALUES; do
                for lambda_a in $LAMBDA_A_VALUES; do
                    for lambda_w in $LAMBDA_W_VALUES; do
                        # VAR
                        run_experiment --dataset_type nonstationary --data_model VAR --method dynotears \
                            --p $p --T $T --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w \
                            --noise_std $NOISE_STD_VAR --mean_log_sigma $MEAN_LOG_SIGMA_VAR

                        # Lorenz F=10
                        run_experiment --dataset_type nonstationary --data_model Lorenz --method dynotears \
                            --p $p --T $T --F 10 --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w \
                            --noise_std $NOISE_STD_LORENZ_F10 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F10

                        # Lorenz F=40
                        run_experiment --dataset_type nonstationary --data_model Lorenz --method dynotears \
                            --p $p --T $T --F 40 --tau_max $tau --wthre $wthre --lambda_a $lambda_a --lambda_w $lambda_w \
                            --noise_std $NOISE_STD_LORENZ_F40 --mean_log_sigma $MEAN_LOG_SIGMA_LORENZ_F40
                    done
                done
            done
        done
    done
done

echo ""
echo "All DYNOTEARS experiments completed!"
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
