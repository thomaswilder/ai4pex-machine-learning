#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name=cnn_train_partial
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=03:00:00
#SBATCH --mem=20G
#SBATCH --account=ai4pex
#SBATCH --qos=standard

export PATH="/home/users/twilder/Python/AI4PEX/tensorflow_env/bin:$PATH"

MODE="train"

domain="SO_JET"

data_dir="/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features/{domain}/coarsened_data/"
model_dir="/home/users/twilder/Python/AI4PEX/machine_learning/training/"
model_filename="cnn_20251118-175732_fold5.keras" # used for pickup and evaluate
# model_save_filename="cnn_20251107-164823_extend_to_300_epochs.keras"

data_filenames="MINT_1d_0061-0072_sa_c_{domain}.nc \
    MINT_1d_0061-0072_vor_cg_{domain}_mod.nc \
    MINT_1d_0061-0072_eke_c_{domain}_shifted.nc \
    MINT_1d_0061-0072_eke_c_{domain}.nc \
    MINT_1d_0061-0072_mke_c_{domain}.nc \
    mesh_mask_exp4_{domain}_xnemo.nc"

features="mke vor sa eke_shift" 
target="eke"
filters="128 64 32 16 8 1"
kernels="(5,5) (5,5) (3,3) (3,3) (3,3) (3,3)"
padding="(2,2) (2,2) (1,1) (1,1) (1,1) (1,1)"
batch_size="15" 
epochs="100"
learning_rate="0.001"
dropout_rate="0.2" # 0.2
kfold="1"

#! not needed
# domain slice
x_c_slice="2:37"
y_c_slice="10:45"
t_slice="300:350"

tensboard_log_dir="./logs/fit/"
verbose_output_filename="./logs/verbose_output.log"

if [ "$MODE" == "train" ]; then

    python tep_cnn.py \
        --data_dir "$data_dir" \
        --model_dir "$model_dir" \
        --data_filenames "$data_filenames" \
        --domain "$domain" \
        --x_c_slice "$x_c_slice" \
        --y_c_slice "$y_c_slice" \
        --t_slice "$t_slice" \
        --features "$features" \
        --target "$target" \
        --epochs "$epochs" \
        --batch_size "$batch_size" \
        --k_fold "$kfold" \
        --filters "$filters" \
        --kernels "$kernels" \
        --padding "$padding" \
        --train \
        --local_norm \
        --verbose \
        --verbose_output_filename "$verbose_output_filename" \
        --learning_rate "$learning_rate" \
        --dropout_rate "$dropout_rate"
        # --tensboard_log_dir "$tensboard_log_dir" 
        # --model_filename "$model_filename" \
        # --model_save_filename "$model_save_filename" \
        # --pickup
        # --use_learning_rate_scheduler
         

elif [ "$MODE" == "evaluate" ]; then

    python tep_cnn.py \
        --data_dir "$data_dir" \
        --model_dir "$model_dir" \
        --model_filename "$model_filename" \
        --data_filenames "$data_filenames" \
        --domain "$domain" \
        --features "$features" \
        --target "$target" \
        --batch_size "$batch_size" \
        --filters "$filters" \
        --kernels "$kernels" \
        --padding "$padding" \
        --global_norm \
        --evaluate \
        --verbose \
        --tensboard_log_dir "$tensboard_log_dir" \
        --verbose_output_filename "$verbose_output_filename"

fi