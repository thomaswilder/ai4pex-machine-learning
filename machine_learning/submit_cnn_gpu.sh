#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --job-name=gpu_test
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

export PATH="/home/users/twilder/Python/AI4PEX/tensorflow_env/bin:$PATH"
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}

echo "=== GPU snapshot ==="
nvidia-smi

# Log GPU utilization every second while python runs
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv -l 1 > logs/gpu_util_${SLURM_JOB_ID}.txt &
SMI_PID=$!

MODE="train"

domain="SO_JET"

data_dir="/gws/nopw/j04/ai4pex/twilder/NEMO_data/DINO/EXP16/features_take2/{domain}/coarsened_data/"
model_dir="/home/users/twilder/Python/AI4PEX/machine_learning/training/"
model_filename="cnn_20251118-175732_fold5.keras" # used for pickup and evaluate
# model_save_filename="cnn_20251107-164823_extend_to_300_epochs.keras"

data_filenames="MINT_1d_0061-0072_sa_cg_{domain}.nc \
    MINT_1d_0061-0072_vor_cg_{domain}_mod.nc \
    MINT_1d_0061-0072_ke_cg_{domain}_with_shifted.nc \
    mesh_mask_exp4_{domain}_xnemo.nc"

features="coarse_ke vor sa" 
target="fine_ke"
filters="128 64 32 32 32 16 8 1"
kernels="(5,5) (5,5) (3,3) (3,3) (3,3) (3,3) (3,3) (1,1)"
padding="(2,2) (2,2) (1,1) (1,1) (1,1) (1,1) (1,1) (0,0)"
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
verbose_output_filename="./logs/verbose_output_gpu.log"

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
        --early_stopping \
        --verbose_output_filename "$verbose_output_filename" \
        --learning_rate "$learning_rate" \
        --dropout_rate "$dropout_rate"
        # --tensboard_log_dir "$tensboard_log_dir" 
        # --model_filename "$model_filename" \
        # --model_save_filename "$model_save_filename" \
        # --pickup
        # --use_learning_rate_scheduler
        #         
         

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

# Stop logging GPU utilization
kill $SMI_PID