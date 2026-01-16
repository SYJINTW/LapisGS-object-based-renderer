#!/bin/bash
# This script runs a pipeline of warmup, training, rendering, and metrics for each experiment.
# It does not exit on the first error, but continues to the next experiment.
# set -e

# [NOTE] copy and modify the config for your own experiments

export CUDA_VISIBLE_DEVICES=0

# ======= Config ======
SCENE_NAME="lego" #! Change to your scene name ("lego", "materials")

TRAINING_DATA_BASE_DIR="/home/syjintw/Desktop/NUS/dataset" #! Change to your dataset path
OUTPUT_DATA_BASE_DIR="/home/syjintw/Desktop/NUS/lapisgs-output" #! Change to your dataset path
OUR_CAM_BASE_DIR="/home/syjintw/Desktop/NUS/tmp" #! Change to your dataset path

GT_DIR="${OUTPUT_DATA_BASE_DIR}/${SCENE_NAME}/opacity/${SCENE_NAME}_res1/ours_uniform_1x/ours_30000/renders"

EXP_NAME="uniform_distortion"

GS_RES_LIST=("1" "2" "4" "8")
CAM_DISTANCE_LIST=("1")

for GS_RES in "${GS_RES_LIST[@]}"; do
    for CAM_DISTANCE in "${CAM_DISTANCE_LIST[@]}"; do
        if (( GS_RES < CAM_DISTANCE )); then
            RENDER_FACTOR=1
        else
            # Bash 的除法本身就是 int()，會自動捨去小數
            RENDER_FACTOR=$(( GS_RES / CAM_DISTANCE ))
        fi
        echo "GS_RES: $GS_RES | Dist: $CAM_DISTANCE | Factor: $RENDER_FACTOR"

        TRAINING_DIR="${TRAINING_DATA_BASE_DIR}/${SCENE_NAME}"
        OUTPUT_DIR="${OUTPUT_DATA_BASE_DIR}/${SCENE_NAME}/opacity/${SCENE_NAME}_res${GS_RES}"
        RESULTS_NAME="ours_uniform_${CAM_DISTANCE}x"
        OUR_CAM_PATH="${OUR_CAM_BASE_DIR}/${SCENE_NAME}/uniform_${CAM_DISTANCE}x/transforms.json"

        # python render-lapisgs_exp.py \
        #     -m "${OUTPUT_DIR}" \
        #     -s "${TRAINING_DIR}" \
        #     --gs_res_list "${RENDER_FACTOR}" \
        #     --skip_train --skip_test \
        #     --our_cam_name "${RESULTS_NAME}" \
        #     --our_cam_path "${OUR_CAM_PATH}" \
        #     --bg_color 0.0 0.0 0.0

        # python ./metrics-lapisgs-exp.py \
        #     -m "${OUTPUT_DIR}" \
        #     --cam_name "${RESULTS_NAME}" \
        #     --gt_dir "${GT_DIR}" \
        #     --mask_dir "${OUR_CAM_BASE_DIR}/${SCENE_NAME}/uniform_${CAM_DISTANCE}x/mask" \
        #     --output_dir "${OUTPUT_DIR}/results_${EXP_NAME}"
    done
done