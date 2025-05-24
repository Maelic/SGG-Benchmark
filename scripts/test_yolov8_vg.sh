MODEL_NAME='baseline-vg/'
CHCEKPOINTS_PATH='YOLOSGG/'
DATASETS='VG'
NUM_GPUS=1

if [ $DATASETS = "VG" ]; then
    PRETRAINED_DETECTOR_CKPT='/userhome/zxh/YoloSGG/react_motifs/best_model_epoch_9.pth'
    CONFIG_FILE="configs/VG150/react_yolov8m.yaml"
fi

mkdir "$CHCEKPOINTS_PATH"
mkdir "$CHCEKPOINTS_PATH"/${MODEL_NAME}/
CUDA_VISIBLE_DEVICES=7 python tools/relation_test_net.py --task sgdet \
--config-file ${CONFIG_FILE} --amp \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
TEST.IMS_PER_BATCH ${NUM_GPUS} \
GLOVE_DIR /userhome/wsss_-knowledge/datasets/vg/glove \
MODEL.WEIGHT ${PRETRAINED_DETECTOR_CKPT} \
OUTPUT_DIR "$CHCEKPOINTS_PATH"/${MODEL_NAME}/ \
SOLVER.PRE_VAL False \
TEST.IMS_PER_BATCH 8