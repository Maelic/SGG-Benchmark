MODEL_NAME='baseline-vg/'
CHCEKPOINTS_PATH='YOLOSGG/'
DATASETS='VG'
NUM_GPUS=1

if [ $DATASETS = "VG" ]; then
    PRETRAINED_DETECTOR_CKPT='/media/zhuxuhan/7226-9559/YoloSGG/react_weights/best_model_epoch_9.pth'
    CONFIG_FILE="configs/VG150/react_yolov8m.yaml"
fi

mkdir "$CHCEKPOINTS_PATH"
mkdir "$CHCEKPOINTS_PATH"/${MODEL_NAME}/
python tools/relation_test_net.py --task sgdet \
--config-file ${CONFIG_FILE} --amp \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
TEST.IMS_PER_BATCH ${NUM_GPUS} \
GLOVE_DIR /home/zhuxuhan/projects/code/Paper_Codes/yolo-sgg-9-v9/datasets/vg/glove \
MODEL.WEIGHT ${PRETRAINED_DETECTOR_CKPT} \
OUTPUT_DIR "$CHCEKPOINTS_PATH"/${MODEL_NAME}/ \
SOLVER.PRE_VAL False \
TEST.IMS_PER_BATCH 8