OUTPUT_DIR='/media/zhuxuhan/Elements/YoloSGG/yolov8_vg'
NUM_GPUS=1

mkdir ${OUTPUT_DIR}

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10021 --nproc_per_node=${NUM_GPUS} tools/relation_train_net.py \
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py \
--save-best --task sgdet --config-file "configs/VG150/react_yolov8m.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor \
SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 DTYPE "float16" SOLVER.PRE_VAL False \
GLOVE_DIR /home/zhuxuhan/projects/code/Paper_Codes/yolo-sgg-9-v9/datasets/vg/glove \
OUTPUT_DIR ${OUTPUT_DIR} \
MODEL.PRETRAINED_DETECTOR_CKPT '/media/zhuxuhan/7226-9559/YoloSGG/yolov8m_vg150.pt' \
MODEL.ROI_RELATION_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN True