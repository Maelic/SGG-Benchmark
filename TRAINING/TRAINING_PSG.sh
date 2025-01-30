# Faster-RCNN

CUDA_VISIBLE_DEVICES=0 python tools/detector_pretest_net.py --config-file "checkpoints/PSG/BACKBONE/faster_rcnn/config.yml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16"  SOLVER.MAX_ITER 50000 SOLVER.STEPS "(30000, 45000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.RELATION_ON False SOLVER.PRE_VAL False OUTPUT_DIR ./checkpoints/PSG/BACKBONE/faster_rcnn

### PE-NET

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/PSG/e2e_relation_yolov8m.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/PSG/SGDET/penet-yolov8m

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task predcls --config-file "configs/PSG/e2e_relation_yolov8m.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/PSG/PREDCLS/penet-yolov8m-2

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task predcls --config-file "configs/PSG/faster_rcnn.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/VG150/PREDCLS/penet-faster_rcnn

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/PSG/react_yolov8m.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/PSG/SGDET/react-yolov8m