### PE-NET


CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --task sgdet --config-file "checkpoints/VG150/SGDET/penet-yolov8m/config.yml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR checkpoints/VG150/SGDET/penet-yolov8m

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --use-wandb --task sgdet --config-file "configs/VG150/e2e_relation_yolov8m.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/VG150/SGDET/penet-yolov8m_clip

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --use-wandb --task sgdet --config-file "configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/VG150/SGDET/penet-faster_rcnn_clip

CUDA_VISIBLE_DEVICES=0 python tools/hyper_param_tuning.py --save-best --task sgdet --config-file "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/configs/VG150/react_yolov8m.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/VG150/SGDET/react-yolov8m