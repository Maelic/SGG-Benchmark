### PE-NET

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --use-wandb --task sgdet --config-file "configs/VG150/e2e_relation_yolov8m.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/VG150/SGDET/penet-yolov8m

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --use-wandb --task sgdet --config-file "configs/VG150/demo_motifs_faster_rcnn.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/VG150/SGDET/penet-faster_rcnn
