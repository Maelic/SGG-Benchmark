CUDA_VISIBLE_DEVICES=0 python tools/detector_pretest_net.py --config-file "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/IndoorVG4/Backbones/faster_rcnn2/config.yml" SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 1 DTYPE "float16" OUTPUT_DIR /home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/IndoorVG4/Backbones/faster_rcnn2

### SGDET - INDOORVG4 - YOLOV8m

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --save-best --task sgdet --config-file "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/PSG/SGDET/transformer-yolov8m/config.yml" GLOVE_DIR /home/maelic/glove

# Causal Motifs TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/causal-motifs-yolov8m

# Transformer
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --task sgdet --save-best --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_EPOCH 50 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/transformer-yolov8l

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --task sgdet --config-file "/home/maelic/ros2_humble/src/Robots-Scene-Understanding/rsu_scene_graph_generation/models/transformer" GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/ros2_humble/src/Robots-Scene-Understanding/rsu_scene_graph_generation/models/transformer

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --task sgdet --save-best --config-file "configs/IndoorVG/e2e_relation_faster_rcnn.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_EPOCH 50 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/transformer-faster_rcnn

# Causal Vctree TDE
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER vctree SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/causal-vctree-yolov8m

# GPS-Net
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR GPSNetPredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/gpsnet-yolov8l

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR GPSNetPredictor SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/gpsnet-yolov8l


# PENET
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov8.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 16 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR checkpoints/IndoorVG4/SGDET2/penet-yolov8m

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov8world.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 16 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR /home/maelic/ros2_humble/src/Robots-Scene-Understanding/rsu_scene_graph_generation/models/penet-yolov8x_world_clip

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov8world.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 16 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/penet_yolov8x_world_visual

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --use-wandb --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov9.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/penet-yolov9c

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --save-best  --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov9.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/penet-yolov9c

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_faster_rcnn.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/penet-faster_rcnn

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_faster_rcnn.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/penet-faster_rcnn MODEL.BACKBONE.NMS_THRESH 0.001 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 30

# Sparse-RCNN

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --save-best --task sgdet --config-file "configs/IndoorVG/e2e_relation_sparse_rcnn.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/penet-faster_rcnn

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --save-best --task sgdet --config-file "configs/simrel_e2e_relation_X_101_32_8_FPN_1x.yaml"  MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH 2 SOLVER.VAL_PERIOD 12000 SOLVER.MAX_ITER 80000 SOLVER.STEPS '(47000, 64000)' SOLVER.BASE_LR 0.000008 SOLVER.OPTIMIZER "ADAMW" MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR Outputs/real025kl_newe2rposition_fullobjbranch_800q_smalldyconv_prequeryaug9_norel_hardlabel MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM 1024 MODEL.SimrelRCNN.ENABLE_BG_OBJ False MODEL.SimrelRCNN.ENABLE_REL_X2Y True MODEL.SimrelRCNN.REL_DIM 256 MODEL.SimrelRCNN.NUM_PROPOSALS 800 MODEL.SimrelRCNN.NUM_HEADS 6 MODEL.SimrelRCNN.REL_STACK_NUM 6 MODEL.SimrelRCNN.TRIPLET_MASK_WEIGHT 1.0 MODEL.SimrelRCNN.FREEZE_BACKBONE True MODEL.SimrelRCNN.CROSS_OBJ_FEAT_FUSION False MODEL.SimrelRCNN.CLASS_WEIGHT 1.333 MODEL.SimrelRCNN.L1_WEIGHT 5.0 MODEL.SimrelRCNN.GIOU_WEIGHT 2.0 GLOVE_DIR '' MODEL.SimrelRCNN.ENABLE_FREQ False MODEL.SimrelRCNN.ENABLE_QUERY_REVERSE False MODEL.SimrelRCNN.USE_REFINE_OBJ_FEATURE True MODEL.SimrelRCNN.FREEZE_PUREE_OBJDET False MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS 100 MODEL.SimrelRCNN.ENABLE_MASK_BRANCH False MODEL.SimrelRCNN.KL_BRANCH_WEIGHT 0.25 MODEL.SimrelRCNN.ENABLE_KL_BRANCH True MODEL.SimrelRCNN.POSI_ENCODE_DIM 64 MODEL.SimrelRCNN.REL_CLASS_WEIGHT 1.334 MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT 5.0 MODEL.SimrelRCNN.AUXILIARY_BRANCH True MODEL.SimrelRCNN.AUXILIARY_BRANCH_SELECT_ENT_MAX_NUM 25 MODEL.SimrelRCNN.AUXILIARY_BRANCH_START 11 MODEL.SimrelRCNN.ENABLE_ENT_PROP True MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False MODEL.SimrelRCNN.USE_CROSS_RANK False MODEL.SimrelRCNN.DISABLE_KQ_FUSION_SELFATTEN False MODEL.SimrelRCNN.DIM_ENT_PRE_CLS 256 MODEL.SimrelRCNN.DIM_ENT_PRE_REG 256 MODEL.SimrelRCNN.ONE_REL_CONV True MODEL.SimrelRCNN.ENABLE_BATCH_REDUCTION True MODEL.SimrelRCNN.USE_HARD_LABEL_KLMATCH True MODEL.SimrelRCNN.DISABLE_OBJ2REL_LOSS True


# TEST

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --save-best --task sgdet --config-file "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/IndoorVG4/SGDET/transformer-faster_rcnn/config.yml" TEST.IMS_PER_BATCH 8 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove