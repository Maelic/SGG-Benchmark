CUDA_VISIBLE_DEVICES=0 python tools/detector_pretest_net.py --config-file "./checkpoints/IndoorVG4/Backbones/faster_rcnn2/config.yml" SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 1 DTYPE "float16" OUTPUT_DIR ./checkpoints/IndoorVG4/Backbones/faster_rcnn2

### SGDET - INDOORVG4 - YOLOV8m

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --save-best --task sgdet --config-file "./checkpoints/PSG/SGDET/transformer-yolov8m/config.yml" GLOVE_DIR /home/maelic/glove

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

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --save-best  --task sgdet --config-file "configs/IndoorVG/e2e_relation_yolov12.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.PRE_VAL False GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG/SGDET/react_yolov12m