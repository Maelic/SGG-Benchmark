
# Sample detection per image from 10 to 200, with a step of 5
for i in $(seq 5 5 100)
do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --task sgdet --config-file "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/PSG/SGDET/M-PE-NET-yolov8m/config.yml" MODEL.ROI_HEADS.DETECTIONS_PER_IMG $i TEST.TOP_K $i TEST.ALLOW_LOAD_FROM_CACHE False TEST.INFORMATIVE False DATASETS.TO_TEST "test" TEST.IMS_PER_BATCH 1
done

# # CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --task sgdet --config-file "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/IndoorVG4/SGDET2/penet-yolov10l_final_glove/config.yml" MODEL.ROI_HEADS.DETECTIONS_PER_IMG 80 TEST.ALLOW_LOAD_FROM_CACHE False TEST.INFORMATIVE True DATASETS.TO_TEST "val" TEST.IMS_PER_BATCH 1

# CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --task sgdet --config-file "checkpoints/IndoorVG4/yolo_variants/penet_simple-yolov8-N/config.yml" MODEL.ROI_HEADS.DETECTIONS_PER_IMG 40 TEST.ALLOW_LOAD_FROM_CACHE True TEST.INFORMATIVE True DATASETS.TO_TEST "test" TEST.IMS_PER_BATCH 1

# CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --task sgdet --config-file "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/IndoorVG4/TO_TEST/penet-yolov8l-no_union/config.yml" MODEL.ROI_HEADS.DETECTIONS_PER_IMG 80 TEST.ALLOW_LOAD_FROM_CACHE False TEST.INFORMATIVE True DATASETS.TO_TEST "test" TEST.IMS_PER_BATCH 1 MODEL.ROI_RELATION_HEAD.LOGIT_ADJUSTMENT True

# CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --task sgdet --config-file "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/IndoorVG4/Expe_union/penet-yolov8l-spatial/config.yml" MODEL.ROI_HEADS.DETECTIONS_PER_IMG 80 TEST.ALLOW_LOAD_FROM_CACHE False TEST.INFORMATIVE True DATASETS.TO_TEST "test" TEST.IMS_PER_BATCH 1 MODEL.ROI_RELATION_HEAD.LOGIT_ADJUSTMENT False