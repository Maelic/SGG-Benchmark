
# Sample detection per image from 10 to 200, with a step of 5
for i in $(seq 5 5 100)
do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --task sgdet --config-file "./checkpoints/PSG/SGDET/M-PE-NET-yolov8m/config.yml" MODEL.ROI_HEADS.DETECTIONS_PER_IMG $i TEST.TOP_K $i TEST.ALLOW_LOAD_FROM_CACHE False TEST.INFORMATIVE False DATASETS.TO_TEST "test" TEST.IMS_PER_BATCH 1
done