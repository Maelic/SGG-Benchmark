## Webcam and Jupyter notebook demo

This folder contains a simple webcam demo where you can perform real-time SGG. First, make sure that you have downloaded or trained a model in SGDet mode, currently only sgdet is supported for demo. 
To run the demo you'll need to:

1. Install the codebase, refer to [INSTALL.md](../INSTALL.md)
2. Train a SGDet model and save the results files in the ```./checkpoints/``` folder
3. Get the path to the config file used for training, for instance `./configs/VG150/baseline/e2e_relation_X_101_32_8_FPN_1x.yaml`
4. Get the path of your trained weights, for instance `./checkpoints/upload_causal_motif_sgdet/model_0028000.pth`

5. OPTIONAL: if you want to spice it up, you can add a real-time tracker on top of the object detection to track relations between the same objects through time with boxmot (`pip install boxmot`). You can then activate it with argument `--tracking`.
6. OPTIONAL: you can also configure the box and rel confidence threshold, by default I put 0.1 for the relation and 0.5 for the boxes, if no relation or too many relations are shown, try to adjust those thresholds with the arguments `--rel_conf` and `--box_conf`.

You can run the demo as follows:

```
python webcam_demo.py --config YOUR_CONFIG_FILE_HERE.yml --weights YOUR_WEIGHTS_FILE.pth --tracking # only activate tracking if boxmot is installed
```

You can also use the [SGDET_on_custom_images.ipynb](SGDET_on_custom_images.ipynb) notebook to visualize detections on images.