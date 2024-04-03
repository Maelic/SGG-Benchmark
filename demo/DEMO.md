## Webcam and Jupyter notebook demo

This folder contains a simple webcam demo where you can perform real-time SGG. First, make sure that you have downloaded or trained a model in SGDet mode, currently only sgdet is supported for demo. 
To run the demo you'll need to:

1. Install the codebase, refer to [INSTALL.md](../INSTALL.md)
2. Train a SGDet model and save the results files in the ```./checkpoints/``` folder
3. Get the path to the config file used for training, it should be under the directory OUTPUT_PATH where you launch the training, e.g. ```./checkpoints/sgdet-causal-motifs```
4. Get the path of the dict file with class names, it should be under ```./datasets/vg/``` and be named something like ```VG-SGG-dicts.json```

Also, make sure that your config.yml file is pointing to the correct path of the weights, either with the MODEL.WEIGHT variable or OUTPUT_PATH that will point to an existing last_checkpoint file.

You can run the demo as follows:

```
conda activate scene_graph_benchmark
python webcam_demo.py --classes PATH_TO_CLASSES_DICT_FILE.json --config YPUR_CONFIG_FILE_HERE.yml
```