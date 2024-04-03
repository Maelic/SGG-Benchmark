## Here is the implementation of the data transfer technic for data augmentation for the Visual Genome dataset

Original implementation in [https://github.com/waxnkw/IETrans-SGG.pytorch](https://github.com/waxnkw/IETrans-SGG.pytorch) for reference. 
My implementation here is a simplified version, which is also inspired by [https://github.com/franciszzj/HiLo](https://github.com/franciszzj/HiLo).

The Visual Genome dataset is of poor quality, to enlarge and refine annotations the data transfer technic has been proposed, it goes as follows:

1. First, we perform internal transfer to update vague and ambiguous predicates into more precise ones. This is done by selecting the best prediction of an SGG model, which possesses higher confidence than the predicted one that corresponds to the ground truth.
2. Second, we perform external transfer to add annotations for the missed unlabelled triplets. This is done by selecting object pairs that intersect as candidates (i.e. iou > 0) and then retrieving the best prediction from an SGG model.

To use this approach, you'll need to first train an SGG model in SGDet mode. Then, you can call the following script as follows:

```
python ietrans.py YOUR_CONFIG_FILE transfer_results.json
```
with YOUR_CONFIG_FILE being the config.yml file in your training directory. 
This will create a ``` transfer_results.json``` file containing new or updated pairs, you'll need then to fuze them with original annotations using the ```data_fusion.ipynb``` notebook.

You can cite original authors if you use this implementation:

```
@inproceedings{zhang2022fine,
  title={Fine-Grained Scene Graph Generation with Data Transfer},
  author={Zhang, Ao and Yao, Yuan and Chen, Qianyu and Ji, Wei and Liu, Zhiyuan and Sun, Maosong and Chua, Tat-Seng},
  booktitle= "ECCV",
  year={2022}
}
```