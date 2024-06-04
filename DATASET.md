# DATASETS

| Dataset  | Scene Graphs (Download Link) | Images (Download Link) | Training Set | Validation Set | Test Set |
|----------|------------------------------|------------------------|--------------|----------------|----------|
| VG-150   | [Download](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed) | [Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)<br>[Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) | 68,538 | 5,000 | 31,876 |
| IndoorVG | [Download](https://drive.google.com/file/d/1zfKXzmLxxYMzwlECtSch84oknCBEXTzI/view?usp=sharing) | Uses VG-150 images | 9,862 | 700 | 4,112 |
| PSG      | [Download](https://entuedu-my.sharepoint.com/personal/jingkang001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fjingkang001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2Fopenpsg%2Fdata%2Fpsg) | [Download](https://entuedu-my.sharepoint.com/:u:/r/personal/jingkang001_e_ntu_edu_sg/Documents/openpsg/data/coco.zip?csf=1&web=1&e=9Z513T) | 45,697 | 1,000 | 1,177 |


## VG-150

This is the data split used by most SGG papers. It is composed of the top 150 object classes and 50 predicate classes from the Visual Genome dataset.
The pre-processing of this split is adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs). All object bounding boxes corresponding to the 150 most common object classes are selected, which means that some images have objects but no relations. 

:warning: This data split has been heavily criticized for having high biases in the data distribution and classes semantics (e.g. classes person, man, men, people, etc highly intersect in the annotations). See [this paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Liang_VrR-VG_Refocusing_Visually-Relevant_Relationships_ICCV_2019_paper.html) and [this one](https://openaccess.thecvf.com/content/ICCV2023W/SG2RL/html/Neau_Fine-Grained_is_Too_Coarse_A_Novel_Data-Centric_Approach_for_Efficient_ICCVW_2023_paper.html?trk=article-ssr-frontend-pulse_x-social-details_comments-action_comment-text) for reference. Use it at your own risks.

Note that VG150 annotations intends to support attributes since the [work from Kaihua](https://openaccess.thecvf.com/content_CVPR_2020/html/Tang_Unbiased_Scene_Graph_Generation_From_Biased_Training_CVPR_2020_paper.html), so the ```VG-SGG.h5``` and ```VG-SGG-dicts.json``` are different with their original versions in [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs). Attribute information has been added and the file renamed to be ```VG-SGG-with-attri.h5``` and ```VG-SGG-dicts-with-attri.json```. The code used to generate them is located at ```process_data/generate_attribute_labels.py```. Attribute head is disabled by default in the codebase due to poor performance in reported papers.

### Download VG150:
1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `sgg_benchmark/config/paths_catelog.py`. 
2. Download the [scene graphs](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed) and extract them to `datasets/vg/VG-SGG-with-attri.h5`, or you can edit the path in `DATASETS['VG_stanford_filtered_with_attribute']['roidb_file']` of `sgg_benchmark/config/paths_catalog.py`.

## IndoorVG

This data split is proposed by us in a [recent approach](https://link.springer.com/chapter/10.1007/978-3-031-55015-7_25). It is another split of Visual Genome targeting real-world applications in indoor settings. It is composed of 83 object classes and 34 predicates classes which have been manually selected and refined using different semi-automatic merging and processing technics. To use it you can download the VG images from link above and the annotated scene graphs [from this link](https://drive.google.com/file/d/1zfKXzmLxxYMzwlECtSch84oknCBEXTzI/view?usp=sharing). We also provide augmented version of the data using the IETrans-type approach detailed in [here](process_data/data_augmentation/README.md). The original split is VG-SGG.h5, then the augmented splits add up to 34% more annotations and refine existing ones.


## PSG

The PSG dataset is a new approach originally targeting the [Panoptic Scene Graph Generation](https://arxiv.org/abs/2207.11247) task. However, its annotations can also be used for traditional SGG. It is composed of images from COCO and VG which have been re-annotated aiming at fixing biases from Visual Genome. The data (images + graphs) can be downloaded using [the authors link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBDAb7v0UgI8iIAExQUJq62Q?e=fIY3zh).
Note that for efficient encoding of the class labels it is necessary to change some names (i.e. removed "-merged" or "-other" suffixes), please see our pre-process class names in [datasets/psg/obj_classes.txt](datasets/psg/obj_classes.txt)

# YOLO Format:

We provide a script to convert any annotations in SGG format (.h5 file) to YOLO format, which make it easier for training an Object Detection backbone. Please have a look if you're interested: [process_data/convert_to_yolo.ipynb](process_data/convert_to_yolo.ipynb)

Other tools related to data augmentation and conversion can be found under ```process_data/```.