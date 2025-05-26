# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
import copy


class DatasetCatalog(object):
    DATA_DIR = "./"
    IMG_DIR = "./datasets/VG150/"
    DATASETS = {
        "VG150": {
            "img_dir": IMG_DIR+"VG_100K",
            "roidb_file": DATA_DIR+"datasets/VG150/VG-SGG-with-attri.h5",
            "dict_file": DATA_DIR+"datasets/VG150/VG-SGG-dicts-with-attri.json",
            "image_file": DATA_DIR+"datasets/vg/image_data.json",
            "zeroshot_file": DATA_DIR+"datasets/VG150/zeroshot_triplet.pytorch",
            "informative_file": "", #DATA_DIR+"datasets/informative_sg.json",
        },
        "PSG": {
            "img_dir": "./datasets/COCO/",
            "ann_file": DATA_DIR+"datasets/psg/psg_train_val.json",
            "informative_file":  "", #DATA_DIR+"datasets/informative_sg.json",
        },
        "VrR-VG_filtered_with_attribute": {
            "img_dir": IMG_DIR+"VG_100K",
            "roidb_file": "VG/VrR-VG/VrR_VG-SGG-with-attri.h5",
            "dict_file": "VG/VrR-VG/VrR_VG-SGG-dicts-with-attri.json",
            "image_file": "VG/VrR-VG/image_data.json",
            "capgraphs_file": "VG/vg_capgraphs_anno.json",
        },
        "VG_indoor_filtered": {
            "img_dir": IMG_DIR+"VG_100K",
            "roidb_file": DATA_DIR+"datasets/IndoorVG_4/VG-SGG-augmented-penet-cat.h5",
            "dict_file": DATA_DIR+"datasets/IndoorVG_4/VG-SGG-dicts.json",
            "image_file": DATA_DIR+"datasets/vg/image_data.json",
            "zeroshot_file": DATA_DIR+"datasets/IndoorVG_4/zero_shot_triplets.pytorch",
            "informative_file": DATA_DIR+"datasets/informative_sg.json",
        },
        "VG178": {
            "img_dir":  IMG_DIR+"VG_100K",
            "roidb_file": DATA_DIR+"VG178/VG-SGG.h5",
            "dict_file": DATA_DIR+"VG178/VG-SGG-dicts.json",
            "image_file": DATA_DIR+"vg/image_data.json",
            "zeroshot_file": DATA_DIR+"VG178/zero_shot_triplets.pytorch",
            "informative_file": DATA_DIR+"datasets/informative_sg.json",
        },
    }

    @staticmethod
    def get(name, cfg):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif ("VG" in name) or ('GQA' in name):
            # name should be something like VG_stanford_filtered_train
            p = name.rfind("_")
            name, split = name[:p], name[p+1:]
            assert name in DatasetCatalog.DATASETS and split in {'train', 'val', 'test'}
            data_dir = DatasetCatalog.DATA_DIR
            args = copy.deepcopy(DatasetCatalog.DATASETS[name])
            # for k, v in args.items():
            #     args[k] = os.path.join(data_dir, v)
            args['split'] = split
            # IF MODEL.RELATION_ON is True, filter images with empty rels
            # else set filter to False, because we need all images for pretraining detector
            args['filter_non_overlap'] = (not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX) and cfg.MODEL.RELATION_ON and cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP
            args['filter_empty_rels'] = True
            args['flip_aug'] = cfg.MODEL.FLIP_AUG
            args['custom_eval'] = cfg.TEST.CUSTUM_EVAL
            args['custom_path'] = cfg.TEST.CUSTUM_PATH
            return dict(
                factory="VGDataset",
                args=args,
            )
        elif "PSG" in name:
            p = name.rfind("_")
            name, split = name[:p], name[p+1:]
            assert name in DatasetCatalog.DATASETS and split in {'train', 'val', 'test'}
            data_dir = DatasetCatalog.DATA_DIR
            args = copy.deepcopy(DatasetCatalog.DATASETS[name])
            # for k, v in args.items():
            #     args[k] = os.path.join(data_dir, v)
            args['split'] = split
            args['filter_empty_rels'] = True
            return dict(
                factory="PSGDataset",
                args=args,
            )

        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag =  ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
