from sgg_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation
from .vg import vg_evaluation


def evaluate(cfg, dataset, dataset_name, predictions, output_folder, logger, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        cfg=cfg, dataset=dataset, dataset_name=dataset_name, predictions=predictions, output_folder=output_folder, logger=logger, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.VGDataset):
        return vg_evaluation(**args)
    elif isinstance(dataset, datasets.PSGDataset):
        return vg_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
