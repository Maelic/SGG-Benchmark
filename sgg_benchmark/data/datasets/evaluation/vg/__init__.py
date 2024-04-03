from .vg_eval import do_vg_evaluation


def vg_evaluation(
    cfg,
    dataset,
    dataset_name,
    predictions,
    output_folder,
    logger,
    iou_types,
    informative,
    **_
):
    return do_vg_evaluation(
        cfg=cfg,
        dataset=dataset,
        dataset_name=dataset_name,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        iou_types=iou_types,
        informative=informative,
    )
