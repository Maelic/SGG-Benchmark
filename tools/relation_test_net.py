from sgg_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import os

import torch
from sgg_benchmark.config import cfg
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.engine.inference import inference
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.comm import synchronize, get_rank
from sgg_benchmark.utils.logger import setup_logger, logger_step
from sgg_benchmark.utils.miscellaneous import mkdir, set_seed
from sgg_benchmark.utils.parser import default_argument_parser
from sgg_benchmark.data import get_dataset_statistics
# from calflops import calculate_flops

def assert_mode(cfg, task):
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
    if task == "sgcls" or task == "predcls":
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = True
    if task == "predcls":
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = True

def main():
    args = default_argument_parser()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.task:
        assert_mode(cfg, args.task)

    # set seed
    set_seed(seed=cfg.SEED)

    output_dir = cfg.OUTPUT_DIR


    logger = setup_logger("sgg_benchmark", output_dir, get_rank(), filename="log.txt", steps=True, verbose=cfg.VERBOSE)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.debug(args)

    logger_step(logger, "Collecting environment info...")
    # logger.debug("\n" + collect_env_info())

    # logger.info("Running with config:\n{}".format(cfg))
    # compute_flops(cfg, output_dir, logger=logger)

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    enable_inplace_relu(model)

    # Initialize mixed-precision if necessary
    use_amp = True if cfg.DTYPE == "float16" or args.amp else False

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    last_check = checkpointer.get_checkpoint_file()
    if last_check == "":
        last_check = cfg.MODEL.WEIGHT
    logger.info("Loading last checkpoint from {}...".format(last_check))
    _ = checkpointer.load(last_check)

    if "world" in cfg.MODEL.BACKBONE.TYPE:
        stats = get_dataset_statistics(cfg)
        obj_classes = stats['obj_classes'][1:]
        # for debugging
        #print("Loading txt embeddings for object classes: ", obj_classes)
        model.backbone.load_txt_feats(obj_classes)

    model.backbone.eval()
    model.roi_heads.eval()
    
    iou_types = ("bbox",)
    if cfg.MODEL.RELATION_ON:
        logger.info("Evaluate relations")
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        logger.info("Evaluate attributes")
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST)

    dataset_names = cfg.DATASETS.TEST

    # This variable enables the script to run the test on any dataset split.
    if cfg.DATASETS.TO_TEST:
        assert cfg.DATASETS.TO_TEST in {'train', 'val', 'test', None}
        if cfg.DATASETS.TO_TEST == 'train':
            dataset_names = cfg.DATASETS.TRAIN
        elif cfg.DATASETS.TO_TEST == 'val':
            dataset_names = cfg.DATASETS.VAL

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_"+str(cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG))
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg=cfg, mode="test", is_distributed=distributed, dataset_to_test=cfg.DATASETS.TO_TEST)

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        # Note: If mixed precision is not used, this ends up doing nothing
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                logger=logger,
                informative=True,
            )
        synchronize()

    del model
    torch.cuda.empty_cache()

def compute_flops(cfg, output_dir, logger):
    logger.info("Computing FLOPs and MACs...")
    cfg.TEST.CUSTUM_EVAL = True
    model = build_detection_model(cfg)

    model.to(cfg.MODEL.DEVICE)

    enable_inplace_relu(model)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    last_check = checkpointer.get_checkpoint_file()
    if last_check == "":
        last_check = cfg.MODEL.WEIGHT
    logger.info("Loading last checkpoint from {}...".format(last_check))
    _ = checkpointer.load(last_check)

    model.backbone.eval()
    model.roi_heads.eval() 

    h_dim = cfg.INPUT.MIN_SIZE_TRAIN
    w_dim = cfg.INPUT.MAX_SIZE_TRAIN    

    input_shape = (1, 3, h_dim, w_dim)
    model.backbone.eval()
    model.roi_heads.eval()
    flops, macs, params = calculate_flops(model=model, 
                                    input_shape=input_shape,
                                    output_as_string=True,
                                    output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    cfg.TEST.CUSTUM_EVAL = False

    del model
    torch.cuda.empty_cache()

def enable_inplace_relu(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            setattr(model, name, torch.nn.ReLU(inplace=True))
        else:
            enable_inplace_relu(module)

if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
