# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from sgg_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import tqdm
import os
import time
import datetime
import numpy as np
import wandb

import torch

from sgg_benchmark.config import cfg
from sgg_benchmark.config.defaults_GCL import _C as cfg_GCL
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.solver import make_lr_scheduler
from sgg_benchmark.solver import make_optimizer
from sgg_benchmark.engine.trainer import reduce_loss_dict
from sgg_benchmark.engine.inference import inference
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.collect_env import collect_env_info
from sgg_benchmark.utils.comm import synchronize, get_rank, all_gather
from sgg_benchmark.utils.logger import setup_logger, logger_step
from sgg_benchmark.utils.miscellaneous import mkdir, save_config, set_seed
from sgg_benchmark.utils.parser import default_argument_parser
                           
def train_one_epoch(model, optimizer, data_loader, device, epoch, logger, cfg, scaler, use_wandb=False, use_amp=True):
    pbar = tqdm.tqdm(total=len(data_loader))

    for images, targets, _ in data_loader:
        pbar.update(1)
        if any(len(target) < 1 for target in targets):
            logger.error(f"Epoch={epoch} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        end = time.time()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        # Note: If mixed precision is not used, this ends up doing nothing
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        if use_wandb:
            wandb.log({"loss": losses_reduced}, step=epoch)

        optimizer.zero_grad()
        
        # Scaling loss
        scaler.scale(losses).backward()
        
        # Unscale the gradients of optimizer's assigned params in-place before cliping
        scaler.unscale_(optimizer)

        # fallback to native clipping, if no clip_grad_norm is used
        torch.nn.utils.clip_grad_norm_([p for _, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP)

        scaler.step(optimizer)
        scaler.update()

        batch_time = time.time() - end
        end = time.time()

        # get memory used from cuda
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

        pbar.set_description(f"Epoch={epoch} | Loss={losses_reduced.item():.2f} | Mem={max_mem:.2f}MB")

    return losses_reduced


def train(cfg, logger, args):
    available_metrics = {"mR": "_mean_recall", "R": "_recall", "zR": "_zeroshot_recall", "ng-zR": "_ng_zeroshot_recall", "ng-R": "_recall_nogc", "ng-mR": "_ng_mean_recall", "topA": ["_accuracy_hit", "_accuracy_count"]}

    best_epoch = 0
    best_metric = 0.0
    best_checkpoint = None

    metric_to_track = available_metrics[cfg.METRIC_TO_TRACK]

    logger_step(logger, 'Building model...')
    model = build_detection_model(cfg) 

    # get run name for logger
    if args['use_wandb']:
        project_name = args['project_name']        
        run_name = cfg.OUTPUT_DIR.split('/')[-1]
        if args['distributed']:
            wandb.init(project=project_name, entity="maelic", group="DDP", name=run_name, config=cfg)
        wandb.init(project=project_name, entity="maelic", name=run_name, config=cfg)

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    if cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN":
        eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
        fix_eval_modules(eval_modules)
    else:
        eval_modules = (model.backbone,)
        fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    elif cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == 'SquatPredictor':
        slow_heads = [
            'roi.heads.relation.predictor.context_layer.mask_predictor'
        ]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}
    
    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=2.5, rl_factor=float(num_batch))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    logger_step(logger, 'Building optimizer and scheduler...')

    # Initialize mixed-precision training
    use_amp = True if cfg.DTYPE == "float16" else False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args['local_rank']], output_device=args['local_rank'],
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )

    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(checkpointer.get_checkpoint_file(), update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
    else:
        if "FPN" in cfg.MODEL.BACKBONE.TYPE:
            # load_mapping is only used when we init current model from detection model.
            checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
        else:
            if args['distributed']:
                model.module.backbone.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT)
                model.module.backbone.model.to(device)
            else:
                model.backbone.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT)
                model.backbone.model.to(device)
            # load backbone weights
            logger_step(logger, 'Loading Backbone weights from '+cfg.MODEL.PRETRAINED_DETECTOR_CKPT)

    trained_params = [n for n, p in model.named_parameters() if p.requires_grad]
    pretrain_mask = (cfg.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.PRETRAIN_MASK and cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "SquatPredictor")
    if pretrain_mask:
        STOP_ITER = cfg.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.PRETRAIN_MASK_EPOCH
        for n, p in model.named_parameters(): 
            if p.requires_grad and 'mask_predictor' not in n: 
                p.requires_grad = False
    else:
        STOP_ITER = -1

    mode = get_mode(cfg)

    logger_step(logger, 'Building checkpointer')

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=args['distributed'],
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=args['distributed'],
    )
    logger_step(logger, 'Building dataloader')

    if cfg.SOLVER.PRE_VAL:
        # model.roi_heads.eval()

        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, args['distributed'], logger, device=device)

    max_epoch = cfg.SOLVER.MAX_EPOCH

    logger.info("Start training for {} epochs".format(max_epoch))
    start_training_time = time.time()

    for epoch in range(0, max_epoch):
        if pretrain_mask and epoch == STOP_ITER:
            for n, p in model.named_parameters(): 
                if n in trained_params: 
                    p.requires_grad = True

        if args['distributed']:
            model.module.roi_heads.train()
            model.module.backbone.eval()
        else:
            model.roi_heads.train()
            model.backbone.eval()
        # fix_eval_modules(eval_modules)

        start_epoch_time = time.time()
        _ = train_one_epoch(model, optimizer, train_data_loader, device, epoch, logger, cfg, scaler, args['use_wandb'], use_amp)
        logger.info("Epoch {} training time: {:.2f} s".format(epoch, time.time() - start_epoch_time))

        if not args['save_best']:
            checkpointer.save("model_epoch_{}".format(epoch), **arguments)
  
        val_result = None # used for scheduler updating
        current_metric = None
        logger.info("Start validating")

        val_result = run_val(cfg, model, val_data_loaders, args['distributed'], logger)
        if mode+metric_to_track not in val_result.keys():
            logger.error("Metric to track not found in validation result, default to R")
            metric_to_track = "_recall"
        results = val_result[mode+metric_to_track]
        current_metric = float(np.mean(list(results.values())))
        logger.info("Average validation Result for %s: %.4f" % (cfg.METRIC_TO_TRACK, current_metric))
        
        if current_metric > best_metric:
            best_epoch = epoch
            best_metric = current_metric
            if args['save_best']:
                to_remove = best_checkpoint
                checkpointer.save("best_model_epoch_{}".format(epoch), **arguments)
                best_checkpoint = os.path.join(cfg.OUTPUT_DIR, "best_model_epoch_{}".format(epoch))

                # We delete last checkpoint only after succesfuly writing a new one, in case of out of memory
                if to_remove is not None:
                    os.remove(to_remove+".pth")
                    logger.info("New best model saved at iteration {}".format(epoch))
            
        logger.info("Now best epoch in {} is : {}, with value is {}".format(cfg.METRIC_TO_TRACK+"@k", best_epoch, best_metric))
        
        if args['use_wandb']:
            res_dict = {
                'avg_'+cfg.METRIC_TO_TRACK+"@k": current_metric,
                mode+'_f1': val_result[mode+"_f1"],
                mode+'_recall': val_result[mode+"_recall"],
                mode+'_mean_recall': val_result[mode+"_mean_recall"],
            }

            if cfg.TEST.INFORMATIVE:
                res_dict[mode+'_informative_recall'] = val_result[mode+"_informative_recall"]

            wandb.log(res_dict, step=epoch)            

        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            # Using current_metric instead of traditionnal recall for scheduler
            scheduler.step(current_metric, epoch=epoch)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(epoch))
                break
        else:
            scheduler.step()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )

    name = "model_epoch_{}".format(best_epoch)
    if args['save_best']:
        name = "best_model_epoch_{}".format(best_epoch)
    last_filename = os.path.join(cfg.OUTPUT_DIR, "{}.pth".format(name))
    output_folder = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
    with open(output_folder, "w") as f:
        f.write(last_filename)
    print('\n\n')
    logger.info("Best Epoch is : %.4f" % best_epoch)

    return model, best_checkpoint

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        # module.model.eval()
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def run_val(cfg, model, val_data_loaders, distributed, logger, device=None):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
                            cfg,
                            model,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=None,
                            logger=logger,
                            informative=cfg.TEST.INFORMATIVE,
                        )
        synchronize()

        val_result.append(dataset_result)

    # VG has only one val dataset
    dataset_result = val_result[0]
    if len(dataset_result) == 1:
        return dataset_result
    if distributed:
        for k1, v1 in dataset_result.items():
            for k2, v2 in v1.items():
                dataset_result[k1][k2] = torch.distributed.all_reduce(torch.tensor(np.mean(v2)).to(device).unsqueeze(0)).item() / torch.distributed.get_world_size()
    else:
        for k1, v1 in dataset_result.items():
            if type(v1) != dict or type(v1) != list:
                dataset_result[k1] = v1
                continue
            for k2, v2 in v1.items():
                if isinstance(v2, list):
                    # mean everything
                    v2 = [np.mean(v) for v in v2]
                dataset_result[k1][k2] = np.mean(v2)

    return dataset_result

def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
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
            informative=cfg.TEST.INFORMATIVE,
        )
        synchronize()

def get_mode(cfg):
    task = "sgdet"
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        task = "sgcls"
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == True:
            task = "predcls"
    return task

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
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    for arg in args.opts:
        if "GCL_SETTING" in arg:
            cfg.set_new_allowed(True) # recursively update set_new_allowed to allow merging of configs and subconfigs
            cfg.merge_from_other_cfg(cfg_GCL)
            break
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.task:
        assert_mode(cfg, args.task)
    cfg.freeze()

    # set seed
    set_seed(seed=cfg.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("sgg_benchmark", output_dir, get_rank(), verbose=cfg.VERBOSE, steps=True)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.debug(args)

    logger_step(logger, "Collecting environment info...")
    logger.debug("Loaded configuration: {}".format(collect_env_info()))

    # logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.debug(config_str)
    logger.debug("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    training_args = {"task:": args.task, 
        "save_best": args.save_best, 
        "use_wandb": args.use_wandb, 
        "skip_test": args.skip_test, 
        "local_rank": args.local_rank, 
        "distributed": args.distributed,
        "project_name": args.name,
    }

    model, best_checkpoint = train(
        cfg=cfg,
        logger=logger,
        args=training_args
    )

    if not args.skip_test:
        checkpointer = DetectronCheckpointer(cfg, model)
        last_check = best_checkpoint+'.pth'
        if last_check != "":
            logger.info("Loading best checkpoint from {}...".format(last_check))
            _ = checkpointer.load(last_check)
        else:
            _ = checkpointer.load(last_check)
        run_test(cfg, model, args.distributed, logger)

    
    logger.info("#"*20+" END TRAINING "+"#"*20)


if __name__ == "__main__":
    main()