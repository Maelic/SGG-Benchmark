# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from sgg_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime

import torch
from sgg_benchmark.config import cfg
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.solver import make_lr_scheduler
from sgg_benchmark.solver import make_optimizer
from sgg_benchmark.engine.trainer import reduce_loss_dict
from sgg_benchmark.engine.inference import inference
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.collect_env import collect_env_info
from sgg_benchmark.utils.comm import synchronize, get_rank, all_gather
from sgg_benchmark.utils.imports import import_file
from sgg_benchmark.utils.logger import setup_logger, logger_step
from sgg_benchmark.utils.miscellaneous import mkdir, save_config
from sgg_benchmark.utils.metric_logger import MetricLogger
import wandb

from tqdm import tqdm

def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch, logger, cfg, scaler, use_wandb=False, use_amp=True):
    pbar = tqdm(total=len(data_loader))

    for images, targets, _ in data_loader:
        pbar.update(1)
        if any(len(target) < 1 for target in targets):
            logger.error(f"Epoch={epoch} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        end = time.time()

        # Note: If mixed precision is not used, this ends up doing nothing
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            images = images.to(device)
            targets = [target.to(device) for target in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        if use_wandb:
            wandb.log({"loss": losses_reduced}, step=epoch)

        optimizer.zero_grad()

        optimizer.step()
        scheduler.step()

        end = time.time()

        # get memory used from cuda
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

        pbar.set_description(f"Epoch={epoch} | Loss={losses_reduced.item():.2f} | Mem={max_mem:.2f}MB")

    return losses_reduced


def train(cfg, local_rank, distributed, logger, use_wandb=False):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)


    optimizer = make_optimizer(cfg, model, logger, rl_factor=float(cfg.SOLVER.IMS_PER_BATCH))
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_amp = True if cfg.DTYPE == "float16" else False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
    arguments.update(extra_checkpoint_data)

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed)

    logger.info("Start training")
    max_iter = len(train_data_loader)
    start_training_time = time.time()

    val_result = 0
    best_metric = 0
    best_epoch = ""
    best_checkpoint = None

    max_epoch = cfg.SOLVER.MAX_EPOCH

    logger.info("Start training for {} epochs".format(max_epoch))
    start_training_time = time.time()

    for epoch in range(0, max_epoch):

        model.train()

        start_epoch_time = time.time()
        loss = train_one_epoch(model, optimizer, scheduler, train_data_loader, device, epoch, logger, cfg, scaler, use_wandb=use_wandb, use_amp=use_amp)
        logger.info("Epoch {} training time: {:.2f} s".format(epoch, time.time() - start_epoch_time))

        val_result = None # used for scheduler updating
        logger.info("Start validating")

        val_result = run_val(cfg, model, val_data_loaders, distributed)

        if val_result > best_metric:
            best_epoch = epoch
            best_metric = val_result
            
            to_remove = best_checkpoint
            checkpointer.save("best_model_epoch_{}".format(epoch), **arguments)
            best_checkpoint = os.path.join(cfg.OUTPUT_DIR, "best_model_epoch_{}".format(epoch))

            # We delete last checkpoint only after succesfuly writing a new one, in case of out of memory
            if to_remove is not None:
                os.remove(to_remove+".pth")
                logger.info("New best model saved at iteration {}".format(epoch))
            
        logger.info("Now best epoch in mAP is : {}, with value {}".format(best_epoch, best_metric))
        if use_wandb:
            wandb.log({"mAp": val_result})

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    return model


def run_val(cfg, model, val_data_loaders, distributed):
    results = []
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
        
    dataset_names = cfg.DATASETS.VAL
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
        )
        synchronize()

    gathered_result = all_gather(torch.tensor(dataset_result['mAP']).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    torch.cuda.empty_cache()
    return val_result

def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(
        cfg, 
        mode='test', 
        is_distributed=distributed
        )
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
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--use-wandb",
        dest="use_wandb",
        help="Use wandb logger (Requires wandb installed)",
        action="store_true",
        default=False
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    if args.use_wandb:
        run_name = cfg.OUTPUT_DIR.split('/')[-1]
        if args.distributed:
            wandb.init(project="scene-graph-benchmark", entity="maelic", group="DDP", name=run_name, config=cfg)
        wandb.init(project="scene-graph-benchmark", entity="maelic", name=run_name, config=cfg)
        use_wandb = True
        
    logger = setup_logger("sgg_benchmark", output_dir, get_rank(), verbose="INFO")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, logger, use_wandb=use_wandb)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
