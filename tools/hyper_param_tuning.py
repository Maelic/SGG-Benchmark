import os
import numpy as np
import pickle
import math
import torch

import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.experimental.tqdm_ray import tqdm
from ray.tune.search.bayesopt import BayesOptSearch
from optuna.samplers import TPESampler

from sgg_benchmark.config import cfg
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.solver import make_lr_scheduler
from sgg_benchmark.solver import make_optimizer
from sgg_benchmark.engine.trainer import reduce_loss_dict
from sgg_benchmark.engine.inference import inference
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.comm import synchronize, get_rank
from sgg_benchmark.utils.logger import setup_logger, logger_step
from sgg_benchmark.utils.miscellaneous import mkdir, save_config
from sgg_benchmark.utils.parser import default_argument_parser

METRICS = {"mR": "_mean_recall", "R": "_recall", "zR": "_zeroshot_recall", "ng-zR": "_ng_zeroshot_recall", "ng-R": "_recall_nogc", "ng-mR": "_ng_mean_recall", "f1": "_f1_score", "topA": ["_accuracy_hit", "_accuracy_count"]}

def train_relation_net(config):
    model, optimizer, train_data_loader, val_data_loaders, device, logger, cfg, scaler, max_iter = setup(config) 
    mode = get_mode(cfg)

    metric_to_track = METRICS["f1"]

    logger.info("Start training for %d iterations" % max_iter)

    # check if "use_amp" key is in config["tuning_config"]
    if "use_amp" not in config["tuning_config"]:
        use_amp = True
    else:
        use_amp = config["tuning_config"]["use_amp"]
    
    for epoch in range(0, config["tuning_config"]["max_epoch"]):
        iter = 0
        if cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN":
            model.train()
            eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
            fix_eval_modules(eval_modules)
        else:
            model.roi_heads.train()
            model.backbone.eval()

        pbar = tqdm(total=max_iter)

        for images, targets, _ in train_data_loader:
            iter += 1
            if iter > max_iter:
                break
            pbar.update(1)
            if any(len(target) < 1 for target in targets):
                logger.error(f"Epoch={epoch} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
                continue

            images = images.to(device)
            targets = [target.to(device) for target in targets]

            # Note: If mixed precision is not used, this ends up doing nothing
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            optimizer.zero_grad()
            
            # Scaling loss
            scaler.scale(losses).backward()
            
            # Unscale the gradients of optimizer's assigned params in-place before cliping
            scaler.unscale_(optimizer)

            # fallback to native clipping, if no clip_grad_norm is used
            torch.nn.utils.clip_grad_norm_([p for _, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP)

            scaler.step(optimizer)
            scaler.update()

            # get memory used from cuda
            if torch.cuda.is_available():
                max_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

            pbar.set_description(f"Epoch={epoch} | Loss={losses_reduced.item():.2f} | Mem={max_mem:.2f}MB")
            losses_report = float(losses_reduced.item())
            # train.report({"loss": losses_report},)

        current_metric = None

        val_result = run_val(cfg, model, val_data_loaders, False, logger)
        if mode+metric_to_track not in val_result.keys():
            logger.error("Metric to track not found in validation result, default to R")
            metric_to_track = "_recall"
        results = val_result[mode+metric_to_track]
        current_metric = float(np.mean(list(results.values())))

        train.report({"loss": losses_report, "f1_score": current_metric},)

def setup(config):
    config_file = config["config_path"]

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(config["opts"])
    if "model_config" in config:
        # config["model_config"] to list
        conf_model = []
        for k, v in config["model_config"].items():
            conf_model.append(k)
            conf_model.append(v)
        print(conf_model)
        cfg.merge_from_list(conf_model)
    if config["task"]:
        assert_mode(cfg,config["task"])

    cfg.SOLVER.IMS_PER_BATCH = config["tuning_config"]["batch_size"] if "batch_size" in config["tuning_config"] else cfg.SOLVER.IMS_PER_BATCH
    cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("sgg_benchmark", output_dir, get_rank(), verbose="WARNING", steps=True)

    tuning_config = config["tuning_config"]

    # logger_step(logger, 'Building model...')
    model = build_detection_model(cfg)

    # Model eval mode settings
    if cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN":
        eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
        fix_eval_modules(eval_modules)
    else:
        model.backbone.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Optimizer and scheduler setup
    cfg.SOLVER.BASE_LR = tuning_config["lr"] if "lr" in tuning_config else cfg.SOLVER.BASE_LR
    cfg.SOLVER.MOMENTUM = tuning_config["momentum"] if tuning_config["optimizer"] == "SGD" else cfg.SOLVER.MOMENTUM
    cfg.SOLVER.WEIGHT_DECAY = tuning_config["decay"] if tuning_config["optimizer"] == "ADAMW" else cfg.SOLVER.WEIGHT_DECAY
    cfg.SOLVER.OPTIMIZER = tuning_config["optimizer"] if "optimizer" in tuning_config else cfg.SOLVER.OPTIMIZER

    if "num_images" in tuning_config:
        max_iter = tuning_config["num_images"] // cfg.SOLVER.IMS_PER_BATCH
    else:
        max_iter = cfg.SOLVER.MAX_ITER

    optimizer = make_optimizer(cfg, model, logger, rl_factor=float(cfg.SOLVER.IMS_PER_BATCH))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)

    # Initialize mixed-precision training
    if "use_amp" not in config["tuning_config"]:
        use_amp = True
    else:
        use_amp = tuning_config["use_amp"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # # DistributedDataParallel setup
    # if args['distributed']:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args['local_rank']], output_device=args['local_rank'],
    #         broadcast_buffers=False,
    #         find_unused_parameters=True,
    #     )

    # Checkpointer
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    model.backbone.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT)
    model.backbone.model.to(device)

    # Data loaders
    train_data_loader = make_data_loader(
        cfg, mode='train', is_distributed=False, start_iter=0,
    )
    val_data_loaders = make_data_loader(
        cfg, mode='val', is_distributed=False, start_iter=0,
    )

    # print the size of val_data_loaders[0]
    print(f"Size of val_data_loaders[0]: {len(val_data_loaders[0].dataset)}")

    return model, optimizer, train_data_loader, val_data_loaders, device, logger, cfg, scaler, max_iter   

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        # module.model.eval()
        for _, param in module.named_parameters():
            param.requires_grad = False

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
                            silence=True,
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

    ray.init(
        _system_config={
            # Allow spilling until the local disk is 99% utilized.
            # This only affects spilling to the local file system.
            "local_fs_capacity_threshold": 0.99,
        },
    )

    max_epoch = 5 # Max number of epochs to run
    max_images = 2000 # One epoch could be too long for tuning, so we limit the number of images
    optimizer = "SGD" # Optimizer to use, choose between "SGD" and "ADAMW"

    # training hypeparameters
    if optimizer == "SGD":
        search_space = {
            "tuning_config": {
                "optimizer": optimizer,
                "lr": tune.loguniform(1e-5, 1e-1), # Learning rate
                "momentum": tune.uniform(0.1, 0.9), # Momentum for SGD
                #"batch_size": tune.choice([2, 4, 8]),
                "max_epoch": max_epoch,
                "num_images": max_images,
                # "use_amp": tune.choice([True, False]),
                # Add other tuning parameters here
            },
            "config_path": args.config_file,
            "task": args.task,
            "opts": args.opts,
        }
    elif optimizer == "ADAMW":
        search_space = {
            "tuning_config": {
                "optimizer": optimizer,
                "lr": tune.loguniform(1e-5, 1e-1), # Learning rate
                "decay": tune.loguniform(1e-5, 1e-1), # Weight decay for AdamW
                # "batch_size": tune.choice([2, 4, 8]),
                "max_epoch": max_epoch,
                "num_images": max_images,
                # "use_amp": tune.choice([True, False]),
                # Add other tuning parameters here
            },
            "config_path": args.config_file,
            "task": args.task,
            "opts": args.opts,
        }

    # model hyperparameters
    model_config = {
        "MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM": tune.choice([512, 1024, 2048, 4096]),
        "MODEL.ROI_RELATION_HEAD.MLP_HEAD_DIM": tune.choice([512, 1024, 2048, 4096]),
        "MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM": tune.choice([256, 512, 1024, 2048]),
    }

    squat_config = {
        "MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.NUM_DECODER": tune.choice([1,2,3,4,5]),
        "MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.RHO": tune.uniform(0.1, 0.9),
        "MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.BETA": tune.uniform(0.1, 0.9),
        "MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.PRE_NORM": tune.choice([True, False]),
    }

    pooler_config = {
        "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION": tune.choice([5, 7, 9]),
        "MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO": tune.choice([0, 1, 2, 3]),
    }

    # experimental
    #search_space.update({"model_config":model_config})

    # config taken from https://docs.ray.io/en/latest/tune/api/schedulers.html
    scheduler = ASHAScheduler(
        metric="f1_score",
        mode="max",
        max_t=max_images//cfg.SOLVER.IMS_PER_BATCH,
        grace_period=1,
        reduction_factor=3,
        brackets=1,
    )
    # TPESampler sampler 
    algo = OptunaSearch(metric="f1_score", mode="max")

    # Configuration for the tuning
    tune_config = tune.TuneConfig(
        search_alg=algo,
        scheduler=scheduler,
        num_samples=50, # Adjust for how many trials you want to run, more is better but will take longer
    )

    # Start the Ray Tune run
    trainable_with_cpu_gpu = tune.with_resources(train_relation_net, {"cpu": 6, "gpu": 1})

    tuner = tune.Tuner(  
        trainable_with_cpu_gpu,
        tune_config=tune_config,
        run_config=train.RunConfig(stop=stopnanloss), # Stop if loss is NaN, useful for AdamW
        param_space=search_space,
    )

    results = tuner.fit()
    # save results
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        save_dir = os.path.join(output_dir, "raytune_results")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "results.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Ray Tune results saved to {save_path}")

nan_loss_counter = {}
max_nan_losses = 20 # Maximum number of NaN loss iterations before stopping the trial

def stopnanloss(trial_id, result):
    # Check if the loss is NaN
    if math.isnan(result["loss"]):
        # If the trial is already in the dictionary, increment its count
        if trial_id in nan_loss_counter:
            nan_loss_counter[trial_id] += 1
        else:
            # If this is the first NaN loss for the trial, add it to the dictionary
            nan_loss_counter[trial_id] = 1
        
        # Check if the trial has exceeded max NaN losses
        if nan_loss_counter[trial_id] > max_nan_losses:
            # If so, return True to stop the trial
            return True
    else:
        # If the loss is not NaN, reset the counter for this trial
        if trial_id in nan_loss_counter:
            del nan_loss_counter[trial_id]
    return False

if __name__ == "__main__":
    main()