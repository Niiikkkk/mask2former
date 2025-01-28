import numpy
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, DefaultPredictor
from detectron2.utils.logger import setup_logger

from component_metric import get_threshold_from_PRC, segment_metrics, default_instancer, anomaly_instances_from_mask
from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig, LoraModel
from train_net import Trainer, setup
from evaluation_on_ood import func, parse_args, setup_cfgs
import os
import sys
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)

    logger = setup_logger(name="fvcore", output=cfg.OUTPUT_DIR)
    logger.info("Arguments: " + str(args))

    stderr_file = os.path.join(cfg.OUTPUT_DIR, 'stderr.txt')
    if not os.path.exists(stderr_file):
        open(stderr_file, 'w').close()
    sys.stderr = open(stderr_file, 'a')
    predictor = DefaultPredictor(cfg)

    lora_path = os.path.join(cfg.OUTPUT_DIR,"lora_model")

    #LOADING LORA MODEL INTO ORIGINAL MODEL
    logger.info(f"Loading LORA model from {lora_path}")
    lora_config = LoraConfig.from_pretrained(lora_path)
    inference_model = get_peft_model(predictor.model,lora_config)
    inference_model.load_state_dict(torch.load(lora_path+"/model.pth"),strict=False)
    predictor.model = inference_model

    # OOD check
    # func(predictor,args,cfg)

    # ID CHECK
    res = Trainer.test(cfg,predictor.model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, predictor.model))
    if comm.is_main_process():
        verify_results(cfg, res)
    logger.info(f"Results: {res}")

    # args = default_argument_parser().parse_args()
    # cfg = setup(args)
    # trainer = Trainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    #
    # stderr_file = os.path.join(cfg.OUTPUT_DIR, 'stderr.txt')
    # if not os.path.exists(stderr_file):
    #     open(stderr_file, 'w').close()
    # sys.stderr = open(stderr_file, 'a')
    #
    # model = trainer._trainer.model
    # model_id = os.path.join(cfg.OUTPUT_DIR, "lora_model")
    # print("Loading LORA weights...")
    # lora_model = PeftModel.from_pretrained(model, model_id, is_trainable=True)
    #
    # lora_model.print_trainable_parameters()
    #
    # print("Building optimizer...")
    # optimizer = trainer.build_optimizer(cfg, lora_model)
    # trainer._trainer.optimizer = optimizer
    # print("Building scheduler...")
    # trainer.scheduler = trainer.build_lr_scheduler(cfg, optimizer)
    #
    # # lora_model.train()
    # print("Starting training...")
    # trainer._trainer.model = lora_model
    #
    # trainer.train()


