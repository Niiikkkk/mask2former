import numpy
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, DefaultPredictor
from detectron2.utils.logger import setup_logger

from component_metric import get_threshold_from_PRC, segment_metrics, default_instancer, anomaly_instances_from_mask
from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig, LoraModel
from train_net import Trainer
from evaluation_on_ood import func, parse_args, setup_cfgs
import os

if __name__ == '__main__':
    args = parse_args()
    cfg = setup_cfgs(args)

    logger = setup_logger(name="fvcore", output=cfg.OUTPUT_DIR)
    logger.info("Arguments: " + str(args))

    predictor = DefaultPredictor(cfg)

    model_id = os.path.join(cfg.OUTPUT_DIR,"lora_model")

    lora_config = LoraConfig.from_pretrained(model_id)
    inference_model = PeftModel.from_pretrained(predictor.model,model_id)

    inference_model.eval()
    predictor.model = inference_model
    # inference_model = PeftModel.from_pretrained(model,model_id)

    # func(predictor,args,cfg)
    res = Trainer.test(cfg,predictor.model)

