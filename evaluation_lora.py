import numpy
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser
from component_metric import get_threshold_from_PRC, segment_metrics, default_instancer, anomaly_instances_from_mask
from peft import PeftConfig, PeftModel
from train_net import Trainer, setup
import os

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS,resume=args.resume)

    model_id = os.path.join(cfg.OUTPUT_DIR, "lora_model")

    inference_model = PeftModel.from_pretrained(model,model_id)

    print(inference_model)

    res = Trainer.test(cfg,inference_model)
    print(res)