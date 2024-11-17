import numpy
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser

from component_metric import get_threshold_from_PRC, segment_metrics, default_instancer, anomaly_instances_from_mask
from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig
from train_net import Trainer, setup
from evaluation_on_ood import func, parse_args
import os

if __name__ == '__main__':
    args = parse_args()
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS,resume=args.resume)

    model_id = os.path.join(cfg.OUTPUT_DIR, "lora_model")

    lora_config = LoraConfig.from_pretrained(model_id)

    inference_model = get_peft_model(model,lora_config)
    inference_model.load_state_dict(torch.load(model_id + "/model.pth"))
    inference_model.eval()
    # inference_model = PeftModel.from_pretrained(model,model_id)

    func(inference_model,args,cfg)
    #res = Trainer.test(cfg,inference_model)

