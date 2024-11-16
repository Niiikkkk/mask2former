import numpy
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser

from component_metric import get_threshold_from_PRC, segment_metrics, default_instancer, anomaly_instances_from_mask
from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig
from train_net import Trainer, setup
import os

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS,resume=args.resume)

    model_id = os.path.join(cfg.OUTPUT_DIR, "lora_model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=r"sem_seg_head\.pixel_decoder\.transformer\.encoder\.layers\.\d\.linear\d"
                       r"|sem_seg_head\.pixel_decoder\.transformer\.encoder\.layers\.\d\.self_attn\.\w+"
                       r"|backbone\.res\d\.\d\.conv\d",
        # lora_dropout=0.1,
        bias="none",
        modules_to_save=["predictor"],
    )

    inference_model = get_peft_model(model,lora_config)
    inference_model.load_state_dict(torch.load(cfg.OUTPUT_DIR + "/model_final.pth"))
    # inference_model = PeftModel.from_pretrained(model,model_id)

    res = Trainer.test(cfg,inference_model)
    print(res)