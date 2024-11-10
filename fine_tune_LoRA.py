from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch
from peft import LoraConfig, get_peft_model, inject_adapter_in_model, LoraModel, PeftModel, cast_mixed_precision_params
import torch
from six import print_
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForSemanticSegmentation, PreTrainedModel
from torch.utils.collect_env import main as collect_env

from train_net import Trainer, setup

def print_trainable_params(model: torch.nn.Module):
    total_params = 0
    trainable_params = 0
    for name,p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
            print(name + " has " + str(p.numel()) + " trainable parameters. Dtype = " + str(p.dtype))

    print(f"Total params: {total_params}, Trainable params: {trainable_params}")

def print_named_modules(model):
    names = [(n, type(m)) for n, m in model.named_modules()]
    for name, module in names:
        print(name, module)

def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    model = trainer._trainer.model
    if isinstance(model, DistributedDataParallel):
        model = trainer._trainer.model.module

    # print_trainable_params(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=r"sem_seg_head\.pixel_decoder\.transformer\.encoder\.layers\.\d\.linear\d"
                       r"|sem_seg_head\.pixel_decoder\.transformer\.encoder\.layers\.\d\.self_attn\.\w+"
                       r"|backbone\.res\d\.\d\.conv\d",
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["predictor"],
    )
    lora_model = inject_adapter_in_model(lora_cfg,model)

    print_trainable_params(lora_model)

    # lora_model.train()
    if isinstance(model, DistributedDataParallel):
        trainer._trainer.model.module = lora_model
    else:
        trainer._trainer.model = lora_model

    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

