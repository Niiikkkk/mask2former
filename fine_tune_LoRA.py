import os.path

from detectron2.engine import default_argument_parser, launch
from peft import LoraConfig, get_peft_model, inject_adapter_in_model, LoraModel, PeftModel, cast_mixed_precision_params
import torch
from torch.nn.parallel import DistributedDataParallel


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

def change_model_dtype(model: torch.nn.Module, dtype: torch.dtype):
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(dtype)

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
        # lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["predictor"],
    )

    lora_model = get_peft_model(model,lora_cfg)
    print_trainable_params(lora_model)

    optimizer = trainer.build_optimizer(cfg, lora_model)
    trainer._trainer.optimizer = optimizer

    # lora_model.train()
    if isinstance(model, DistributedDataParallel):
        trainer._trainer.model.module = lora_model
    else:
        trainer._trainer.model = lora_model

    trainer.train()

    lora_path = os.path.join(cfg.OUTPUT_DIR, "lora_model")
    if isinstance(model, DistributedDataParallel):
        print("Saving model to ", lora_path)
        trainer._trainer.model.module.save_pretrained(save_directory=lora_path)
        torch.save(trainer._trainer.model.module.state_dict(), lora_path + "/model.pth")
    else:
        print("Saving model to ", lora_path)
        trainer._trainer.model.save_pretrained(save_directory=lora_path)
        torch.save(trainer._trainer.model.state_dict(), lora_path + "/model.pth")
    return

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

