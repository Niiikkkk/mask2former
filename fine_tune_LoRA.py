import os.path
import sys
from copy import deepcopy

from detectron2.engine import default_argument_parser, launch
from detectron2.utils.logger import setup_logger
from peft import LoraConfig, get_peft_model, inject_adapter_in_model, LoraModel, PeftModel, cast_mixed_precision_params, \
    get_peft_model_state_dict
import torch
from torch.nn.parallel import DistributedDataParallel
from train_net import Trainer, setup
from safetensors.torch import save_file as safe_save_file

def print_trainable_params(model: torch.nn.Module):
    for name,p in model.named_parameters():
        if p.requires_grad:
            print(name + " has " + str(p.numel()) + " trainable parameters. Dtype = " + str(p.type()))

def print_named_modules(model):
    names = [(n, type(m)) for n, m in model.named_modules()]
    for name, module in names:
        print(name, module)

def print_total_params(model: torch.nn.Module):
    total_params = sum(p.numel() for n,p in model.named_parameters())
    return total_params

def get_lora_weights(model):
    model_weights = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            model_weights[n] = model.state_dict()[n]
    for n, p in model.named_buffers():
        model_weights[n] = p
    return model_weights

def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    stderr_file = os.path.join(cfg.OUTPUT_DIR, 'stderr.txt')
    if not os.path.exists(stderr_file):
        open(stderr_file, 'w').close()
    sys.stderr = open(stderr_file, 'a')

    model = trainer._trainer.model
    tmp_model = deepcopy(model)

    if isinstance(model, DistributedDataParallel):
        print("Model is DistributedDataParallel")
        model = trainer._trainer.model.module
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=r"sem_seg_head\.pixel_decoder\.
        target_modules=#r"sem_seg_head\.pixel_decoder\.transformer\.encoder\.layers\.\d\.self_attn\.\w+"
                        r"backbone\.res\d\.\d\.conv\d"
                        # r"|sem_seg_head\.predictor\.transformer_ffn_layers\.\d\.linear.+"
                       r"|sem_seg_head\.predictor\.transformer_cross_attention_layers\.\d\.multihead_attn\.\w+"
                       r"|sem_seg_head\.predictor\.transformer_self_attention_layers\.\d\.self_attn\.\w+",
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["sem_seg_head.predictor.mask_embed",
                         "sem_seg_head.pixel_decoder.input_proj.0",
                         "sem_seg_head.pixel_decoder.input_proj.1",
                         "sem_seg_head.pixel_decoder.input_proj.2",
                         "sem_seg_head.predictor.query_embed",
                         "sem_seg_head.predictor.query_feat",
                         "sem_seg_head.predictor.class_embed",],
        #query_embed, query_feat, class_embed, mask_embed.
    )

    ##CONTROLLARE I TRAINABLE PARAMS!!!! E TUTTI I PARAMS



    lora_cfg_old = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=r"sem_seg_head\.pixel_decoder\.
        target_modules=r"sem_seg_head\.pixel_decoder\.transformer\.encoder\.layers\.\d\.self_attn\.\w+"
                       r"|sem_seg_head\.predictor\.transformer_ffn_layers\.\d\.linear.+"
                       r"|sem_seg_head\.predictor\.transformer_cross_attention_layers\.\d\.multihead_attn\.\w+"
                       r"|sem_seg_head\.predictor\.transformer_self_attention_layers\.\d\.self_attn\.\w+",
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["backbone",
                         "sem_seg_head.predictor.mask_embed",
                         "sem_seg_head.pixel_decoder.input_proj.0",
                         "sem_seg_head.pixel_decoder.input_proj.1",
                         "sem_seg_head.pixel_decoder.input_proj.2"],
        #query_embed, query_feat, class_embed, mask_embed.
    )

    logger = setup_logger(name="info", output=cfg.OUTPUT_DIR)
    logger.info("Number of trainable parameters before LoRA: " + str(print_total_params(model)))
    lora_model = get_peft_model(model,deepcopy(lora_cfg))
    trainable, total = lora_model.get_nb_trainable_parameters()

    logger.info("Number of trainable parameters after LoRA: " + str(trainable) + " Total: " + str(total) +
                f" Percentage: {100*trainable/total} %")

    optimizer = trainer.build_optimizer(cfg, lora_model)
    trainer._trainer.optimizer = optimizer
    trainer.scheduler = trainer.build_lr_scheduler(cfg,optimizer)

    # lora_model.train()
    if isinstance(model, DistributedDataParallel):
        trainer._trainer.model.module = lora_model
    else:
        trainer._trainer.model = lora_model

    trainer.train()

    lora_path = os.path.join(cfg.OUTPUT_DIR, "lora_model")
    if isinstance(model, DistributedDataParallel):
        logger.info("Saving Dist Model to ", lora_path)
        trainer._trainer.model.module.save_pretrained(lora_path)
    else:
        logger.info("Saving Model to ", lora_path)
        lora_cfg.save_pretrained(lora_path)

        model_weights = get_lora_weights(trainer._trainer.model)
        torch.save(model_weights,lora_path+"/model.pth")


    #TEST:
    trainer._trainer.model.eval()
    lora_cfg = LoraConfig.from_pretrained(lora_path)
    loaded = get_peft_model(tmp_model, lora_cfg)
    loaded.load_state_dict(torch.load(lora_path+"/model.pth"),strict=False)
    loaded.eval()
    x = torch.rand(3, 512, 512).cuda()
    inputs = {"image": x, "height": 512, "width": 512}
    y_peft = trainer._trainer.model([inputs])[0]["sem_seg"]
    y_loaded = loaded([inputs])[0]["sem_seg"]
    print(torch.allclose(y_peft, y_loaded))

    return

if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

