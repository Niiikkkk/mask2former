import os.path
from functools import partial

from detectron2.engine import default_argument_parser, launch, HookBase
from detectron2.solver import build_optimizer
from peft import LoraConfig, get_peft_model, inject_adapter_in_model, LoraModel, PeftModel, cast_mixed_precision_params
import torch
from ray.tune.schedulers import ASHAScheduler
from torch.nn.parallel import DistributedDataParallel
from ray import tune, train



from train_net import Trainer, setup

def main(args):
    cfg = setup(args)
    config = {
        "lr": tune.loguniform(1e-6,1e-4)
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
    )
    res = tune.run(
        partial(train_ray,cfg=cfg),
        resources_per_trial={"cpu": 8, "gpu": 2},
        config=config,
        num_samples=2,
        scheduler=scheduler,
    )
    best = res.get_best_trial("loss","min", "last")
    print(best.config)
    best_ckp = res.get_best_checkpoint(best,"loss","min").as_directory()
    print("Checkpoint dir: " + str(best_ckp))
    return

class My_Hook(HookBase):
    def after_step(self):
        loss = self.trainer.storage.history("total_loss")
        train.report({"loss": loss})

def train_ray(config, cfg=None):

    cfg.SOLVER.BASE_LR = config["lr"]

    trainer = Trainer(cfg)
    trainer.register_hooks([My_Hook()])
    trainer.resume_or_load()
    trainer.train()
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

