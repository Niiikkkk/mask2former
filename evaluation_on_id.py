from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser

from train_net import Trainer, setup

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS,resume=args.resume)
    res = Trainer.test(cfg,model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    print(res)
