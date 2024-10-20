from train_net import Trainer
from train_net import setup

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

class My_Trainer(Trainer):
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        layers_to_freeze = [0, 1, 2, 3, 4]
        layer_names = ["stem", "res2", "res3", "res4", "res5"]

        for layer in layers_to_freeze:
            if layer == 0:
                getattr(model.backbone, layer_names[layer]).freeze()
            else:
                for child in getattr(model.backbone, layer_names[layer]).children():
                    child.freeze()

        return model

def main(args):
    cfg = setup(args)

    # Create a model and do print it in order to get where the freeze is happening
    my_trainer = My_Trainer(cfg)

    print(format(my_trainer._trainer.model))

    my_trainer.resume_or_load(resume=args.resume)
    return my_trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )



