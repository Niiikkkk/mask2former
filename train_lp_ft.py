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

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)

    layers_to_freeze = [0,1,2,3]
    layer_names = ["res2", "res3", "res4", "res5"]
    cfg = setup(args)

    #Create a model and do print it in order to get where the freeze is happening
    model = Trainer.build_model(cfg)
    for layer in layers_to_freeze:
        for param in getattr(model.backbone, layer_names[layer]).parameters():
            param.requires_grad = False
    for name, param in model.named_parameters():
        print(name + " -> " + str(param.requires_grad))

