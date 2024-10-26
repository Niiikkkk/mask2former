import numpy
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser
from component_metric import get_threshold_from_PRC, segment_metrics, default_instancer, anomaly_instances_from_mask

from train_net import Trainer, setup

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS,resume=args.resume)
    model.eval()
    # res = Trainer.test(cfg,model)
    for d in cfg.DATASETS.TEST:
        data_loader = Trainer.build_test_loader(cfg,d)
        for input in data_loader:
            gt = input[0]["sem_seg"].detach().cpu().numpy()
            output = model(input)[0]["sem_seg"].unsqueeze(0)
            pred = torch.max(output,axis=1)[1]
            pred = pred.detach().cpu().numpy().squeeze()
            print(numpy.unique(pred,return_counts=True))
            gt_instances , pred_instances =  anomaly_instances_from_mask(pred,gt)
            metric = segment_metrics(gt_instances,pred_instances)
            print(metric)
            break

