import argparse
import logging
import os.path
import sys
from glob import glob

import matplotlib.pyplot as plt
import torch
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from detectron2.projects.deeplab import add_deeplab_config
import cv2
import numpy as np
from detectron2.utils.logger import setup_logger
from peft import LoraConfig
import detectron2.utils.comm as comm

from evaluation_on_ood import func

from mask2former import add_maskformer2_config
from fine_tune_LoRA import main
from train_net import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Stuff')
    parser.add_argument("--config-file",
        default="configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_TMP.yaml",
        metavar="FILE",
        help="path to config file",)
    parser.add_argument("--input",
        default="/Users/nicholas.berardo/Desktop/RoadAnomaly/images/*.jpg",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",)
    parser.add_argument("--output",
        default="/Users/nicholas.berardo/Desktop/RoadAnomaly/output.jpg",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",)

    args = parser.parse_args()
    return args

def setup_cfgs(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    # if args.output:
    #     cfg.OUTPUT_DIR = args.output
    # if args.weights:
    #     cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    return cfg


colors = [
        #[  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

# makes a dictionary with key:value. For example 0:[128, 64, 128]
label_colours = dict(zip(range(19), colors))

def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def print_img(image_to_plot,path_to_save):
    print(image_to_plot.shape, path_to_save)
    sys.stdout.flush()
    plt.axis("off")
    plt.tight_layout()
    if "image" in path_to_save or "label" in path_to_save:
        plt.imshow(cv2.cvtColor(image_to_plot, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(decode_segmap(image_to_plot[0]))
    plt.savefig(path_to_save)
    plt.clf()

def draw_prediction(model, img_paths, img_out, ssl_name):
    os.makedirs(os.path.join(img_out, ssl_name),exist_ok=True)
    for num,img_path in enumerate(tqdm.tqdm(img_paths)):
        image = read_image(img_path, format="BGR")
        prediction = model(image)["sem_seg"].unsqueeze(0)
        prediction_img = torch.max(prediction,dim=1)[1].detach().cpu().numpy()
        save_prediction_path = os.path.join(img_out, ssl_name, "prediction_" + str(num) + ".png")
        print_img(prediction_img,save_prediction_path)
        save_image_path = os.path.join(img_out, ssl_name, "image_" + str(num) + ".png")
        print_img(image,save_image_path)
        pathGT = img_path.replace("leftImg8bit", "gtFine")
        pathGT = pathGT.replace("gtFine.png", "gtFine_color.png")
        label = read_image(pathGT, format="BGR")
        save_label_path = os.path.join(img_out, ssl_name, "label_" + str(num) + ".png")
        print_img(label,save_label_path)

def ood():
    args = parse_args()
    cfg = setup_cfgs(args)

    cfg.defrost()
    inputs = [
        "/home/nberardo/Datasets/RoadAnomaly21/images/*.png",
        "/home/nberardo/Datasets/RoadObsticle21/images/*.webp",
        "/home/nberardo/Datasets/RoadAnomaly/images/*.jpg",
        "/home/nberardo/Datasets/FS_LostFound_full/images/*.png",
        "/home/nberardo/Datasets/fs_static/images/*.jpg"
    ]

    models = [
        "moco_v2_downloaded_2000_4e-05",
        "moco_v2_downloaded_4000_4e-05",
        "moco_v2_downloaded_6000_4e-05",
        "moco_v2_downloaded_8000_4e-05",
        "moco_v2_downloaded_2000_6e-05",
        "moco_v2_downloaded_4000_6e-05",
        "moco_v2_downloaded_6000_6e-05",
        "moco_v2_downloaded_8000_6e-05",
        "moco_v2_downloaded_2000_8e-05",
        "moco_v2_downloaded_4000_8e-05",
        "moco_v2_downloaded_6000_8e-05",
        "moco_v2_downloaded_8000_8e-05",
    ]

    for input in inputs:
        args.input = input
        for model in models:
            cfg.MODEL.WEIGHTS = os.path.join("/home/nberardo/mask2former/output/LP_FT", model, "model_final.pth")
            cfg.OUTPUT_DIR = os.path.join("/home/nberardo/mask2former/results/", model)
            model = DefaultPredictor(cfg)
            func(model, args, cfg)

def get_lora_config_backbone_only():
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules= r"backbone\.res\d\.\d\.conv\d",
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["sem_seg_head.predictor.mask_embed",
                         "sem_seg_head.pixel_decoder.input_proj.0",
                         "sem_seg_head.pixel_decoder.input_proj.1",
                         "sem_seg_head.pixel_decoder.input_proj.2",
                         "sem_seg_head.predictor.query_embed",
                         "sem_seg_head.predictor.query_feat",
                         "sem_seg_head.predictor.class_embed", ],)
        # query_embed, query_feat, class_embed, mask

    return lora_cfg

def get_lora_config_backbone_only_noOQ():
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules= r"backbone\.res\d\.\d\.conv\d",
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["sem_seg_head.predictor.mask_embed",
                         "sem_seg_head.pixel_decoder.input_proj.0",
                         "sem_seg_head.pixel_decoder.input_proj.1",
                         "sem_seg_head.pixel_decoder.input_proj.2",],)
        # query_embed, query_feat, class_embed, mask

    return lora_cfg

def get_lora_config_predictor_only():
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules= r"sem_seg_head\.predictor\.transformer_cross_attention_layers\.\d\.multihead_attn\.\w+"
                       r"|sem_seg_head\.predictor\.transformer_self_attention_layers\.\d\.self_attn\.\w+"
                       r"|sem_seg_head\.predictor\.transformer_ffn_layers\.\d\.linear.+",
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["sem_seg_head.predictor.mask_embed",
                         "sem_seg_head.pixel_decoder.input_proj.0",
                         "sem_seg_head.pixel_decoder.input_proj.1",
                         "sem_seg_head.pixel_decoder.input_proj.2",
                         "sem_seg_head.predictor.query_embed",
                         "sem_seg_head.predictor.query_feat",
                         "sem_seg_head.predictor.class_embed", ],)
        # query_embed, query_feat, class_embed, mask
    return lora_cfg

def get_lora_config_predictor_and_backbone():
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules= r"backbone\.res\d\.\d\.conv\d"
                       r"|sem_seg_head\.predictor\.transformer_cross_attention_layers\.\d\.multihead_attn\.\w+"
                       r"|sem_seg_head\.predictor\.transformer_self_attention_layers\.\d\.self_attn\.\w+"
                       r"|sem_seg_head\.predictor\.transformer_ffn_layers\.\d\.linear.+",
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["sem_seg_head.predictor.mask_embed",
                         "sem_seg_head.pixel_decoder.input_proj.0",
                         "sem_seg_head.pixel_decoder.input_proj.1",
                         "sem_seg_head.pixel_decoder.input_proj.2",
                         "sem_seg_head.predictor.query_embed",
                         "sem_seg_head.predictor.query_feat",
                         "sem_seg_head.predictor.class_embed", ],)
        # query_embed, query_feat, class_embed, mask
    return lora_cfg

def get_lora_config_predictor_only_noFFN():
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules= r"sem_seg_head\.predictor\.transformer_cross_attention_layers\.\d\.multihead_attn\.\w+"
                       r"|sem_seg_head\.predictor\.transformer_self_attention_layers\.\d\.self_attn\.\w+",
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["sem_seg_head.predictor.mask_embed",
                         "sem_seg_head.pixel_decoder.input_proj.0",
                         "sem_seg_head.pixel_decoder.input_proj.1",
                         "sem_seg_head.pixel_decoder.input_proj.2",
                         "sem_seg_head.predictor.query_embed",
                         "sem_seg_head.predictor.query_feat",
                         "sem_seg_head.predictor.class_embed", ],)
        # query_embed, query_feat, class_embed, mask
    return lora_cfg

def get_lora_config_predictor_only_noFFN_no_OQ():
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules= r"sem_seg_head\.predictor\.transformer_cross_attention_layers\.\d\.multihead_attn\.\w+"
                       r"|sem_seg_head\.predictor\.transformer_self_attention_layers\.\d\.self_attn\.\w+",
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["sem_seg_head.predictor.mask_embed",
                         "sem_seg_head.pixel_decoder.input_proj.0",
                         "sem_seg_head.pixel_decoder.input_proj.1",
                         "sem_seg_head.pixel_decoder.input_proj.2",],)
        # query_embed, query_feat, class_embed, mask
    return lora_cfg

def id_lora(args):
    cfg = setup_cfgs(args)
    cfg.defrost()

    lrs = [8e-5, 6e-5]
    max_iters = [2000]
    lora_configs = [
        {"name" : "backbone_only",
            "lora_cfg" : get_lora_config_backbone_only()},
        {"name" : "backbone_only_noOQ",
            "lora_cfg" : get_lora_config_backbone_only_noOQ()},
        {"name" : "predictor_only",
            "lora_cfg" : get_lora_config_predictor_only()},
        {"name" : "predictor_and_backbone",
            "lora_cfg" : get_lora_config_predictor_and_backbone()},
        {"name" : "predictor_only_noFFN",
            "lora_cfg" : get_lora_config_predictor_only_noFFN()},
        {"name" : "predictor_only_noFFN_noOQ",
            "lora_cfg" : get_lora_config_predictor_only_noFFN_no_OQ()}
    ]

    for lr in lrs:
        for max_iter in max_iters:
            for lora in lora_configs:

                cfg.OUTPUT_DIR = cfg.MODEL.WEIGHTS.replace("train","LORA").replace("model_final.pth","")
                model_name = cfg.OUTPUT_DIR.split("/")[-2]
                old_name = model_name
                model_name = model_name + "_" + str(max_iter) + "_" + str(lr) + "_" + lora["name"]
                cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace(old_name,model_name)
                cfg.SOLVER.BASE_LR = lr
                cfg.SOLVER.MAX_ITER = max_iter

                #Remove all loggers, that are created previously...
                loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
                for log in loggers:
                    log.handlers.clear()

                #Finish the setup of cfgs, loggers...
                default_setup(cfg, args)
                setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")

                main(args,cfg,lora["lora_cfg"])

def id_lp_ft(args):
    cfg = setup_cfgs(args)
    cfg.defrost()

    lrs = [8e-5, 6e-5, 4e-5]
    max_iters = [2000,4000,6000,8000]

    for lr in lrs:
        for max_iter in max_iters:
            cfg.OUTPUT_DIR = cfg.MODEL.WEIGHTS.replace("train","LP_FT").replace("model_final.pth","")
            model_name = cfg.OUTPUT_DIR.split("/")[-2]
            old_name = model_name
            model_name = model_name + "_" + str(max_iter) + "_" + str(lr)
            cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace(old_name,model_name)
            cfg.SOLVER.BASE_LR = lr
            cfg.SOLVER.MAX_ITER = max_iter

            #Remove all loggers, that are created previously...
            loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
            for log in loggers:
                log.handlers.clear()

            #Finish the setup of cfgs, loggers...
            default_setup(cfg, args)
            setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")

            #Training
            sys.stderr = open(os.path.join(cfg.OUTPUT_DIR, "stderr.txt"), "w")
            trainer = Trainer(cfg)
            trainer.resume_or_load(resume=args.resume)
            trainer.train()


if __name__ == "__main__":
    ood()

    #COMMENT OUT IF RUNNING ID
    # args = default_argument_parser().parse_args()
    # launch(
    #     # id_lora,
    #     id_lp_ft,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )


