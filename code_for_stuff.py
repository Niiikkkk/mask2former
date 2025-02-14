import argparse
import logging
import os.path
import sys
from glob import glob
import re
import numpy as np

import matplotlib.pyplot as plt
import torch
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from detectron2.projects.deeplab import add_deeplab_config
import cv2

from detectron2.utils.logger import setup_logger
from peft import LoraConfig, get_peft_model
import detectron2.utils.comm as comm

from evaluation_on_ood import func

from mask2former import add_maskformer2_config
from fine_tune_LoRA import main, print_trainable_params, print_named_modules
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

    ]

    for input in inputs:
        args.input = input
        for model in models:
            cfg.MODEL.WEIGHTS = os.path.join("/home/nberardo/mask2former/output/LP_FT", model, "model_final.pth")
            cfg.OUTPUT_DIR = os.path.join("/home/nberardo/mask2former/results_LORA/", model)
            model = DefaultPredictor(cfg)
            func(model, args, cfg)

def ood_lora():
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
        "vicreg_down_freeze_2000_6e-05_backbone_only",
        "vicreg_down_freeze_2000_6e-05_backbone_only_noOQ",
        "vicreg_down_freeze_2000_6e-05_predictor_only",
        "vicreg_down_freeze_2000_6e-05_predictor_and_backbone",
        "vicreg_down_freeze_2000_6e-05_predictor_only_noFFN",
        "vicreg_down_freeze_2000_6e-05_predictor_only_noFFN_noOQ",
        "vicreg_down_freeze_2000_8e-05_backbone_only",
        "vicreg_down_freeze_2000_8e-05_backbone_only_noOQ",
        "vicreg_down_freeze_2000_8e-05_predictor_only",
        "vicreg_down_freeze_2000_8e-05_predictor_and_backbone",
        "vicreg_down_freeze_2000_8e-05_predictor_only_noFFN",
        "vicreg_down_freeze_2000_8e-05_predictor_only_noFFN_noOQ",
        "vicreg_down_freeze_4000_6e-05_backbone_only",
        "vicreg_down_freeze_4000_6e-05_backbone_only_noOQ",
        "vicreg_down_freeze_4000_6e-05_predictor_only",
        "vicreg_down_freeze_4000_6e-05_predictor_and_backbone",
        "vicreg_down_freeze_4000_6e-05_predictor_only_noFFN",
        "vicreg_down_freeze_4000_6e-05_predictor_only_noFFN_noOQ",
        "vicreg_down_freeze_4000_8e-05_backbone_only",
        "vicreg_down_freeze_4000_8e-05_backbone_only_noOQ",
        "vicreg_down_freeze_4000_8e-05_predictor_only",
        "vicreg_down_freeze_4000_8e-05_predictor_and_backbone",
        "vicreg_down_freeze_4000_8e-05_predictor_only_noFFN",
        "vicreg_down_freeze_4000_8e-05_predictor_only_noFFN_noOQ",
        "vicreg_down_freeze_8000_6e-05_backbone_only",
        "vicreg_down_freeze_8000_6e-05_backbone_only_noOQ",
        "vicreg_down_freeze_8000_6e-05_predictor_only",
        "vicreg_down_freeze_8000_6e-05_predictor_and_backbone",
        "vicreg_down_freeze_8000_6e-05_predictor_only_noFFN",
        "vicreg_down_freeze_8000_6e-05_predictor_only_noFFN_noOQ",
        "vicreg_down_freeze_8000_8e-05_backbone_only",
        "vicreg_down_freeze_8000_8e-05_backbone_only_noOQ",
        "vicreg_down_freeze_8000_8e-05_predictor_only",
        "vicreg_down_freeze_8000_8e-05_predictor_and_backbone",
        "vicreg_down_freeze_8000_8e-05_predictor_only_noFFN",
        "vicreg_down_freeze_8000_8e-05_predictor_only_noFFN_noOQ",
    ]

    for input in inputs:
        args.input = input
        for model in models:
            cfg.MODEL.WEIGHTS = os.path.join("/home/nberardo/mask2former/output/train", "vicreg_down_freeze", "model_final.pth")
            cfg.OUTPUT_DIR = os.path.join("/home/nberardo/mask2former/results_LORA/", model)
            predictor = DefaultPredictor(cfg)
            lora_path = os.path.join("/home/nberardo/mask2former/output/LORA", model, "lora_model")
            print(f"Loading LORA model from {lora_path}")
            lora_config = LoraConfig.from_pretrained(lora_path)
            inference_model = get_peft_model(predictor.model, lora_config)
            inference_model.load_state_dict(torch.load(lora_path + "/model.pth"), strict=False)
            predictor.model = inference_model
            func(predictor, args, cfg)

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

def get_lora_config_all():
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=r"sem_seg_head\.pixel_decoder\.
        target_modules=
        r"sem_seg_head\.pixel_decoder\.transformer\.encoder\.layers\.\d\.self_attn\.\w+"
        r"|sem_seg_head\.pixel_decoder\.transformer\.encoder\.layers\.\d\.linear\d"
        r"|backbone\.res\d\.\d\.conv\d"
        r"|backbone\.stem\.conv\d"
        r"|backbone\.res\d\.\d\.shortcut"
        r"|sem_seg_head\.predictor\.transformer_ffn_layers\.\d\.linear.+"
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
                         "sem_seg_head.predictor.class_embed",
                         "sem_seg_head.predictor.level_embed"],
    )
    return lora_cfg

def id_lora(args):
    cfg = setup_cfgs(args)
    cfg.defrost()

    lrs = [8e-5, 6e-5]
    max_iters = [2000,4000,8000]
    lora_configs = [
        {"name" : "all",
            "lora_cfg" : get_lora_config_all()},
        # {"name" : "backbone_only",
        #     "lora_cfg" : get_lora_config_backbone_only()},
        # {"name" : "backbone_only_noOQ",
        #     "lora_cfg" : get_lora_config_backbone_only_noOQ()},
        # {"name" : "predictor_only",
        #     "lora_cfg" : get_lora_config_predictor_only()},
        # {"name" : "predictor_and_backbone",
        #     "lora_cfg" : get_lora_config_predictor_and_backbone()},
        # {"name" : "predictor_only_noFFN",
        #     "lora_cfg" : get_lora_config_predictor_only_noFFN()},
        # {"name" : "predictor_only_noFFN_noOQ",
        #     "lora_cfg" : get_lora_config_predictor_only_noFFN_no_OQ()}
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

def test():

    ra21 = """
        bt-down-freezed & 2000 & 4e-05 & 91.94 & 48.70 & 69.86 & 59.17 & 34.48 & 41.02 \\
        bt-down-freezed & 2000 & 6e-05 & 91.23 & 58.25 & 67.17 & 50.19 & 20.27 & 58.25 \\
        bt-down-freezed & 2000 & 8e-05 & 83.44 & 72.22 & 53.02 & 59.47 & 22.73 & 45.31 \\
        bt-down-freezed & 4000 & 4e-05 & 86.30 & 64.82 & 55.86 & 52.89 & 20.25 & 27.20 \\
        bt-down-freezed & 4000 & 6e-05 & 91.73 & 52.10 & 66.21 & 59.06 & 31.73 & 42.61 \\
        bt-down-freezed & 4000 & 8e-05 & 90.27 & 40.49 & 54.68 & 63.97 & 29.26 & 65.91 \\
        bt-down-freezed & 6000 & 4e-05 & 90.82 & 50.14 & 57.05 & 51.71 & 24.76 & 66.22 \\
        bt-down-freezed & 6000 & 6e-05 & 88.16 & 55.39 & 56.25 & 45.46 & 16.63 & 63.09 \\
        bt-down-freezed & 6000 & 8e-05 & 86.36 & 47.35 & 48.85 & 52.77 & 28.30 & 60.36 \\
        bt-down-freezed & 8000 & 4e-05 & 91.60 & 37.67 & 61.02 & 67.09 & 33.27 & 66.76 \\
        bt-down-freezed & 8000 & 6e-05 & 90.79 & 50.20 & 58.96 & 66.40 & 29.90 & 65.38 \\
        bt-down-freezed & 8000 & 8e-05 & 92.03 & 14.43 & 62.05 & 62.57 & 36.08 & 68.33 \\
        dino-down-freezed & 2000 & 4e-05 & 86.96 & 63.51 & 63.25 & 55.84 & 30.53 & 28.31 \\
        dino-down-freezed & 2000 & 6e-05 & 84.80 & 82.56 & 60.69 & 51.08 & 25.82 & 18.78 \\
        dino-down-freezed & 2000 & 8e-05 & 82.35 & 58.08 & 43.34 & 53.63 & 22.80 & 46.12 \\
        dino-down-freezed & 4000 & 4e-05 & 90.85 & 55.20 & 76.39 & 59.21 & 28.83 & 40.71 \\
        dino-down-freezed & 4000 & 6e-05 & 89.41 & 63.42 & 66.98 & 57.65 & 24.12 & 57.79 \\
        dino-down-freezed & 4000 & 8e-05 & 84.32 & 67.26 & 61.61 & 50.82 & 29.52 & 42.38 \\
        dino-down-freezed & 6000 & 4e-05 & 85.36 & 62.75 & 65.88 & 56.66 & 34.46 & 45.19 \\
        dino-down-freezed & 6000 & 6e-05 & 88.14 & 54.68 & 67.94 & 53.30 & 28.56 & 29.86 \\
        dino-down-freezed & 6000 & 8e-05 & 86.90 & 58.48 & 68.90 & 52.19 & 24.22 & 34.18 \\
        dino-down-freezed & 8000 & 4e-05 & 88.51 & 65.44 & 70.31 & 52.61 & 30.95 & 45.25 \\
        dino-down-freezed & 8000 & 6e-05 & 91.38 & 58.60 & 77.89 & 58.13 & 32.96 & 43.72 \\
        dino-down-freezed & 8000 & 8e-05 & 88.94 & 65.32 & 75.18 & 54.05 & 33.71 & 40.76 \\
        vicreg\_down\_freeze & 2000 & 4e-05 & 79.83 & 76.69 & 53.88 & 63.42 & 34.99 & 38.31 \\
        vicreg\_down\_freeze & 2000 & 6e-05 & 87.96 & 56.46 & 65.26 & 66.50 & 42.11 & 48.70 \\
        vicreg\_down\_freeze & 2000 & 8e-05 & 89.18 & 49.73 & 66.99 & 64.41 & 36.77 & 52.88 \\
        vicreg\_down\_freeze & 4000 & 4e-05 & 86.21 & 61.67 & 59.51 & 62.84 & 35.81 & 43.07 \\
        vicreg\_down\_freeze & 4000 & 6e-05 & 85.38 & 67.37 & 61.17 & 64.29 & 30.45 & 46.80 \\
        vicreg\_down\_freeze & 4000 & 8e-05 & 88.43 & 74.00 & 68.22 & 65.18 & 36.48 & 41.51 \\
        vicreg\_down\_freeze & 6000 & 4e-05 & 89.27 & 41.81 & 64.82 & 69.59 & 31.52 & 55.95 \\
        vicreg\_down\_freeze & 6000 & 6e-05 & 88.43 & 61.87 & 64.16 & 65.22 & 42.85 & 49.08 \\
        vicreg\_down\_freeze & 6000 & 8e-05 & 83.68 & 63.21 & 52.05 & 66.07 & 31.03 & 56.84 \\
        vicreg\_down\_freeze & 8000 & 4e-05 & 87.39 & 55.87 & 62.94 & 68.48 & 45.07 & 48.13 \\
        vicreg\_down\_freeze & 8000 & 6e-05 & 90.02 & 52.19 & 66.13 & 68.34 & 41.31 & 54.68 \\
        vicreg\_down\_freeze & 8000 & 8e-05 & 89.71 & 53.56 & 65.99 & 68.13 & 45.34 & 39.06 \\
        simsiam\_freezed & 2000 & 4e-05 & 93.18 & 13.98 & 76.64 & 68.69 & 39.96 & 58.27 \\
        simsiam\_freezed & 2000 & 6e-05 & 90.75 & 44.12 & 71.35 & 57.56 & 42.60 & 61.33 \\
        simsiam\_freezed & 2000 & 8e-05 & 91.04 & 48.48 & 69.88 & 65.45 & 32.18 & 56.53 \\
        simsiam\_freezed & 4000 & 4e-05 & 92.63 & 22.60 & 72.65 & 67.61 & 43.44 & 64.20 \\
        simsiam\_freezed & 4000 & 6e-05 & 91.32 & 30.41 & 70.46 & 64.35 & 34.79 & 58.11 \\
        simsiam\_freezed & 4000 & 8e-05 & 92.49 & 19.24 & 70.85 & 66.28 & 50.75 & 61.50 \\
        simsiam\_freezed & 6000 & 4e-05 & 92.54 & 20.37 & 74.91 & 66.71 & 63.62 & 56.75 \\
        simsiam\_freezed & 6000 & 6e-05 & 93.10 & 21.70 & 75.01 & 69.68 & 45.15 & 64.00 \\
        simsiam\_freezed & 6000 & 8e-05 & 93.07 & 18.77 & 76.72 & 68.01 & 55.58 & 62.24 \\
        simsiam\_freezed & 8000 & 4e-05 & 92.08 & 20.31 & 72.08 & 66.47 & 34.78 & 61.57 \\
        simsiam\_freezed & 8000 & 6e-05 & 93.26 & 16.75 & 74.99 & 68.12 & 53.39 & 58.19 \\
        simsiam\_freezed & 8000 & 8e-05 & 94.10 & 15.57 & 75.58 & 72.91 & 53.73 & 68.10 \\
        moco\_v1\_downloaded & 2000 & 4e-05 & 89.94 & 31.51 & 70.76 & 42.24 & 38.22 & 43.70 \\
        moco\_v1\_downloaded & 2000 & 6e-05 & 89.82 & 59.36 & 73.18 & 44.62 & 33.76 & 32.73 \\
        moco\_v1\_downloaded & 2000 & 8e-05 & 84.52 & 61.56 & 61.13 & 41.99 & 34.14 & 43.36 \\
        moco\_v1\_downloaded & 4000 & 4e-05 & 90.14 & 27.98 & 72.33 & 41.49 & 33.61 & 53.07 \\
        moco\_v1\_downloaded & 4000 & 6e-05 & 87.70 & 71.09 & 67.74 & 41.05 & 35.94 & 42.80 \\
        moco\_v1\_downloaded & 4000 & 8e-05 & 88.37 & 48.61 & 63.04 & 38.58 & 35.62 & 58.09 \\
        moco\_v1\_downloaded & 6000 & 4e-05 & 91.61 & 41.46 & 73.11 & 45.16 & 30.30 & 56.94 \\
        moco\_v1\_downloaded & 6000 & 6e-05 & 86.43 & 64.42 & 65.75 & 38.29 & 33.36 & 43.76 \\
        moco\_v1\_downloaded & 6000 & 8e-05 & 87.36 & 70.39 & 67.29 & 38.40 & 35.21 & 34.27 \\
        moco\_v1\_downloaded & 8000 & 4e-05 & 88.41 & 65.15 & 69.36 & 43.45 & 36.26 & 46.19 \\
        moco\_v1\_downloaded & 8000 & 6e-05 & 94.54 & 14.89 & 78.00 & 43.53 & 33.21 & 63.95 \\
        moco\_v1\_downloaded & 8000 & 8e-05 & 85.64 & 52.22 & 67.51 & 43.53 & 31.84 & 44.29 \\
        moco\_v2\_downloaded & 2000 & 4e-05 & 85.48 & 67.76 & 63.24 & 54.47 & 30.40 & 41.69 \\
        moco\_v2\_downloaded & 2000 & 6e-05 & 81.54 & 72.49 & 63.63 & 58.16 & 30.22 & 20.56 \\
        moco\_v2\_downloaded & 2000 & 8e-05 & 89.33 & 59.06 & 70.19 & 58.82 & 48.12 & 54.77 \\
        moco\_v2\_downloaded & 4000 & 4e-05 & 87.56 & 64.79 & 64.46 & 49.64 & 30.00 & 28.79 \\
        moco\_v2\_downloaded & 4000 & 6e-05 & 92.31 & 44.24 & 76.76 & 55.09 & 46.31 & 53.06 \\
        moco\_v2\_downloaded & 4000 & 8e-05 & 87.91 & 57.95 & 66.48 & 53.69 & 40.67 & 51.05 \\
        moco\_v2\_downloaded & 6000 & 4e-05 & 89.20 & 32.95 & 73.25 & 58.94 & 57.46 & 53.73 \\
        moco\_v2\_downloaded & 6000 & 6e-05 & 87.93 & 52.99 & 70.44 & 59.49 & 47.90 & 26.07 \\
        moco\_v2\_downloaded & 6000 & 8e-05 & 89.97 & 56.03 & 74.15 & 50.87 & 46.12 & 47.41 \\
        moco\_v2\_downloaded & 8000 & 4e-05 & 88.15 & 69.02 & 69.96 & 55.70 & 40.85 & 46.94 \\
        moco\_v2\_downloaded & 8000 & 6e-05 & 83.30 & 77.94 & 66.12 & 54.24 & 33.46 & 15.77 \\
        moco\_v2\_downloaded & 8000 & 8e-05 & 92.23 & 47.13 & 77.20 & 59.40 & 46.38 & 57.14 \\
    """
    ro21 = """bt-down-freezed & 2000 & 4e-05 & 87.24 & 77.53 & 56.19 & 39.53 & 38.43 & 34.46 \\
        bt-down-freezed & 2000 & 6e-05 & 93.98 & 67.96 & 65.15 & 41.81 & 35.91 & 37.44 \\
        bt-down-freezed & 2000 & 8e-05 & 97.07 & 7.87 & 74.53 & 44.49 & 50.50 & -17.31 \\
        bt-down-freezed & 4000 & 4e-05 & 94.10 & 66.28 & 70.39 & 43.02 & 42.24 & 52.33 \\
        bt-down-freezed & 4000 & 6e-05 & 95.73 & 18.78 & 71.41 & 43.67 & 44.31 & 34.40 \\
        bt-down-freezed & 4000 & 8e-05 & 91.32 & 54.12 & 53.91 & 42.39 & 34.93 & 71.91 \\
        bt-down-freezed & 6000 & 4e-05 & 96.35 & 13.16 & 56.88 & 42.22 & 35.35 & 13.27 \\
        bt-down-freezed & 6000 & 6e-05 & 93.11 & 78.04 & 73.07 & 40.72 & 38.90 & 20.87 \\
        bt-down-freezed & 6000 & 8e-05 & 96.85 & 12.78 & 64.72 & 47.62 & 42.45 & 90.71 \\
        bt-down-freezed & 8000 & 4e-05 & 96.04 & 14.03 & 58.80 & 41.67 & 35.73 & 30.24 \\
        bt-down-freezed & 8000 & 6e-05 & 95.61 & 29.30 & 73.84 & 42.73 & 45.87 & 60.56 \\
        bt-down-freezed & 8000 & 8e-05 & 91.46 & 75.54 & 71.45 & 47.19 & 44.39 & 82.43 \\
        dino-down-freezed & 2000 & 4e-05 & 97.60 & 4.06 & 62.40 & 41.86 & 54.68 & -15.25 \\
        dino-down-freezed & 2000 & 6e-05 & 96.05 & 12.22 & 36.72 & 40.41 & 39.34 & -10.43 \\
        dino-down-freezed & 2000 & 8e-05 & 91.31 & 76.32 & 42.43 & 37.74 & 35.33 & -39.81 \\
        dino-down-freezed & 4000 & 4e-05 & 96.17 & 12.45 & 36.57 & 44.89 & 62.11 & -11.59 \\
        dino-down-freezed & 4000 & 6e-05 & 97.10 & 8.85 & 33.36 & 39.87 & 52.67 & 24.42 \\
        dino-down-freezed & 4000 & 8e-05 & 95.73 & 36.83 & 68.35 & 43.99 & 43.89 & 1.21 \\
        dino-down-freezed & 6000 & 4e-05 & 94.83 & 15.97 & 26.43 & 40.39 & 50.64 & 6.19 \\
        dino-down-freezed & 6000 & 6e-05 & 95.97 & 8.04 & 50.49 & 40.05 & 57.04 & -10.57 \\
        dino-down-freezed & 6000 & 8e-05 & 97.55 & 7.22 & 65.08 & 40.38 & 44.94 & -0.19 \\
        dino-down-freezed & 8000 & 4e-05 & 96.09 & 5.23 & 61.78 & 40.89 & 34.79 & 38.27 \\
        dino-down-freezed & 8000 & 6e-05 & 85.54 & 76.79 & 42.76 & 40.41 & 42.35 & 18.57 \\
        dino-down-freezed & 8000 & 8e-05 & 97.07 & 5.12 & 62.50 & 44.76 & 58.34 & 41.66 \\
        vicreg\_down\_freeze & 2000 & 4e-05 & 94.81 & 55.41 & 74.97 & 45.28 & 41.99 & -14.02 \\
        vicreg\_down\_freeze & 2000 & 6e-05 & 94.12 & 52.13 & 72.73 & 42.99 & 39.29 & -31.58 \\
        vicreg\_down\_freeze & 2000 & 8e-05 & 95.92 & 40.63 & 82.84 & 45.48 & 57.85 & -68.70 \\
        vicreg\_down\_freeze & 4000 & 4e-05 & 95.54 & 33.86 & 79.38 & 43.09 & 52.62 & -62.67 \\
        vicreg\_down\_freeze & 4000 & 6e-05 & 95.53 & 41.67 & 76.59 & 44.76 & 42.81 & -8.13 \\
        vicreg\_down\_freeze & 4000 & 8e-05 & 95.92 & 48.14 & 78.74 & 42.72 & 47.91 & -24.97 \\
        vicreg\_down\_freeze & 6000 & 4e-05 & 96.43 & 39.00 & 77.35 & 46.08 & 48.63 & -1.76 \\
        vicreg\_down\_freeze & 6000 & 6e-05 & 96.75 & 26.58 & 74.87 & 45.96 & 43.33 & 11.80 \\
        vicreg\_down\_freeze & 6000 & 8e-05 & 96.30 & 33.48 & 76.89 & 44.42 & 51.04 & 7.36 \\
        vicreg\_down\_freeze & 8000 & 4e-05 & 96.48 & 25.31 & 80.64 & 45.21 & 47.27 & -9.24 \\
        vicreg\_down\_freeze & 8000 & 6e-05 & 97.01 & 15.53 & 73.77 & 46.78 & 38.91 & -41.32 \\
        vicreg\_down\_freeze & 8000 & 8e-05 & 96.35 & 38.45 & 73.34 & 45.58 & 42.41 & 12.88 \\
        simsiam\_freezed & 2000 & 4e-05 & 95.65 & 18.83 & 78.15 & 46.41 & 38.49 & 17.12 \\
        simsiam\_freezed & 2000 & 6e-05 & 97.09 & 5.58 & 77.23 & 46.16 & 43.91 & 49.96 \\
        simsiam\_freezed & 2000 & 8e-05 & 92.99 & 83.92 & 72.53 & 46.79 & 43.18 & 20.77 \\
        simsiam\_freezed & 4000 & 4e-05 & 97.65 & 10.74 & 81.23 & 45.47 & 41.66 & 66.33 \\
        simsiam\_freezed & 4000 & 6e-05 & 97.08 & 5.92 & 77.04 & 44.34 & 46.16 & 59.24 \\
        simsiam\_freezed & 4000 & 8e-05 & 96.93 & 5.41 & 82.28 & 44.37 & 41.23 & 42.96 \\
        simsiam\_freezed & 6000 & 4e-05 & 95.91 & 13.21 & 77.62 & 46.28 & 49.44 & 35.16 \\
        simsiam\_freezed & 6000 & 6e-05 & 98.64 & 2.31 & 83.46 & 45.80 & 54.89 & 94.45 \\
        simsiam\_freezed & 6000 & 8e-05 & 95.26 & 33.99 & 80.36 & 48.82 & 48.98 & 42.24 \\
        simsiam\_freezed & 8000 & 4e-05 & 97.26 & 4.70 & 82.27 & 47.52 & 45.23 & 32.22 \\
        simsiam\_freezed & 8000 & 6e-05 & 93.14 & 82.25 & 69.71 & 45.08 & 45.30 & 39.12 \\
        simsiam\_freezed & 8000 & 8e-05 & 98.37 & 5.08 & 77.89 & 46.80 & 39.61 & 91.15 \\
        moco\_v1\_downloaded & 2000 & 4e-05 & 90.52 & 42.21 & 48.06 & 49.36 & 50.97 & 63.62 \\
        moco\_v1\_downloaded & 2000 & 6e-05 & 94.56 & 29.92 & 64.94 & 40.62 & 44.77 & 53.98 \\
        moco\_v1\_downloaded & 2000 & 8e-05 & 83.80 & 59.36 & 20.49 & 40.85 & 36.61 & -20.93 \\
        moco\_v1\_downloaded & 4000 & 4e-05 & 95.63 & 24.77 & 66.04 & 44.41 & 40.64 & 57.42 \\
        moco\_v1\_downloaded & 4000 & 6e-05 & 93.52 & 38.41 & 67.15 & 46.00 & 47.62 & 59.58 \\
        moco\_v1\_downloaded & 4000 & 8e-05 & 97.37 & 7.67 & 68.23 & 47.31 & 41.19 & 65.26 \\
        moco\_v1\_downloaded & 6000 & 4e-05 & 90.53 & 57.78 & 62.48 & 43.36 & 42.22 & 59.69 \\
        moco\_v1\_downloaded & 6000 & 6e-05 & 91.35 & 36.61 & 49.09 & 47.07 & 50.84 & 67.22 \\
        moco\_v1\_downloaded & 6000 & 8e-05 & 95.72 & 29.96 & 68.66 & 43.94 & 43.84 & 61.10 \\
        moco\_v1\_downloaded & 8000 & 4e-05 & 95.54 & 28.59 & 69.40 & 47.33 & 48.91 & 41.26 \\
        moco\_v1\_downloaded & 8000 & 6e-05 & 93.86 & 35.34 & 62.86 & 48.72 & 46.61 & 77.42 \\
        moco\_v1\_downloaded & 8000 & 8e-05 & 96.77 & 9.94 & 67.83 & 47.29 & 48.87 & 73.85 \\
        moco\_v2\_downloaded & 2000 & 4e-05 & 90.94 & 65.61 & 55.04 & 45.10 & 41.98 & 6.13 \\
        moco\_v2\_downloaded & 2000 & 6e-05 & 92.98 & 64.04 & 58.83 & 46.59 & 37.44 & -14.26 \\
        moco\_v2\_downloaded & 2000 & 8e-05 & 95.56 & 8.86 & 40.72 & 41.72 & 37.94 & 18.79 \\
        moco\_v2\_downloaded & 4000 & 4e-05 & 91.02 & 72.74 & 44.11 & 43.20 & 54.81 & 18.17 \\
        moco\_v2\_downloaded & 4000 & 6e-05 & 96.00 & 15.71 & 42.68 & 48.36 & 53.12 & 55.32 \\
        moco\_v2\_downloaded & 4000 & 8e-05 & 96.09 & 10.40 & 49.97 & 51.95 & 62.36 & 36.90 \\
        moco\_v2\_downloaded & 6000 & 4e-05 & 94.73 & 15.91 & 49.36 & 46.85 & 49.41 & 83.28 \\
        moco\_v2\_downloaded & 6000 & 6e-05 & 97.40 & 6.86 & 52.64 & 48.77 & 45.79 & 56.58 \\
        moco\_v2\_downloaded & 6000 & 8e-05 & 97.87 & 3.44 & 71.64 & 45.20 & 51.18 & 79.76 \\
        moco\_v2\_downloaded & 8000 & 4e-05 & 94.11 & 26.32 & 45.82 & 47.03 & 46.24 & 50.54 \\
        moco\_v2\_downloaded & 8000 & 6e-05 & 94.73 & 33.44 & 41.28 & 45.59 & 49.30 & 91.23 \\
        moco\_v2\_downloaded & 8000 & 8e-05 & 95.70 & 20.77 & 77.02 & 48.28 & 63.96 & 5.61 \\"""
    ra = """bt-down-freezed & 2000 & 4e-05 & 72.50 & 73.76 & 25.08 & 29.52 & 25.42 & 43.15 \\
        bt-down-freezed & 2000 & 6e-05 & 79.79 & 63.00 & 31.55 & 26.76 & 24.87 & 44.94 \\
        bt-down-freezed & 2000 & 8e-05 & 69.67 & 83.60 & 23.30 & 22.17 & 22.73 & 37.81 \\
        bt-down-freezed & 4000 & 4e-05 & 74.38 & 79.62 & 27.72 & 29.67 & 28.52 & 47.73 \\
        bt-down-freezed & 4000 & 6e-05 & 76.56 & 71.72 & 27.46 & 23.62 & 21.46 & 49.65 \\
        bt-down-freezed & 4000 & 8e-05 & 71.85 & 77.50 & 23.42 & 26.28 & 28.59 & 43.27 \\
        bt-down-freezed & 6000 & 4e-05 & 73.38 & 80.66 & 23.84 & 23.55 & 21.84 & 44.54 \\
        bt-down-freezed & 6000 & 6e-05 & 72.65 & 84.96 & 20.66 & 22.39 & 20.60 & 47.73 \\
        bt-down-freezed & 6000 & 8e-05 & 70.75 & 77.03 & 22.61 & 28.08 & 25.59 & 44.66 \\
        bt-down-freezed & 8000 & 4e-05 & 76.88 & 61.16 & 26.13 & 26.85 & 25.50 & 54.16 \\
        bt-down-freezed & 8000 & 6e-05 & 75.87 & 75.33 & 24.99 & 24.87 & 27.00 & 46.73 \\
        bt-down-freezed & 8000 & 8e-05 & 78.35 & 68.48 & 27.98 & 27.59 & 26.39 & 53.14 \\
        dino-down-freezed & 2000 & 4e-05 & 76.15 & 75.62 & 25.80 & 30.36 & 32.18 & 42.12 \\
        dino-down-freezed & 2000 & 6e-05 & 73.14 & 78.15 & 23.72 & 31.12 & 30.16 & 35.16 \\
        dino-down-freezed & 2000 & 8e-05 & 67.40 & 83.67 & 19.97 & 30.13 & 31.46 & 28.66 \\
        dino-down-freezed & 4000 & 4e-05 & 76.76 & 78.79 & 25.47 & 31.79 & 29.16 & 49.12 \\
        dino-down-freezed & 4000 & 6e-05 & 78.03 & 80.22 & 28.45 & 29.40 & 32.39 & 51.02 \\
        dino-down-freezed & 4000 & 8e-05 & 70.65 & 82.17 & 23.06 & 29.50 & 32.45 & 45.26 \\
        dino-down-freezed & 6000 & 4e-05 & 74.01 & 80.59 & 23.10 & 28.32 & 28.94 & 48.35 \\
        dino-down-freezed & 6000 & 6e-05 & 74.41 & 88.37 & 25.27 & 31.43 & 30.95 & 40.63 \\
        dino-down-freezed & 6000 & 8e-05 & 71.07 & 86.85 & 21.58 & 29.14 & 26.19 & 45.44 \\
        dino-down-freezed & 8000 & 4e-05 & 72.37 & 82.51 & 24.83 & 29.38 & 28.09 & 40.21 \\
        dino-down-freezed & 8000 & 6e-05 & 74.26 & 87.06 & 25.78 & 29.66 & 29.63 & 46.49 \\
        dino-down-freezed & 8000 & 8e-05 & 78.51 & 76.79 & 30.56 & 31.54 & 29.40 & 45.48 \\
        vicreg\_down\_freeze & 2000 & 4e-05 & 76.69 & 85.31 & 32.12 & 31.20 & 34.06 & 42.70 \\
        vicreg\_down\_freeze & 2000 & 6e-05 & 79.79 & 62.61 & 35.53 & 29.18 & 28.66 & 52.62 \\
        vicreg\_down\_freeze & 2000 & 8e-05 & 77.94 & 71.35 & 38.35 & 29.73 & 32.65 & 46.72 \\
        vicreg\_down\_freeze & 4000 & 4e-05 & 76.80 & 74.66 & 31.59 & 30.42 & 30.11 & 45.64 \\
        vicreg\_down\_freeze & 4000 & 6e-05 & 81.01 & 68.31 & 36.85 & 32.05 & 31.31 & 53.95 \\
        vicreg\_down\_freeze & 4000 & 8e-05 & 74.96 & 74.32 & 31.88 & 31.59 & 30.32 & 45.97 \\
        vicreg\_down\_freeze & 6000 & 4e-05 & 74.55 & 88.43 & 31.86 & 32.16 & 32.17 & 47.08 \\
        vicreg\_down\_freeze & 6000 & 6e-05 & 73.24 & 72.50 & 30.19 & 29.57 & 27.96 & 45.97 \\
        vicreg\_down\_freeze & 6000 & 8e-05 & 77.39 & 85.61 & 30.38 & 30.54 & 28.65 & 50.37 \\
        vicreg\_down\_freeze & 8000 & 4e-05 & 75.17 & 87.18 & 32.28 & 32.49 & 32.34 & 48.59 \\
        vicreg\_down\_freeze & 8000 & 6e-05 & 77.76 & 75.48 & 31.81 & 27.75 & 24.96 & 50.63 \\
        vicreg\_down\_freeze & 8000 & 8e-05 & 76.94 & 76.06 & 30.69 & 29.06 & 26.82 & 48.69 \\
        simsiam\_freezed & 2000 & 4e-05 & 75.94 & 86.03 & 35.87 & 31.07 & 34.40 & 34.32 \\
        simsiam\_freezed & 2000 & 6e-05 & 75.37 & 86.26 & 35.52 & 30.94 & 33.44 & 29.17 \\
        simsiam\_freezed & 2000 & 8e-05 & 76.55 & 83.25 & 30.39 & 27.18 & 28.42 & 46.94 \\
        simsiam\_freezed & 4000 & 4e-05 & 77.49 & 83.85 & 36.93 & 32.20 & 32.19 & 43.41 \\
        simsiam\_freezed & 4000 & 6e-05 & 74.53 & 90.31 & 36.26 & 29.58 & 34.71 & 33.36 \\
        simsiam\_freezed & 4000 & 8e-05 & 74.87 & 90.93 & 33.57 & 28.30 & 33.25 & 40.66 \\
        simsiam\_freezed & 6000 & 4e-05 & 79.18 & 85.07 & 41.17 & 34.86 & 33.01 & 39.61 \\
        simsiam\_freezed & 6000 & 6e-05 & 77.78 & 88.98 & 42.14 & 31.98 & 38.56 & 40.31 \\
        simsiam\_freezed & 6000 & 8e-05 & 76.86 & 84.34 & 37.67 & 29.58 & 32.30 & 38.12 \\
        simsiam\_freezed & 8000 & 4e-05 & 76.02 & 84.41 & 31.17 & 30.28 & 32.78 & 46.00 \\
        simsiam\_freezed & 8000 & 6e-05 & 77.38 & 83.03 & 36.12 & 32.92 & 36.38 & 39.26 \\
        simsiam\_freezed & 8000 & 8e-05 & 78.84 & 84.70 & 40.92 & 32.75 & 34.79 & 48.37 \\
        moco\_v1\_downloaded & 2000 & 4e-05 & 81.04 & 83.08 & 48.91 & 27.45 & 32.43 & 35.96 \\
        moco\_v1\_downloaded & 2000 & 6e-05 & 80.58 & 78.02 & 52.73 & 27.74 & 31.59 & 35.24 \\
        moco\_v1\_downloaded & 2000 & 8e-05 & 77.56 & 75.56 & 41.94 & 23.94 & 29.00 & 42.76 \\
        moco\_v1\_downloaded & 4000 & 4e-05 & 82.66 & 80.68 & 52.11 & 27.50 & 34.02 & 25.77 \\
        moco\_v1\_downloaded & 4000 & 6e-05 & 81.34 & 80.31 & 52.13 & 30.27 & 38.33 & 39.82 \\
        moco\_v1\_downloaded & 4000 & 8e-05 & 79.29 & 80.04 & 54.26 & 25.45 & 29.29 & 29.56 \\
        moco\_v1\_downloaded & 6000 & 4e-05 & 78.24 & 82.99 & 52.57 & 33.40 & 33.52 & 25.53 \\
        moco\_v1\_downloaded & 6000 & 6e-05 & 79.08 & 85.25 & 50.31 & 27.42 & 30.30 & 30.04 \\
        moco\_v1\_downloaded & 6000 & 8e-05 & 82.77 & 80.68 & 49.88 & 29.52 & 33.32 & 46.14 \\
        moco\_v1\_downloaded & 8000 & 4e-05 & 83.68 & 79.37 & 57.95 & 27.65 & 35.02 & 34.04 \\
        moco\_v1\_downloaded & 8000 & 6e-05 & 81.81 & 81.10 & 49.81 & 29.22 & 36.23 & 31.16 \\
        moco\_v1\_downloaded & 8000 & 8e-05 & 80.47 & 86.06 & 55.48 & 32.01 & 36.14 & 33.38 \\
        moco\_v2\_downloaded & 2000 & 4e-05 & 76.77 & 73.29 & 40.73 & 32.74 & 36.25 & 31.39 \\
        moco\_v2\_downloaded & 2000 & 6e-05 & 75.08 & 86.10 & 37.29 & 32.77 & 32.52 & 31.45 \\
        moco\_v2\_downloaded & 2000 & 8e-05 & 77.01 & 82.30 & 40.85 & 27.62 & 32.43 & 35.60 \\
        moco\_v2\_downloaded & 4000 & 4e-05 & 72.91 & 88.09 & 34.66 & 30.39 & 35.32 & 21.16 \\
        moco\_v2\_downloaded & 4000 & 6e-05 & 75.98 & 78.29 & 36.71 & 31.16 & 34.57 & 34.41 \\
        moco\_v2\_downloaded & 4000 & 8e-05 & 80.89 & 77.89 & 47.20 & 35.60 & 41.50 & 36.42 \\
        moco\_v2\_downloaded & 6000 & 4e-05 & 84.00 & 70.90 & 47.16 & 30.31 & 34.93 & 48.03 \\
        moco\_v2\_downloaded & 6000 & 6e-05 & 80.81 & 79.80 & 41.35 & 34.97 & 36.08 & 37.73 \\
        moco\_v2\_downloaded & 6000 & 8e-05 & 81.56 & 81.35 & 46.55 & 32.22 & 39.53 & 39.52 \\
        moco\_v2\_downloaded & 8000 & 4e-05 & 83.77 & 62.57 & 45.10 & 33.14 & 39.28 & 44.05 \\
        moco\_v2\_downloaded & 8000 & 6e-05 & 82.19 & 75.49 & 45.83 & 37.43 & 41.13 & 46.65 \\
        moco\_v2\_downloaded & 8000 & 8e-05 & 81.49 & 87.73 & 51.89 & 34.70 & 37.83 & 35.42 \\"""
    lf = """bt-down-freezed & 2000 & 4e-05 & 83.91 & 93.57 & 25.33 & 29.44 & 5.65 & 29.12 \\
        bt-down-freezed & 2000 & 6e-05 & 88.74 & 94.39 & 24.08 & 30.34 & 5.64 & 30.51 \\
        bt-down-freezed & 2000 & 8e-05 & 85.27 & 98.09 & 27.37 & 30.53 & 5.38 & 26.10 \\
        bt-down-freezed & 4000 & 4e-05 & 83.89 & 92.48 & 22.25 & 29.47 & 5.30 & 32.58 \\
        bt-down-freezed & 4000 & 6e-05 & 85.53 & 92.52 & 24.42 & 28.12 & 5.68 & 32.21 \\
        bt-down-freezed & 4000 & 8e-05 & 81.29 & 95.69 & 16.85 & 27.77 & 5.49 & 37.37 \\
        bt-down-freezed & 6000 & 4e-05 & 87.15 & 86.35 & 21.52 & 30.29 & 6.54 & 30.87 \\
        bt-down-freezed & 6000 & 6e-05 & 83.97 & 94.63 & 20.61 & 28.12 & 5.64 & 41.45 \\
        bt-down-freezed & 6000 & 8e-05 & 84.60 & 88.17 & 20.75 & 26.35 & 5.32 & 33.06 \\
        bt-down-freezed & 8000 & 4e-05 & 85.54 & 91.82 & 22.85 & 31.94 & 6.30 & 19.98 \\
        bt-down-freezed & 8000 & 6e-05 & 85.60 & 94.16 & 23.56 & 30.04 & 6.06 & 43.93 \\
        bt-down-freezed & 8000 & 8e-05 & 85.05 & 95.58 & 25.22 & 29.92 & 5.99 & 33.68 \\
        dino-down-freezed & 2000 & 4e-05 & 81.65 & 90.04 & 5.96 & 29.42 & 4.67 & 24.06 \\
        dino-down-freezed & 2000 & 6e-05 & 85.76 & 91.94 & 30.73 & 29.71 & 4.49 & 45.47 \\
        dino-down-freezed & 2000 & 8e-05 & 86.95 & 86.10 & 26.98 & 30.80 & 4.86 & 0.21 \\
        dino-down-freezed & 4000 & 4e-05 & 86.92 & 89.90 & 15.98 & 34.15 & 4.82 & 26.38 \\
        dino-down-freezed & 4000 & 6e-05 & 86.13 & 87.19 & 26.36 & 31.84 & 5.55 & 34.14 \\
        dino-down-freezed & 4000 & 8e-05 & 89.64 & 76.04 & 33.08 & 33.25 & 5.32 & 32.48 \\
        dino-down-freezed & 6000 & 4e-05 & 85.28 & 94.17 & 32.86 & 30.49 & 4.84 & 33.20 \\
        dino-down-freezed & 6000 & 6e-05 & 87.03 & 91.61 & 24.99 & 31.99 & 4.46 & 37.01 \\
        dino-down-freezed & 6000 & 8e-05 & 87.27 & 91.69 & 39.41 & 30.53 & 5.16 & 38.06 \\
        dino-down-freezed & 8000 & 4e-05 & 85.96 & 94.12 & 30.27 & 30.52 & 5.87 & 23.73 \\
        dino-down-freezed & 8000 & 6e-05 & 89.19 & 91.23 & 30.12 & 30.90 & 5.48 & 30.81 \\
        dino-down-freezed & 8000 & 8e-05 & 85.73 & 92.20 & 25.38 & 31.43 & 6.23 & 20.24 \\
        vicreg\_down\_freeze & 2000 & 4e-05 & 81.81 & 93.77 & 26.28 & 28.29 & 4.35 & 45.51 \\
        vicreg\_down\_freeze & 2000 & 6e-05 & 84.68 & 93.21 & 28.17 & 28.95 & 5.76 & 50.58 \\
        vicreg\_down\_freeze & 2000 & 8e-05 & 83.08 & 94.97 & 34.05 & 29.96 & 5.52 & 44.65 \\
        vicreg\_down\_freeze & 4000 & 4e-05 & 83.62 & 92.01 & 22.71 & 28.56 & 4.37 & 47.05 \\
        vicreg\_down\_freeze & 4000 & 6e-05 & 84.27 & 91.01 & 34.28 & 33.52 & 4.67 & 36.01 \\
        vicreg\_down\_freeze & 4000 & 8e-05 & 83.94 & 94.09 & 27.82 & 30.31 & 4.84 & 43.52 \\
        vicreg\_down\_freeze & 6000 & 4e-05 & 86.53 & 86.36 & 25.03 & 30.06 & 5.78 & 40.39 \\
        vicreg\_down\_freeze & 6000 & 6e-05 & 84.25 & 97.03 & 23.77 & 28.78 & 5.52 & 41.25 \\
        vicreg\_down\_freeze & 6000 & 8e-05 & 86.07 & 96.10 & 35.26 & 31.62 & 5.20 & 41.79 \\
        vicreg\_down\_freeze & 8000 & 4e-05 & 84.08 & 94.22 & 22.23 & 29.46 & 5.23 & 47.57 \\
        vicreg\_down\_freeze & 8000 & 6e-05 & 83.94 & 95.49 & 31.61 & 30.57 & 4.81 & 47.09 \\
        vicreg\_down\_freeze & 8000 & 8e-05 & 84.48 & 96.12 & 27.73 & 28.92 & 4.47 & 48.10 \\
        simsiam\_freezed & 2000 & 4e-05 & 85.19 & 94.97 & 12.65 & 27.32 & 5.07 & 50.04 \\
        simsiam\_freezed & 2000 & 6e-05 & 83.95 & 91.62 & 14.91 & 27.53 & 5.63 & 44.48 \\
        simsiam\_freezed & 2000 & 8e-05 & 85.25 & 86.49 & 27.36 & 30.31 & 5.57 & 41.25 \\
        simsiam\_freezed & 4000 & 4e-05 & 89.66 & 89.60 & 20.89 & 32.37 & 5.33 & 48.74 \\
        simsiam\_freezed & 4000 & 6e-05 & 80.94 & 93.01 & 18.69 & 30.39 & 5.52 & 42.69 \\
        simsiam\_freezed & 4000 & 8e-05 & 86.97 & 92.35 & 26.27 & 30.38 & 6.35 & 47.05 \\
        simsiam\_freezed & 6000 & 4e-05 & 84.71 & 92.43 & 18.77 & 31.73 & 5.12 & 39.31 \\
        simsiam\_freezed & 6000 & 6e-05 & 85.31 & 92.86 & 23.15 & 31.10 & 6.08 & 39.22 \\
        simsiam\_freezed & 6000 & 8e-05 & 88.45 & 85.35 & 20.68 & 28.14 & 5.09 & 44.78 \\
        simsiam\_freezed & 8000 & 4e-05 & 88.59 & 90.03 & 22.68 & 29.73 & 6.16 & 38.27 \\
        simsiam\_freezed & 8000 & 6e-05 & 88.85 & 93.07 & 28.90 & 31.59 & 6.83 & 38.03 \\
        simsiam\_freezed & 8000 & 8e-05 & 90.90 & 81.86 & 34.52 & 31.25 & 4.84 & 46.64 \\
        moco\_v1\_downloaded & 2000 & 4e-05 & 85.60 & 90.18 & 18.09 & 27.24 & 4.15 & 43.37 \\
        moco\_v1\_downloaded & 2000 & 6e-05 & 86.96 & 89.08 & 18.47 & 29.43 & 4.46 & 49.69 \\
        moco\_v1\_downloaded & 2000 & 8e-05 & 85.80 & 92.09 & 18.08 & 28.30 & 4.72 & 45.26 \\
        moco\_v1\_downloaded & 4000 & 4e-05 & 85.95 & 87.80 & 17.19 & 28.11 & 3.76 & 31.40 \\
        moco\_v1\_downloaded & 4000 & 6e-05 & 86.47 & 91.22 & 20.68 & 30.07 & 5.08 & 48.85 \\
        moco\_v1\_downloaded & 4000 & 8e-05 & 87.45 & 91.34 & 22.59 & 30.07 & 4.69 & 49.95 \\
        moco\_v1\_downloaded & 6000 & 4e-05 & 87.67 & 92.64 & 22.38 & 30.80 & 3.87 & 56.67 \\
        moco\_v1\_downloaded & 6000 & 6e-05 & 88.46 & 89.71 & 20.12 & 29.53 & 4.02 & 52.84 \\
        moco\_v1\_downloaded & 6000 & 8e-05 & 87.09 & 93.25 & 21.36 & 30.84 & 5.27 & 46.50 \\
        moco\_v1\_downloaded & 8000 & 4e-05 & 88.51 & 90.11 & 22.20 & 31.13 & 4.97 & 49.87 \\
        moco\_v1\_downloaded & 8000 & 6e-05 & 89.90 & 85.35 & 25.76 & 30.87 & 5.14 & 48.64 \\
        moco\_v1\_downloaded & 8000 & 8e-05 & 88.44 & 93.85 & 31.48 & 32.73 & 4.97 & 52.88 \\
        moco\_v2\_downloaded & 2000 & 4e-05 & 88.73 & 90.98 & 32.42 & 33.75 & 5.87 & 25.55 \\
        moco\_v2\_downloaded & 2000 & 6e-05 & 87.07 & 93.67 & 30.17 & 31.76 & 5.11 & 40.29 \\
        moco\_v2\_downloaded & 2000 & 8e-05 & 88.55 & 93.83 & 40.08 & 31.68 & 4.99 & 40.91 \\
        moco\_v2\_downloaded & 4000 & 4e-05 & 86.96 & 96.82 & 31.59 & 32.66 & 4.43 & 33.99 \\
        moco\_v2\_downloaded & 4000 & 6e-05 & 87.85 & 92.85 & 37.03 & 32.95 & 5.67 & 40.00 \\
        moco\_v2\_downloaded & 4000 & 8e-05 & 91.41 & 92.39 & 41.45 & 34.56 & 6.05 & 25.84 \\
        moco\_v2\_downloaded & 6000 & 4e-05 & 89.65 & 93.35 & 31.90 & 33.85 & 5.11 & 30.98 \\
        moco\_v2\_downloaded & 6000 & 6e-05 & 90.44 & 87.82 & 34.89 & 34.84 & 5.53 & 48.20 \\
        moco\_v2\_downloaded & 6000 & 8e-05 & 89.75 & 94.49 & 30.94 & 32.26 & 5.50 & 48.12 \\
        moco\_v2\_downloaded & 8000 & 4e-05 & 88.92 & 96.17 & 27.33 & 33.63 & 4.79 & 42.59 \\
        moco\_v2\_downloaded & 8000 & 6e-05 & 90.34 & 93.61 & 36.16 & 32.39 & 4.88 & 38.80 \\
        moco\_v2\_downloaded & 8000 & 8e-05 & 88.85 & 95.92 & 34.39 & 32.96 & 4.86 & 45.07 \\"""
    sta = """bt-down-freezed & 2000 & 4e-05 & 91.28 & 69.86 & 43.46 & 48.68 & 17.78 & 37.79 \\
        bt-down-freezed & 2000 & 6e-05 & 91.24 & 82.92 & 42.25 & 44.01 & 22.63 & 38.31 \\
        bt-down-freezed & 2000 & 8e-05 & 93.79 & 17.07 & 35.63 & 48.08 & 13.80 & 42.88 \\
        bt-down-freezed & 4000 & 4e-05 & 94.66 & 19.16 & 49.85 & 47.23 & 17.51 & 46.20 \\
        bt-down-freezed & 4000 & 6e-05 & 94.48 & 17.88 & 48.88 & 42.89 & 21.84 & 29.62 \\
        bt-down-freezed & 4000 & 8e-05 & 94.08 & 18.61 & 45.05 & 46.42 & 21.24 & 10.94 \\
        bt-down-freezed & 6000 & 4e-05 & 93.93 & 17.27 & 40.53 & 47.50 & 20.24 & 42.25 \\
        bt-down-freezed & 6000 & 6e-05 & 90.93 & 69.79 & 39.14 & 41.82 & 20.13 & 80.51 \\
        bt-down-freezed & 6000 & 8e-05 & 89.34 & 77.71 & 44.97 & 42.05 & 15.46 & 78.09 \\
        bt-down-freezed & 8000 & 4e-05 & 94.54 & 17.77 & 47.77 & 48.25 & 22.80 & 46.84 \\
        bt-down-freezed & 8000 & 6e-05 & 91.26 & 72.86 & 48.06 & 39.76 & 15.12 & 38.67 \\
        bt-down-freezed & 8000 & 8e-05 & 93.22 & 21.00 & 43.61 & 53.18 & 17.82 & 82.69 \\
        dino-down-freezed & 2000 & 4e-05 & 86.80 & 80.12 & 16.61 & 31.87 & 12.21 & 54.96 \\
        dino-down-freezed & 2000 & 6e-05 & 89.69 & 71.02 & 23.32 & 36.46 & 12.70 & 44.55 \\
        dino-down-freezed & 2000 & 8e-05 & 86.65 & 78.10 & 21.59 & 32.04 & 12.17 & 65.67 \\
        dino-down-freezed & 4000 & 4e-05 & 89.81 & 70.08 & 26.91 & 35.72 & 14.28 & 46.10 \\
        dino-down-freezed & 4000 & 6e-05 & 92.62 & 44.67 & 35.34 & 41.46 & 12.37 & 37.35 \\
        dino-down-freezed & 4000 & 8e-05 & 92.01 & 41.69 & 30.03 & 37.73 & 13.15 & 53.81 \\
        dino-down-freezed & 6000 & 4e-05 & 89.34 & 65.20 & 23.51 & 40.65 & 12.32 & 84.86 \\
        dino-down-freezed & 6000 & 6e-05 & 90.13 & 63.60 & 28.02 & 39.30 & 12.25 & 48.64 \\
        dino-down-freezed & 6000 & 8e-05 & 81.33 & 90.17 & 23.63 & 32.88 & 14.15 & 15.74 \\
        dino-down-freezed & 8000 & 4e-05 & 90.96 & 30.83 & 26.14 & 40.47 & 10.33 & 84.19 \\
        dino-down-freezed & 8000 & 6e-05 & 91.83 & 31.78 & 32.94 & 38.05 & 14.28 & 28.00 \\
        dino-down-freezed & 8000 & 8e-05 & 90.07 & 64.21 & 35.67 & 32.41 & 15.17 & 81.10 \\
        vicreg\_down\_freeze & 2000 & 4e-05 & 82.70 & 87.12 & 21.62 & 44.53 & 17.36 & 16.90 \\
        vicreg\_down\_freeze & 2000 & 6e-05 & 88.19 & 70.41 & 29.77 & 47.35 & 17.59 & 31.24 \\
        vicreg\_down\_freeze & 2000 & 8e-05 & 85.67 & 80.41 & 24.78 & 44.34 & 16.03 & 40.20 \\
        vicreg\_down\_freeze & 4000 & 4e-05 & 85.91 & 79.99 & 20.98 & 38.19 & 17.67 & 33.24 \\
        vicreg\_down\_freeze & 4000 & 6e-05 & 91.19 & 37.36 & 38.28 & 48.43 & 19.51 & 81.82 \\
        vicreg\_down\_freeze & 4000 & 8e-05 & 85.90 & 84.68 & 26.05 & 41.20 & 15.13 & 24.26 \\
        vicreg\_down\_freeze & 6000 & 4e-05 & 87.01 & 83.06 & 30.52 & 46.04 & 17.76 & 25.50 \\
        vicreg\_down\_freeze & 6000 & 6e-05 & 85.50 & 81.87 & 27.70 & 47.66 & 20.74 & 66.36 \\
        vicreg\_down\_freeze & 6000 & 8e-05 & 91.70 & 34.81 & 34.66 & 46.70 & 21.57 & 82.17 \\
        vicreg\_down\_freeze & 8000 & 4e-05 & 88.47 & 65.74 & 28.81 & 44.54 & 15.16 & 81.20 \\
        vicreg\_down\_freeze & 8000 & 6e-05 & 86.63 & 87.03 & 27.82 & 46.49 & 14.94 & 57.84 \\
        vicreg\_down\_freeze & 8000 & 8e-05 & 87.16 & 85.53 & 26.45 & 52.47 & 16.29 & 65.30 \\
        simsiam\_freezed & 2000 & 4e-05 & 88.50 & 73.20 & 39.35 & 38.88 & 18.33 & 77.92 \\
        simsiam\_freezed & 2000 & 6e-05 & 89.18 & 40.47 & 36.90 & 44.19 & 15.16 & 62.80 \\
        simsiam\_freezed & 2000 & 8e-05 & 89.97 & 67.74 & 35.30 & 37.39 & 15.49 & 82.19 \\
        simsiam\_freezed & 4000 & 4e-05 & 93.30 & 27.86 & 56.97 & 43.35 & 23.59 & 80.01 \\
        simsiam\_freezed & 4000 & 6e-05 & 92.91 & 25.86 & 50.31 & 42.40 & 19.70 & 84.29 \\
        simsiam\_freezed & 4000 & 8e-05 & 92.75 & 27.10 & 52.52 & 42.46 & 18.10 & 22.95 \\
        simsiam\_freezed & 6000 & 4e-05 & 92.15 & 30.50 & 44.33 & 38.22 & 17.23 & 76.14 \\
        simsiam\_freezed & 6000 & 6e-05 & 93.62 & 32.21 & 55.12 & 52.66 & 16.78 & 82.91 \\
        simsiam\_freezed & 6000 & 8e-05 & 96.01 & 22.87 & 66.12 & 46.71 & 23.38 & 83.17 \\
        simsiam\_freezed & 8000 & 4e-05 & 92.82 & 28.18 & 47.40 & 38.37 & 19.68 & 77.01 \\
        simsiam\_freezed & 8000 & 6e-05 & 91.21 & 43.90 & 48.39 & 36.89 & 20.76 & 76.46 \\
        simsiam\_freezed & 8000 & 8e-05 & 92.42 & 55.10 & 60.61 & 43.84 & 20.25 & 77.96 \\
        moco\_v1\_downloaded & 2000 & 4e-05 & 88.99 & 77.30 & 36.33 & 40.52 & 13.27 & 31.90 \\
        moco\_v1\_downloaded & 2000 & 6e-05 & 93.59 & 29.32 & 39.99 & 43.69 & 16.20 & 56.13 \\
        moco\_v1\_downloaded & 2000 & 8e-05 & 88.30 & 86.32 & 34.19 & 37.25 & 16.16 & 45.06 \\
        moco\_v1\_downloaded & 4000 & 4e-05 & 91.67 & 39.09 & 37.61 & 45.04 & 14.43 & 77.28 \\
        moco\_v1\_downloaded & 4000 & 6e-05 & 92.93 & 32.27 & 46.77 & 43.75 & 17.56 & 67.58 \\
        moco\_v1\_downloaded & 4000 & 8e-05 & 89.82 & 54.34 & 37.78 & 40.51 & 15.08 & 32.65 \\
        moco\_v1\_downloaded & 6000 & 4e-05 & 93.20 & 30.23 & 41.02 & 44.34 & 14.67 & 78.56 \\
        moco\_v1\_downloaded & 6000 & 6e-05 & 93.90 & 31.10 & 46.77 & 43.75 & 17.56 & 68.07 \\
        moco\_v1\_downloaded & 6000 & 8e-05 & 91.46 & 54.34 & 37.78 & 40.51 & 15.08 & 32.65 \\
        moco\_v1\_downloaded & 8000 & 4e-05 & 92.76 & 34.38 & 42.91 & 44.56 & 17.67 & 75.81 \\
        moco\_v1\_downloaded & 8000 & 6e-05 & 89.59 & 59.44 & 42.16 & 40.76 & 17.18 & 28.97 \\
        moco\_v1\_downloaded & 8000 & 8e-05 & 89.71 & 64.83 & 42.32 & 44.15 & 16.98 & 40.00 \\
        moco\_v2\_downloaded & 2000 & 4e-05 & 94.34 & 19.79 & 61.00 & 42.83 & 22.08 & 36.00 \\
        moco\_v2\_downloaded & 2000 & 6e-05 & 94.80 & 23.84 & 70.21 & 43.56 & 29.91 & 8.97 \\
        moco\_v2\_downloaded & 2000 & 8e-05 & 95.90 & 16.46 & 65.17 & 47.72 & 26.17 & 78.60 \\
        moco\_v2\_downloaded & 4000 & 4e-05 & 91.92 & 30.64 & 52.24 & 40.62 & 24.57 & 26.63 \\
        moco\_v2\_downloaded & 4000 & 6e-05 & 92.05 & 32.39 & 57.24 & 42.84 & 22.74 & 19.67 \\
        moco\_v2\_downloaded & 4000 & 8e-05 & 93.09 & 30.04 & 61.90 & 39.08 & 22.24 & 13.49 \\
        moco\_v2\_downloaded & 6000 & 4e-05 & 95.05 & 23.14 & 66.84 & 39.05 & 28.43 & 64.98 \\
        moco\_v2\_downloaded & 6000 & 6e-05 & 93.63 & 24.62 & 58.56 & 37.79 & 19.56 & 20.81 \\
        moco\_v2\_downloaded & 6000 & 8e-05 & 93.08 & 28.72 & 61.00 & 35.08 & 23.51 & 58.70 \\
        moco\_v2\_downloaded & 8000 & 4e-05 & 95.10 & 20.50 & 64.27 & 41.96 & 22.45 & 78.69 \\
        moco\_v2\_downloaded & 8000 & 6e-05 & 94.36 & 23.11 & 60.33 & 40.02 & 20.02 & 77.42 \\
        moco\_v2\_downloaded & 8000 & 8e-05 & 94.34 & 22.14 & 63.71 & 45.86 & 26.27 & 29.04 \\"""

    pattern = re.compile(r"\d+e-\d+ & ([\d.\-]+(?: & [\d.\-]+)*)")
    matches_ra21 = pattern.findall(ra21)
    res = [r.split("&") for r in matches_ra21]
    data_ra21 = np.array(res, dtype=float)

    matches_ro21 = pattern.findall(ro21)
    res = [r.split("&") for r in matches_ro21]
    data_ro21 = np.array(res, dtype=float)

    matches_ra = pattern.findall(ra)
    res = [r.split("&") for r in matches_ra]
    data_ra = np.array(res, dtype=float)

    matches_lf = pattern.findall(lf)
    res = [r.split("&") for r in matches_lf]
    data_lf = np.array(res, dtype=float)

    matches_sta = pattern.findall(sta)
    res = [r.split("&") for r in matches_sta]
    data_sta = np.array(res, dtype=float)

    print(np.round(np.mean((data_ra21, data_ro21, data_ra, data_lf, data_sta), axis=0),decimals=2))
    # Define the LaTeX table content
    latex_content = """
    vicreg\_down\_freeze & 8000 & 8e-05 & 87.16 & 85.53 & 26.45 & 52.47 & 16.29 & 65.30 \\
    vicreg\_down\_freeze & 8000 & 8e-05 & 84.48 & 96.12 & 27.73 & 28.92 & 4.47 & 48.10 \\
    vicreg\_down\_freeze & 8000 & 8e-05 & 76.94 & 76.06 & 30.69 & 29.06 & 26.82 & 48.69 \\
    vicreg\_down\_freeze & 8000 & 8e-05 & 96.35 & 38.45 & 73.34 & 45.58 & 42.41 & 12.88 \\
    vicreg\_down\_freeze & 8000 & 8e-05 & 89.71 & 53.56 & 65.99 & 68.13 & 45.34 & 39.06 \\
    """

    # Extract the numerical values from the LaTeX table
    pattern = re.compile(r"\d+e-\d+ & ([\d.\-]+(?: & [\d.\-]+)*)")
    matches = pattern.findall(latex_content)
    res = [r.split("&") for r in matches]

    # Convert the extracted values to a numpy array
    data = np.array(res, dtype=float)
    print(data)

    # Compute the average values for each column
    average_values = np.mean(data, axis=0)

    # Print the average values
    metrics = ["AUROC", "FPR@TPR95", "AUPRC", "sIoU", "PPV", "PRR"]
    for metric, avg_value in zip(metrics, average_values):
        print(f"{metric}: {avg_value:.2f}")

def id_lora_FT(args):
    cfg = setup_cfgs(args)
    cfg.defrost()

    lrs = [8e-5, 6e-5]
    max_iters = [2000, 4000]

    models = [
        "moco_v2_downloaded_2000_6e-05_all",
        "moco_v2_downloaded_2000_6e-05_backbone_only",
        "moco_v2_downloaded_2000_6e-05_backbone_only_noOQ",
        "moco_v2_downloaded_2000_6e-05_predictor_and_backbone",
        "moco_v2_downloaded_2000_6e-05_predictor_only",
        "moco_v2_downloaded_2000_6e-05_predictor_only_noFFN",
        "moco_v2_downloaded_2000_6e-05_predictor_only_noFFN_noOQ",
        "moco_v2_downloaded_2000_8e-05_all",
        "moco_v2_downloaded_2000_8e-05_backbone_only",
        "moco_v2_downloaded_2000_8e-05_backbone_only_noOQ",
        "moco_v2_downloaded_2000_8e-05_predictor_and_backbone",
        "moco_v2_downloaded_2000_8e-05_predictor_only",
        "moco_v2_downloaded_2000_8e-05_predictor_only_noFFN",
        "moco_v2_downloaded_2000_8e-05_predictor_only_noFFN_noOQ",
        "moco_v2_downloaded_4000_6e-05_all",
        "moco_v2_downloaded_4000_6e-05_backbone_only",
        "moco_v2_downloaded_4000_6e-05_backbone_only_noOQ",
        "moco_v2_downloaded_4000_6e-05_predictor_and_backbone",
        "moco_v2_downloaded_4000_6e-05_predictor_only",
        "moco_v2_downloaded_4000_6e-05_predictor_only_noFFN",
        "moco_v2_downloaded_4000_6e-05_predictor_only_noFFN_noOQ",
        "moco_v2_downloaded_4000_8e-05_all",
        "moco_v2_downloaded_4000_8e-05_backbone_only",
        "moco_v2_downloaded_4000_8e-05_backbone_only_noOQ",
        "moco_v2_downloaded_4000_8e-05_predictor_and_backbone",
        "moco_v2_downloaded_4000_8e-05_predictor_only",
        "moco_v2_downloaded_4000_8e-05_predictor_only_noFFN",
        "moco_v2_downloaded_4000_8e-05_predictor_only_noFFN_noOQ",
        "moco_v2_downloaded_8000_6e-05_all",
        "moco_v2_downloaded_8000_6e-05_backbone_only",
        "moco_v2_downloaded_8000_6e-05_backbone_only_noOQ",
        "moco_v2_downloaded_8000_6e-05_predictor_and_backbone",
        "moco_v2_downloaded_8000_6e-05_predictor_only",
        "moco_v2_downloaded_8000_6e-05_predictor_only_noFFN",
        "moco_v2_downloaded_8000_6e-05_predictor_only_noFFN_noOQ",
        "moco_v2_downloaded_8000_8e-05_all",
        "moco_v2_downloaded_8000_8e-05_backbone_only",
        "moco_v2_downloaded_8000_8e-05_backbone_only_noOQ",
        "moco_v2_downloaded_8000_8e-05_predictor_and_backbone",
        "moco_v2_downloaded_8000_8e-05_predictor_only",
        "moco_v2_downloaded_8000_8e-05_predictor_only_noFFN",
        "moco_v2_downloaded_8000_8e-05_predictor_only_noFFN_noOQ",
    ]

    for lr in [8e-5]:
        for iter in [2000]:
            for model in models:
                cfg.MODEL.WEIGHTS = os.path.join("/home/nberardo/mask2former/output/train", "moco_v2_downloaded",
                                                 "model_final.pth")
                model_name = model + "_" + str(iter) + "_" + str(lr)
                cfg.OUTPUT_DIR = os.path.join("/home/nberardo/mask2former/output/LORA_FT", model_name)
                cfg.SOLVER.BASE_LR = lr
                cfg.SOLVER.MAX_ITER = iter

                loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
                for log in loggers:
                    log.handlers.clear()

                default_setup(cfg, args)
                setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")

                # sys.stderr = open(os.path.join(cfg.OUTPUT_DIR, "stderr.txt"), "w")


                trainer = Trainer(cfg)
                lora_path = os.path.join("/home/nberardo/mask2former/output/LORA", model, "lora_model")
                print(f"Loading LORA model from {lora_path}")
                lora_config = LoraConfig.from_pretrained(lora_path)
                new_model = get_peft_model(trainer._trainer.model, lora_config)
                new_model.load_state_dict(torch.load(lora_path + "/model.pth"), strict=False)
                new_model.merge_and_unload()
                print_named_modules(new_model)
                trainer._trainer.model = new_model
                trainer.test()
                return
                trainer.train()

if __name__ == "__main__":
    # test()
    # ood()
    # ood_lora()
    #COMMENT OUT IF RUNNING ID
    args = default_argument_parser().parse_args()
    launch(
        # id_lora,
        # id_lp_ft,
        id_lora_FT,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


