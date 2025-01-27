import os.path
import sys
from copy import deepcopy

from detectron2.engine import default_argument_parser, launch
from peft import LoraConfig, get_peft_model, inject_adapter_in_model, LoraModel, PeftModel, cast_mixed_precision_params, \
    get_peft_model_state_dict
import torch
from torch.nn.parallel import DistributedDataParallel
from train_net import Trainer, setup
from safetensors.torch import save_file as safe_save_file

def print_params(model: torch.nn.Module):
    for name,p in model.named_parameters():
        print(name + " has " + str(p.numel()) + " trainable parameters. Dtype = " + str(p.dtype))

def print_trainable_params(model: torch.nn.Module):
    for name,p in model.named_parameters():
        if p.requires_grad:
            print(name + " has " + str(p.numel()) + " trainable parameters. Dtype = " + str(p.type()))


def change_model_dtype(model: torch.nn.Module, dtype: torch.dtype):
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(dtype)

def print_named_modules(model):
    names = [(n, type(m)) for n, m in model.named_modules()]
    for name, module in names:
        print(name, module)


"""
backbone.stem.conv1.weight has 9408 trainable parameters. Dtype = torch.float32
backbone.stem.conv1.norm.weight has 64 trainable parameters. Dtype = torch.float32
backbone.stem.conv1.norm.bias has 64 trainable parameters. Dtype = torch.float32
backbone.res2.0.shortcut.weight has 16384 trainable parameters. Dtype = torch.float32
backbone.res2.0.shortcut.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res2.0.shortcut.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res2.0.conv1.weight has 4096 trainable parameters. Dtype = torch.float32
backbone.res2.0.conv1.norm.weight has 64 trainable parameters. Dtype = torch.float32
backbone.res2.0.conv1.norm.bias has 64 trainable parameters. Dtype = torch.float32
backbone.res2.0.conv2.weight has 36864 trainable parameters. Dtype = torch.float32
backbone.res2.0.conv2.norm.weight has 64 trainable parameters. Dtype = torch.float32
backbone.res2.0.conv2.norm.bias has 64 trainable parameters. Dtype = torch.float32
backbone.res2.0.conv3.weight has 16384 trainable parameters. Dtype = torch.float32
backbone.res2.0.conv3.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res2.0.conv3.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res2.1.conv1.weight has 16384 trainable parameters. Dtype = torch.float32
backbone.res2.1.conv1.norm.weight has 64 trainable parameters. Dtype = torch.float32
backbone.res2.1.conv1.norm.bias has 64 trainable parameters. Dtype = torch.float32
backbone.res2.1.conv2.weight has 36864 trainable parameters. Dtype = torch.float32
backbone.res2.1.conv2.norm.weight has 64 trainable parameters. Dtype = torch.float32
backbone.res2.1.conv2.norm.bias has 64 trainable parameters. Dtype = torch.float32
backbone.res2.1.conv3.weight has 16384 trainable parameters. Dtype = torch.float32
backbone.res2.1.conv3.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res2.1.conv3.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res2.2.conv1.weight has 16384 trainable parameters. Dtype = torch.float32
backbone.res2.2.conv1.norm.weight has 64 trainable parameters. Dtype = torch.float32
backbone.res2.2.conv1.norm.bias has 64 trainable parameters. Dtype = torch.float32
backbone.res2.2.conv2.weight has 36864 trainable parameters. Dtype = torch.float32
backbone.res2.2.conv2.norm.weight has 64 trainable parameters. Dtype = torch.float32
backbone.res2.2.conv2.norm.bias has 64 trainable parameters. Dtype = torch.float32
backbone.res2.2.conv3.weight has 16384 trainable parameters. Dtype = torch.float32
backbone.res2.2.conv3.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res2.2.conv3.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res3.0.shortcut.weight has 131072 trainable parameters. Dtype = torch.float32
backbone.res3.0.shortcut.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res3.0.shortcut.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res3.0.conv1.weight has 32768 trainable parameters. Dtype = torch.float32
backbone.res3.0.conv1.norm.weight has 128 trainable parameters. Dtype = torch.float32
backbone.res3.0.conv1.norm.bias has 128 trainable parameters. Dtype = torch.float32
backbone.res3.0.conv2.weight has 147456 trainable parameters. Dtype = torch.float32
backbone.res3.0.conv2.norm.weight has 128 trainable parameters. Dtype = torch.float32
backbone.res3.0.conv2.norm.bias has 128 trainable parameters. Dtype = torch.float32
backbone.res3.0.conv3.weight has 65536 trainable parameters. Dtype = torch.float32
backbone.res3.0.conv3.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res3.0.conv3.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res3.1.conv1.weight has 65536 trainable parameters. Dtype = torch.float32
backbone.res3.1.conv1.norm.weight has 128 trainable parameters. Dtype = torch.float32
backbone.res3.1.conv1.norm.bias has 128 trainable parameters. Dtype = torch.float32
backbone.res3.1.conv2.weight has 147456 trainable parameters. Dtype = torch.float32
backbone.res3.1.conv2.norm.weight has 128 trainable parameters. Dtype = torch.float32
backbone.res3.1.conv2.norm.bias has 128 trainable parameters. Dtype = torch.float32
backbone.res3.1.conv3.weight has 65536 trainable parameters. Dtype = torch.float32
backbone.res3.1.conv3.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res3.1.conv3.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res3.2.conv1.weight has 65536 trainable parameters. Dtype = torch.float32
backbone.res3.2.conv1.norm.weight has 128 trainable parameters. Dtype = torch.float32
backbone.res3.2.conv1.norm.bias has 128 trainable parameters. Dtype = torch.float32
backbone.res3.2.conv2.weight has 147456 trainable parameters. Dtype = torch.float32
backbone.res3.2.conv2.norm.weight has 128 trainable parameters. Dtype = torch.float32
backbone.res3.2.conv2.norm.bias has 128 trainable parameters. Dtype = torch.float32
backbone.res3.2.conv3.weight has 65536 trainable parameters. Dtype = torch.float32
backbone.res3.2.conv3.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res3.2.conv3.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res3.3.conv1.weight has 65536 trainable parameters. Dtype = torch.float32
backbone.res3.3.conv1.norm.weight has 128 trainable parameters. Dtype = torch.float32
backbone.res3.3.conv1.norm.bias has 128 trainable parameters. Dtype = torch.float32
backbone.res3.3.conv2.weight has 147456 trainable parameters. Dtype = torch.float32
backbone.res3.3.conv2.norm.weight has 128 trainable parameters. Dtype = torch.float32
backbone.res3.3.conv2.norm.bias has 128 trainable parameters. Dtype = torch.float32
backbone.res3.3.conv3.weight has 65536 trainable parameters. Dtype = torch.float32
backbone.res3.3.conv3.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res3.3.conv3.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res4.0.shortcut.weight has 524288 trainable parameters. Dtype = torch.float32
backbone.res4.0.shortcut.norm.weight has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.0.shortcut.norm.bias has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.0.conv1.weight has 131072 trainable parameters. Dtype = torch.float32
backbone.res4.0.conv1.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.0.conv1.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.0.conv2.weight has 589824 trainable parameters. Dtype = torch.float32
backbone.res4.0.conv2.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.0.conv2.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.0.conv3.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.0.conv3.norm.weight has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.0.conv3.norm.bias has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.1.conv1.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.1.conv1.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.1.conv1.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.1.conv2.weight has 589824 trainable parameters. Dtype = torch.float32
backbone.res4.1.conv2.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.1.conv2.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.1.conv3.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.1.conv3.norm.weight has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.1.conv3.norm.bias has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.2.conv1.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.2.conv1.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.2.conv1.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.2.conv2.weight has 589824 trainable parameters. Dtype = torch.float32
backbone.res4.2.conv2.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.2.conv2.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.2.conv3.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.2.conv3.norm.weight has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.2.conv3.norm.bias has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.3.conv1.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.3.conv1.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.3.conv1.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.3.conv2.weight has 589824 trainable parameters. Dtype = torch.float32
backbone.res4.3.conv2.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.3.conv2.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.3.conv3.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.3.conv3.norm.weight has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.3.conv3.norm.bias has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.4.conv1.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.4.conv1.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.4.conv1.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.4.conv2.weight has 589824 trainable parameters. Dtype = torch.float32
backbone.res4.4.conv2.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.4.conv2.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.4.conv3.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.4.conv3.norm.weight has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.4.conv3.norm.bias has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.5.conv1.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.5.conv1.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.5.conv1.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.5.conv2.weight has 589824 trainable parameters. Dtype = torch.float32
backbone.res4.5.conv2.norm.weight has 256 trainable parameters. Dtype = torch.float32
backbone.res4.5.conv2.norm.bias has 256 trainable parameters. Dtype = torch.float32
backbone.res4.5.conv3.weight has 262144 trainable parameters. Dtype = torch.float32
backbone.res4.5.conv3.norm.weight has 1024 trainable parameters. Dtype = torch.float32
backbone.res4.5.conv3.norm.bias has 1024 trainable parameters. Dtype = torch.float32
backbone.res5.0.shortcut.weight has 2097152 trainable parameters. Dtype = torch.float32
backbone.res5.0.shortcut.norm.weight has 2048 trainable parameters. Dtype = torch.float32
backbone.res5.0.shortcut.norm.bias has 2048 trainable parameters. Dtype = torch.float32
backbone.res5.0.conv1.weight has 524288 trainable parameters. Dtype = torch.float32
backbone.res5.0.conv1.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res5.0.conv1.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res5.0.conv2.weight has 2359296 trainable parameters. Dtype = torch.float32
backbone.res5.0.conv2.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res5.0.conv2.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res5.0.conv3.weight has 1048576 trainable parameters. Dtype = torch.float32
backbone.res5.0.conv3.norm.weight has 2048 trainable parameters. Dtype = torch.float32
backbone.res5.0.conv3.norm.bias has 2048 trainable parameters. Dtype = torch.float32
backbone.res5.1.conv1.weight has 1048576 trainable parameters. Dtype = torch.float32
backbone.res5.1.conv1.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res5.1.conv1.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res5.1.conv2.weight has 2359296 trainable parameters. Dtype = torch.float32
backbone.res5.1.conv2.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res5.1.conv2.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res5.1.conv3.weight has 1048576 trainable parameters. Dtype = torch.float32
backbone.res5.1.conv3.norm.weight has 2048 trainable parameters. Dtype = torch.float32
backbone.res5.1.conv3.norm.bias has 2048 trainable parameters. Dtype = torch.float32
backbone.res5.2.conv1.weight has 1048576 trainable parameters. Dtype = torch.float32
backbone.res5.2.conv1.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res5.2.conv1.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res5.2.conv2.weight has 2359296 trainable parameters. Dtype = torch.float32
backbone.res5.2.conv2.norm.weight has 512 trainable parameters. Dtype = torch.float32
backbone.res5.2.conv2.norm.bias has 512 trainable parameters. Dtype = torch.float32
backbone.res5.2.conv3.weight has 1048576 trainable parameters. Dtype = torch.float32
backbone.res5.2.conv3.norm.weight has 2048 trainable parameters. Dtype = torch.float32
backbone.res5.2.conv3.norm.bias has 2048 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.0.0.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.0.0.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.0.1.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.0.1.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.1.0.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.1.0.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.1.1.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.1.1.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.2.0.weight has 131072 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.2.0.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.2.1.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.input_proj.2.1.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.level_embed has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.sampling_offsets.weight has 49152 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.sampling_offsets.bias has 192 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.attention_weights.weight has 24576 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.attention_weights.bias has 96 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.value_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.value_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.output_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.output_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm1.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm1.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear1.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear1.bias has 1024 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear2.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm2.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.sampling_offsets.weight has 49152 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.sampling_offsets.bias has 192 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.attention_weights.weight has 24576 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.attention_weights.bias has 96 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.value_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.value_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.output_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.output_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm1.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm1.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear1.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear1.bias has 1024 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear2.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm2.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.sampling_offsets.weight has 49152 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.sampling_offsets.bias has 192 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.attention_weights.weight has 24576 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.attention_weights.bias has 96 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.value_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.value_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.output_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.output_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm1.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm1.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear1.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear1.bias has 1024 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear2.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm2.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.sampling_offsets.weight has 49152 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.sampling_offsets.bias has 192 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.attention_weights.weight has 24576 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.attention_weights.bias has 96 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.value_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.value_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.output_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.output_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm1.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm1.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear1.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear1.bias has 1024 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear2.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm2.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.sampling_offsets.weight has 49152 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.sampling_offsets.bias has 192 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.attention_weights.weight has 24576 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.attention_weights.bias has 96 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.value_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.value_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.output_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.output_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm1.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm1.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear1.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear1.bias has 1024 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear2.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm2.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.sampling_offsets.weight has 49152 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.sampling_offsets.bias has 192 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.attention_weights.weight has 24576 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.attention_weights.bias has 96 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.value_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.value_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.output_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.output_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm1.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm1.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear1.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear1.bias has 1024 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear2.weight has 262144 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm2.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.mask_features.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.mask_features.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.adapter_1.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.adapter_1.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.adapter_1.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.layer_1.weight has 589824 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.layer_1.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.pixel_decoder.layer_1.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.0.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.0.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.1.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.1.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.2.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.2.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.3.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.3.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.4.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.4.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.5.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.5.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.6.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.6.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.7.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.7.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.8.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_self_attention_layers.8.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.0.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.0.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.1.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.1.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.2.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.2.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.3.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.3.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.4.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.4.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.5.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.5.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.6.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.6.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.7.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.7.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.in_proj_weight has 196608 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.in_proj_bias has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.out_proj.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.out_proj.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.8.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_cross_attention_layers.8.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.0.linear1.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.0.linear1.bias has 2048 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.0.linear2.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.0.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.0.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.0.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.1.linear1.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.1.linear1.bias has 2048 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.1.linear2.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.1.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.1.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.1.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.2.linear1.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.2.linear1.bias has 2048 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.2.linear2.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.2.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.2.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.2.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.3.linear1.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.3.linear1.bias has 2048 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.3.linear2.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.3.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.3.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.3.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.4.linear1.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.4.linear1.bias has 2048 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.4.linear2.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.4.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.4.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.4.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.5.linear1.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.5.linear1.bias has 2048 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.5.linear2.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.5.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.5.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.5.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.6.linear1.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.6.linear1.bias has 2048 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.6.linear2.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.6.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.6.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.6.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.7.linear1.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.7.linear1.bias has 2048 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.7.linear2.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.7.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.7.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.7.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.8.linear1.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.8.linear1.bias has 2048 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.8.linear2.weight has 524288 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.8.linear2.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.8.norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.transformer_ffn_layers.8.norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.decoder_norm.weight has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.decoder_norm.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.query_feat.weight has 25600 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.query_embed.weight has 25600 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.level_embed.weight has 768 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.class_embed.weight has 5120 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.class_embed.bias has 20 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.mask_embed.layers.0.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.mask_embed.layers.0.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.mask_embed.layers.1.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.mask_embed.layers.1.bias has 256 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.mask_embed.layers.2.weight has 65536 trainable parameters. Dtype = torch.float32
sem_seg_head.predictor.mask_embed.layers.2.bias has 256 trainable parameters. Dtype = torch.float32
"""

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
                        #r"|sem_seg_head\.predictor\.transformer_ffn_layers\.\d\.linear.+"
                       r"sem_seg_head\.predictor\.transformer_cross_attention_layers\.\d\.multihead_attn\.\w+"
                       r"|sem_seg_head\.predictor\.transformer_self_attention_layers\.\d\.self_attn\.\w+",
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["sem_seg_head.predictor.mask_embed",
                         "sem_seg_head.pixel_decoder.input_proj.0",
                         "sem_seg_head.pixel_decoder.input_proj.1",
                         "sem_seg_head.pixel_decoder.input_proj.2"],
                         # "sem_seg_head.predictor.query_embed",
                         # "sem_seg_head.predictor.query_feat",
                         # "sem_seg_head.predictor.class_embed"],
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


    lora_model = get_peft_model(model,deepcopy(lora_cfg))

    lora_model.print_trainable_parameters()

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
        print("Saving Dist Model to ", lora_path)
        trainer._trainer.model.module.save_pretrained(lora_path)
    else:
        print("Saving Model to ", lora_path)
        trainer._trainer.model.save_pretrained(lora_path)
        torch.save(trainer._trainer.model.state_dict(),lora_path+"model.pth")


    #TEST:
    trainer._trainer.model.eval()
    print_named_modules(tmp_model)
    loaded = PeftModel.from_pretrained(tmp_model, lora_path)
    loaded.load_state_dict(torch.load(cfg.OUTPUT_DIR+"model_final.pth"))
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

