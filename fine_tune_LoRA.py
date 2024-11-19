import os.path

from detectron2.engine import default_argument_parser, launch
from peft import LoraConfig, get_peft_model, inject_adapter_in_model, LoraModel, PeftModel, cast_mixed_precision_params
import torch
from torch.nn.parallel import DistributedDataParallel
from train_net import Trainer, setup

def print_params(model: torch.nn.Module):
    for name,p in model.named_parameters():
        print(name + " has " + str(p.numel()) + " trainable parameters. Dtype = " + str(p.dtype))


def change_model_dtype(model: torch.nn.Module, dtype: torch.dtype):
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(dtype)

def print_named_modules(model):
    names = [(n, type(m)) for n, m in model.named_modules()]
    for name, module in names:
        print(name, module)


"""
PARAMS:
module.backbone <class 'detectron2.modeling.backbone.resnet.ResNet'>
module.backbone.stem <class 'detectron2.modeling.backbone.resnet.BasicStem'>
module.backbone.stem.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.stem.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res2 <class 'torch.nn.modules.container.Sequential'>
module.backbone.res2.0 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res2.0.shortcut <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res2.0.shortcut.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res2.0.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res2.0.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res2.0.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res2.0.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res2.0.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res2.0.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res2.1 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res2.1.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res2.1.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res2.1.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res2.1.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res2.1.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res2.1.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res2.2 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res2.2.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res2.2.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res2.2.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res2.2.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res2.2.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res2.2.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3 <class 'torch.nn.modules.container.Sequential'>
module.backbone.res3.0 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res3.0.shortcut <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.0.shortcut.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.0.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.0.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.0.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.0.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.0.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.0.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.1 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res3.1.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.1.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.1.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.1.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.1.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.1.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.2 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res3.2.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.2.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.2.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.2.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.2.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.2.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.3 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res3.3.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.3.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.3.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.3.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res3.3.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res3.3.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4 <class 'torch.nn.modules.container.Sequential'>
module.backbone.res4.0 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res4.0.shortcut <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.0.shortcut.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.0.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.0.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.0.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.0.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.0.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.0.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.1 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res4.1.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.1.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.1.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.1.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.1.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.1.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.2 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res4.2.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.2.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.2.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.2.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.2.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.2.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.3 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res4.3.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.3.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.3.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.3.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.3.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.3.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.4 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res4.4.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.4.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.4.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.4.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.4.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.4.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.5 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res4.5.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.5.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.5.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.5.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res4.5.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res4.5.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res5 <class 'torch.nn.modules.container.Sequential'>
module.backbone.res5.0 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res5.0.shortcut <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res5.0.shortcut.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res5.0.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res5.0.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res5.0.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res5.0.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res5.0.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res5.0.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res5.1 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res5.1.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res5.1.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res5.1.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res5.1.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res5.1.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res5.1.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res5.2 <class 'detectron2.modeling.backbone.resnet.BottleneckBlock'>
module.backbone.res5.2.conv1 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res5.2.conv1.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res5.2.conv2 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res5.2.conv2.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.backbone.res5.2.conv3 <class 'detectron2.layers.wrappers.Conv2d'>
module.backbone.res5.2.conv3.norm <class 'torch.nn.modules.batchnorm.SyncBatchNorm'>
module.sem_seg_head <class 'mask2former.modeling.meta_arch.mask_former_head.MaskFormerHead'>
module.sem_seg_head.pixel_decoder <class 'mask2former.modeling.pixel_decoder.msdeformattn.MSDeformAttnPixelDecoder'>
module.sem_seg_head.pixel_decoder.input_proj <class 'torch.nn.modules.container.ModuleList'>
module.sem_seg_head.pixel_decoder.input_proj.0 <class 'torch.nn.modules.container.Sequential'>
module.sem_seg_head.pixel_decoder.input_proj.0.0 <class 'torch.nn.modules.conv.Conv2d'>
module.sem_seg_head.pixel_decoder.input_proj.0.1 <class 'torch.nn.modules.normalization.GroupNorm'>
module.sem_seg_head.pixel_decoder.input_proj.1 <class 'torch.nn.modules.container.Sequential'>
module.sem_seg_head.pixel_decoder.input_proj.1.0 <class 'torch.nn.modules.conv.Conv2d'>
module.sem_seg_head.pixel_decoder.input_proj.1.1 <class 'torch.nn.modules.normalization.GroupNorm'>
module.sem_seg_head.pixel_decoder.input_proj.2 <class 'torch.nn.modules.container.Sequential'>
module.sem_seg_head.pixel_decoder.input_proj.2.0 <class 'torch.nn.modules.conv.Conv2d'>
module.sem_seg_head.pixel_decoder.input_proj.2.1 <class 'torch.nn.modules.normalization.GroupNorm'>
module.sem_seg_head.pixel_decoder.transformer <class 'mask2former.modeling.pixel_decoder.msdeformattn.MSDeformAttnTransformerEncoderOnly'>
module.sem_seg_head.pixel_decoder.transformer.encoder <class 'mask2former.modeling.pixel_decoder.msdeformattn.MSDeformAttnTransformerEncoder'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers <class 'torch.nn.modules.container.ModuleList'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0 <class 'mask2former.modeling.pixel_decoder.msdeformattn.MSDeformAttnTransformerEncoderLayer'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn <class 'mask2former.modeling.pixel_decoder.ops.modules.ms_deform_attn.MSDeformAttn'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.sampling_offsets <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.attention_weights <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.value_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.output_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.dropout1 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm1 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.dropout2 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.dropout3 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm2 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1 <class 'mask2former.modeling.pixel_decoder.msdeformattn.MSDeformAttnTransformerEncoderLayer'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn <class 'mask2former.modeling.pixel_decoder.ops.modules.ms_deform_attn.MSDeformAttn'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.sampling_offsets <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.attention_weights <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.value_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.output_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.dropout1 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm1 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.dropout2 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.dropout3 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm2 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2 <class 'mask2former.modeling.pixel_decoder.msdeformattn.MSDeformAttnTransformerEncoderLayer'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn <class 'mask2former.modeling.pixel_decoder.ops.modules.ms_deform_attn.MSDeformAttn'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.sampling_offsets <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.attention_weights <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.value_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.output_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.dropout1 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm1 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.dropout2 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.dropout3 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm2 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3 <class 'mask2former.modeling.pixel_decoder.msdeformattn.MSDeformAttnTransformerEncoderLayer'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn <class 'mask2former.modeling.pixel_decoder.ops.modules.ms_deform_attn.MSDeformAttn'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.sampling_offsets <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.attention_weights <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.value_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.output_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.dropout1 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm1 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.dropout2 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.dropout3 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm2 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4 <class 'mask2former.modeling.pixel_decoder.msdeformattn.MSDeformAttnTransformerEncoderLayer'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn <class 'mask2former.modeling.pixel_decoder.ops.modules.ms_deform_attn.MSDeformAttn'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.sampling_offsets <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.attention_weights <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.value_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.output_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.dropout1 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm1 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.dropout2 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.dropout3 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm2 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5 <class 'mask2former.modeling.pixel_decoder.msdeformattn.MSDeformAttnTransformerEncoderLayer'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn <class 'mask2former.modeling.pixel_decoder.ops.modules.ms_deform_attn.MSDeformAttn'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.sampling_offsets <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.attention_weights <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.value_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.output_proj <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.dropout1 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm1 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.dropout2 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.dropout3 <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm2 <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.pixel_decoder.pe_layer <class 'mask2former.modeling.transformer_decoder.position_encoding.PositionEmbeddingSine'>
module.sem_seg_head.pixel_decoder.mask_features <class 'detectron2.layers.wrappers.Conv2d'>
module.sem_seg_head.pixel_decoder.adapter_1 <class 'detectron2.layers.wrappers.Conv2d'>
module.sem_seg_head.pixel_decoder.adapter_1.norm <class 'torch.nn.modules.normalization.GroupNorm'>
module.sem_seg_head.pixel_decoder.layer_1 <class 'detectron2.layers.wrappers.Conv2d'>
module.sem_seg_head.pixel_decoder.layer_1.norm <class 'torch.nn.modules.normalization.GroupNorm'>
module.sem_seg_head.predictor <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.MultiScaleMaskedTransformerDecoder'>
module.sem_seg_head.predictor.pe_layer <class 'mask2former.modeling.transformer_decoder.position_encoding.PositionEmbeddingSine'>
module.sem_seg_head.predictor.transformer_self_attention_layers <class 'torch.nn.modules.container.ModuleList'>
module.sem_seg_head.predictor.transformer_self_attention_layers.0 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.SelfAttentionLayer'>
module.sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_self_attention_layers.0.self_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_self_attention_layers.0.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_self_attention_layers.0.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_self_attention_layers.1 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.SelfAttentionLayer'>
module.sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_self_attention_layers.1.self_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_self_attention_layers.1.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_self_attention_layers.1.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_self_attention_layers.2 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.SelfAttentionLayer'>
module.sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_self_attention_layers.2.self_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_self_attention_layers.2.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_self_attention_layers.2.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_self_attention_layers.3 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.SelfAttentionLayer'>
module.sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_self_attention_layers.3.self_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_self_attention_layers.3.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_self_attention_layers.3.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_self_attention_layers.4 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.SelfAttentionLayer'>
module.sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_self_attention_layers.4.self_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_self_attention_layers.4.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_self_attention_layers.4.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_self_attention_layers.5 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.SelfAttentionLayer'>
module.sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_self_attention_layers.5.self_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_self_attention_layers.5.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_self_attention_layers.5.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_self_attention_layers.6 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.SelfAttentionLayer'>
module.sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_self_attention_layers.6.self_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_self_attention_layers.6.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_self_attention_layers.6.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_self_attention_layers.7 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.SelfAttentionLayer'>
module.sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_self_attention_layers.7.self_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_self_attention_layers.7.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_self_attention_layers.7.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_self_attention_layers.8 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.SelfAttentionLayer'>
module.sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_self_attention_layers.8.self_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_self_attention_layers.8.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_self_attention_layers.8.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_cross_attention_layers <class 'torch.nn.modules.container.ModuleList'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.0 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.CrossAttentionLayer'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.0.multihead_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.0.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.0.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.1 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.CrossAttentionLayer'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.1.multihead_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.1.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.1.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.2 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.CrossAttentionLayer'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.2.multihead_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.2.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.2.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.3 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.CrossAttentionLayer'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.3.multihead_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.3.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.3.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.4 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.CrossAttentionLayer'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.4.multihead_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.4.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.4.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.5 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.CrossAttentionLayer'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.5.multihead_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.5.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.5.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.6 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.CrossAttentionLayer'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.6.multihead_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.6.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.6.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.7 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.CrossAttentionLayer'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.7.multihead_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.7.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.7.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.8 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.CrossAttentionLayer'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn <class 'torch.nn.modules.activation.MultiheadAttention'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.8.multihead_attn.out_proj <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.8.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_cross_attention_layers.8.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_ffn_layers <class 'torch.nn.modules.container.ModuleList'>
module.sem_seg_head.predictor.transformer_ffn_layers.0 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.FFNLayer'>
module.sem_seg_head.predictor.transformer_ffn_layers.0.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.0.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_ffn_layers.0.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.0.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_ffn_layers.1 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.FFNLayer'>
module.sem_seg_head.predictor.transformer_ffn_layers.1.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.1.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_ffn_layers.1.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.1.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_ffn_layers.2 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.FFNLayer'>
module.sem_seg_head.predictor.transformer_ffn_layers.2.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.2.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_ffn_layers.2.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.2.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_ffn_layers.3 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.FFNLayer'>
module.sem_seg_head.predictor.transformer_ffn_layers.3.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.3.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_ffn_layers.3.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.3.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_ffn_layers.4 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.FFNLayer'>
module.sem_seg_head.predictor.transformer_ffn_layers.4.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.4.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_ffn_layers.4.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.4.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_ffn_layers.5 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.FFNLayer'>
module.sem_seg_head.predictor.transformer_ffn_layers.5.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.5.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_ffn_layers.5.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.5.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_ffn_layers.6 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.FFNLayer'>
module.sem_seg_head.predictor.transformer_ffn_layers.6.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.6.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_ffn_layers.6.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.6.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_ffn_layers.7 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.FFNLayer'>
module.sem_seg_head.predictor.transformer_ffn_layers.7.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.7.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_ffn_layers.7.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.7.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.transformer_ffn_layers.8 <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.FFNLayer'>
module.sem_seg_head.predictor.transformer_ffn_layers.8.linear1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.8.dropout <class 'torch.nn.modules.dropout.Dropout'>
module.sem_seg_head.predictor.transformer_ffn_layers.8.linear2 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.transformer_ffn_layers.8.norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.decoder_norm <class 'torch.nn.modules.normalization.LayerNorm'>
module.sem_seg_head.predictor.query_feat <class 'torch.nn.modules.sparse.Embedding'>
module.sem_seg_head.predictor.query_embed <class 'torch.nn.modules.sparse.Embedding'>
module.sem_seg_head.predictor.level_embed <class 'torch.nn.modules.sparse.Embedding'>
module.sem_seg_head.predictor.input_proj <class 'torch.nn.modules.container.ModuleList'>
module.sem_seg_head.predictor.input_proj.0 <class 'torch.nn.modules.container.Sequential'>
module.sem_seg_head.predictor.input_proj.1 <class 'torch.nn.modules.container.Sequential'>
module.sem_seg_head.predictor.input_proj.2 <class 'torch.nn.modules.container.Sequential'>
module.sem_seg_head.predictor.class_embed <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.mask_embed <class 'mask2former.modeling.transformer_decoder.mask2former_transformer_decoder.MLP'>
module.sem_seg_head.predictor.mask_embed.layers <class 'torch.nn.modules.container.ModuleList'>
module.sem_seg_head.predictor.mask_embed.layers.0 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.mask_embed.layers.1 <class 'torch.nn.modules.linear.Linear'>
module.sem_seg_head.predictor.mask_embed.layers.2 <class 'torch.nn.modules.linear.Linear'>
"""

def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    model = trainer._trainer.model

    if isinstance(model, DistributedDataParallel):
        model = trainer._trainer.model.module

    print_params(model)
    return

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=r"sem_seg_head\.pixel_decoder\.transformer\.encoder\.layers\.\d\.linear\d"
                       r"|sem_seg_head\.pixel_decoder\.transformer\.encoder\.layers\.\d\.self_attn\.\w+"
                       r"|backbone\.res\d\.\d\.conv\d",
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["predictor"],
    )

    lora_model = get_peft_model(model,lora_cfg)

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

