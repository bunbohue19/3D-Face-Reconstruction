import os
import shutil
import sys
import cv2
import numpy as np 
import onnx
import torch
import torchvision
import tensorflow as tf 
from PIL import Image
from torchvision import transforms
from torchvision.models import *
from torchsummary import summary
from onnx_tf.backend import prepare

def convert_torch_to_onnx(onnx_path, torch_path=None):
    """
        Coverts Pytorch model file to ONNX
        :param torch_path: Torch model path to load
        :param onnx_path: ONNX model path to save
    """
    model = torchvision.models.resnet50(weights=torch.load('/mnt/4TData/vuquang/3d-face-rec/Deep3DFaceRecon_pytorch/checkpoints/12-05-2024/epoch_latest.pth'))
    
    x = torch.randn(1, 3, 224, 224, requires_grad=True)

    torch.onnx.export(
        model = model,
        args = x,
        f = onnx_path,
        verbose = False,
        export_params=True,
        do_constant_folding = False,
        input_names = ['input'],
        opset_version = 10,
        output_names = ['output'])

if __name__ == '__main__':
    convert_torch_to_onnx('/mnt/4TData/vuquang/3d-face-rec/Deep3DFaceRecon_pytorch/checkpoints/12-05-2024/epoch_latest.onnx',
                          '/mnt/4TData/vuquang/3d-face-rec/Deep3DFaceRecon_pytorch/checkpoints/12-05-2024/epoch_latest.pth')