import torch
import torchvision

if __name__ == '__main__':
    # Load from root model
    model_fp32 = torchvision.models.resnet50()
    checkpoint = torch.load('/mnt/4TData/vuquang/3d-face-rec/Deep3DFaceRecon_pytorch/checkpoints/init_model/resnet50-0676ba61.pth')
    model_fp32.load_state_dict(checkpoint)
    
    # Quantize from fp32 to int8
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_fp32,             # the original model
        {torch.nn.Linear},      # a set of layers to dynamically quantize
        dtype=torch.qint8       # the target dtype for quantized weights
    )
    
    # Save int8 model
    torch.save(model_int8.state_dict(), '/mnt/4TData/vuquang/3d-face-rec/Deep3DFaceRecon_pytorch/checkpoints/init_model/resnet50-0676ba61_int8.pth')
    
    # x = torch.randn(1, 3, 224, 224, requires_grad=True)