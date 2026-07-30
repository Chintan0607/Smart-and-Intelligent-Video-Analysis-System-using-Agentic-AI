import torch

print("="*50)
print("PyTorch Version:",torch.__version__)
print("CUDA Available:",torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Version: ",torch.version.cuda)
    print("GPU: ",torch.cuda.get_device_name(0))
    print("GPU Count: ",torch.cuda.device_count())

    x = torch.rand(3,3).cuda()
    print("\nTensor on GPU:")
    print(x)
else:
    print("No GPU available. Please check your CUDA installation.")
print("="*50)