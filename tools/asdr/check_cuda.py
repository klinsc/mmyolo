import torch

if torch.cuda.is_available():
    print("CUDA is available")
    torch.cuda.device_count()
    torch.cuda.current_device()
    torch.cuda.device(0)
    torch.cuda.get_device_name(0)
else:
    print("CUDA is not available")
