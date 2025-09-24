import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device count:', torch.cuda.device_count())
    print('Current device:', torch.cuda.current_device())
    print('Device name:', torch.cuda.get_device_name())
else:
    print('Using CPU for training')
