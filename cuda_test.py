import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.backends.cudnn as cudnn

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", cudnn.version() if cudnn.is_available() else "Not available")

# Force CUDA initialization
if not torch.cuda.is_available():
    print("CUDA is not available. Trying to initialize...")
    try:
        torch.cuda.init()
        print("CUDA initialized successfully")
    except Exception as e:
        print(f"Failed to initialize CUDA: {e}")

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))

    # Try to allocate a tensor on GPU
    x = torch.rand(5, 3).cuda()
    print("GPU tensor:", x)

    # Try a simple GPU operation
    y = x + x
    print("GPU operation result:", y)
else:
    print("CUDA is not available")

# Print CUDA memory info
if torch.cuda.is_available():
    print("\nCUDA Memory Summary:")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
