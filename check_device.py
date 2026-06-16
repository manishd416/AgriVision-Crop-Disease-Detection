import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

print(f"PyTorch Version: {torch.__version__}")

# 1. Check if the Intel GPU (XPU) backend is available
xpu_available = torch.xpu.is_available()
print(f"Intel GPU (XPU) Available: {xpu_available}")

if xpu_available:
    # 2. Get the name of your specific Intel Arc GPU
    device_name = torch.xpu.get_device_name(0)
    print(f"Device Name: {device_name}\n")
    
    # 3. Define the device as XPU
    device = torch.device("xpu")
    
    # 4. Perform a simple matrix multiplication test on the GPU
    print("Testing GPU matrix multiplication...")
    
    # Create two random tensors directly on the Intel GPU
    x = torch.randn(2000, 2000, device=device)
    y = torch.randn(2000, 2000, device=device)
    
    # Multiply them together on the GPU
    z = torch.matmul(x, y)
    
    # Bring a tiny piece back to CPU just to print it
    print("Test successful! First few elements of the result:")
    print(z[:2, :2])
else:
    print("\n[ERROR] Intel GPU not detected.")
    print("Please ensure you uninstalled the old version and installed via:")
    print("pip install torch --index-url https://download.pytorch.org/whl/xpu")
