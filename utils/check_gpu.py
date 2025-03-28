import torch
import sys
import platform

def check_gpu_status():
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    
    print("\n=== CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            
        # Test CUDA with a simple operation
        print("\n=== CUDA Test ===")
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("CUDA matrix multiplication test: SUCCESS")
            print(f"Result shape: {z.shape}")
            print(f"Device: {z.device}")
        except Exception as e:
            print(f"CUDA test failed: {str(e)}")
    else:
        print("CUDA is not available. Please check your PyTorch installation and GPU drivers.")

if __name__ == "__main__":
    check_gpu_status() 