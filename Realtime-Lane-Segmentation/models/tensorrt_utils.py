import os
import torch
import numpy as np
from typing import Optional, Tuple, List
import logging

# TensorRT imports with fallback
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
    print("TensorRT support enabled")
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT not available. Install with: pip install tensorrt pycuda")

__all__ = ['TensorRTOptimizer', 'is_tensorrt_available']


def is_tensorrt_available():
    """Check if TensorRT is available"""
    return TRT_AVAILABLE


class TensorRTOptimizer:
    """TensorRT model optimizer for faster inference"""
    
    def __init__(self, input_shape: Tuple[int, int, int, int] = (1, 3, 480, 640)):
        """
        Initialize TensorRT optimizer
        
        Args:
            input_shape: Input tensor shape (batch, channels, height, width)
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install with: pip install tensorrt pycuda")
            
        self.input_shape = input_shape
        self.logger = trt.Logger(trt.Logger.WARNING)
        
    def pytorch_to_tensorrt(self, model: torch.nn.Module, 
                          engine_path: str,
                          fp16: bool = True,
                          max_workspace_size: int = 1 << 30) -> bool:
        """
        Convert PyTorch model to TensorRT engine
        
        Args:
            model: PyTorch model to convert
            engine_path: Path to save TensorRT engine
            fp16: Use FP16 precision for faster inference
            max_workspace_size: Maximum workspace size in bytes
            
        Returns:
            bool: Success status
        """
        try:
            # Create sample input
            dummy_input = torch.randn(*self.input_shape).cuda()
            
            # Export to ONNX first
            onnx_path = engine_path.replace('.trt', '.onnx')
            
            model.eval()
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
            
            # Convert ONNX to TensorRT
            success = self.onnx_to_tensorrt(onnx_path, engine_path, fp16, max_workspace_size)
            
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
                
            return success
            
        except Exception as e:
            print(f"Error converting PyTorch to TensorRT: {e}")
            return False
    
    def onnx_to_tensorrt(self, onnx_path: str, 
                        engine_path: str,
                        fp16: bool = True,
                        max_workspace_size: int = 1 << 30) -> bool:
        """
        Convert ONNX model to TensorRT engine
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            fp16: Use FP16 precision
            max_workspace_size: Maximum workspace size
            
        Returns:
            bool: Success status
        """
        try:
            # Create builder and network
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    print("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("Using FP16 precision for faster inference")
            
            # Build engine
            print(f"Building TensorRT engine... This may take a few minutes.")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                print("Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"TensorRT engine saved to: {engine_path}")
            return True
            
        except Exception as e:
            print(f"Error converting ONNX to TensorRT: {e}")
            return False


class TensorRTInference:
    """TensorRT inference engine wrapper"""
    
    def __init__(self, engine_path: str):
        """
        Initialize TensorRT inference engine
        
        Args:
            engine_path: Path to TensorRT engine file
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available")
            
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = None
        self.host_inputs = []
        self.host_outputs = []
        self.cuda_inputs = []
        self.cuda_outputs = []
        self.bindings = []
        
        self._load_engine()
        self._allocate_buffers()
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        try:
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()
                self.stream = cuda.Stream()
                print(f"TensorRT engine loaded from: {self.engine_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")
    
    def _allocate_buffers(self):
        """Allocate host and device buffers"""
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
            
            self.bindings.append(int(cuda_mem))
    
    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference with TensorRT engine
        
        Args:
            input_tensor: Input numpy array
            
        Returns:
            np.ndarray: Output predictions
        """
        try:
            # Copy input to host buffer
            np.copyto(self.host_inputs[0], input_tensor.ravel())
            
            # Copy host input to device
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            
            # Execute inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Copy device output to host
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            
            # Synchronize stream
            self.stream.synchronize()
            
            # Reshape output
            output_shape = self.engine.get_binding_shape(1)  # Assuming single output
            output = self.host_outputs[0].reshape(output_shape)
            
            return output
            
        except Exception as e:
            raise RuntimeError(f"TensorRT inference failed: {e}")
    
    def __del__(self):
        """Cleanup GPU memory"""
        if hasattr(self, 'cuda_inputs'):
            for cuda_mem in self.cuda_inputs + self.cuda_outputs:
                cuda_mem.free()


def optimize_model_with_tensorrt(model: torch.nn.Module, 
                                model_name: str,
                                weights_dir: str = './weights',
                                input_shape: Tuple[int, int, int, int] = (1, 3, 480, 640),
                                fp16: bool = True) -> Optional[str]:
    """
    Optimize a PyTorch model with TensorRT
    
    Args:
        model: PyTorch model to optimize
        model_name: Name for the TensorRT engine file
        weights_dir: Directory to save TensorRT engine
        input_shape: Input tensor shape
        fp16: Use FP16 precision
        
    Returns:
        str: Path to TensorRT engine file, None if failed
    """
    if not TRT_AVAILABLE:
        print("TensorRT not available. Skipping optimization.")
        return None
    
    engine_path = os.path.join(weights_dir, f"{model_name}_trt.engine")
    
    # Check if engine already exists
    if os.path.exists(engine_path):
        print(f"TensorRT engine already exists: {engine_path}")
        return engine_path
    
    try:
        optimizer = TensorRTOptimizer(input_shape)
        
        # Move model to GPU for optimization
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            print("CUDA not available. TensorRT optimization requires GPU.")
            return None
            
        model = model.to(device)
        
        print(f"Optimizing {model_name} with TensorRT...")
        success = optimizer.pytorch_to_tensorrt(model, engine_path, fp16)
        
        if success:
            print(f"TensorRT optimization completed: {engine_path}")
            return engine_path
        else:
            print(f"TensorRT optimization failed for {model_name}")
            return None
            
    except Exception as e:
        print(f"Error during TensorRT optimization: {e}")
        return None


if __name__ == '__main__':
    # Test TensorRT availability
    print(f"TensorRT available: {is_tensorrt_available()}")
    
    if TRT_AVAILABLE:
        print("TensorRT version:", trt.__version__)
    else:
        print("Install TensorRT with: pip install tensorrt pycuda") 