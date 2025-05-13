from typing import List, Dict, Any

import numpy as np
import onnxruntime as rt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# Base class for Runtime
class BaseRuntime:
    def __init__(self) -> None:
        pass

    def run(self) -> None:
        pass

    def get_inputs(self) -> List:
        pass

    def get_outputs(self) -> List:
        pass


# ONNX Runtime implementation of BaseRuntime
class ONNXRuntime(BaseRuntime):
    def __init__(
        self,
        path: str,
        providers: List[str] = [
            "CPUExecutionProvider",
        ],
    ) -> None:
        super(ONNXRuntime, self).__init__()
        self.ort_session = rt.InferenceSession(path, providers=providers)

    def run(
        self,
        input_data: dict,
        output_names: List[str] = None,
    ) -> np.array:
        """Perform ONNX Runtime inference.

        Args:
            input_data (np.array): The input data for inference.

        Returns:
            np.array: The output prediction from the ONNX Runtime.
        """
        # Return the output prediction
        return self.ort_session.run(output_names, input_data)

    def get_inputs(self) -> List:
        """Get the input details of the ONNX model.

        Returns:
            List: List of input details.
        """
        return self.ort_session.get_inputs()

    def get_outputs(self) -> List:
        """Get the output details of the ONNX model.

        Returns:
            List: List of output details.
        """
        return self.ort_session.get_outputs()



import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Important for initializing CUDA context
import numpy as np
from typing import Dict, Union, List

class TensorRTRuntime(BaseRuntime): # Inherit from BaseRuntime
    TRT_TO_NP_DTYPE = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_
    }

    def __init__(self, path: str, batch_size: int = 1): # Added batch_size to constructor
        super().__init__()
        self.engine_path = path # Store engine_path for clarity
        self.batch_size = batch_size
        self.logger = trt.Logger(trt.Logger.WARNING)

        try:
            with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}: {e}")

        if not self.engine:
            raise RuntimeError(f"Failed to deserialize engine from {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.h_inputs: List[np.ndarray] = []
        self.d_inputs: List[cuda.DeviceAllocation] = []
        self.input_np_dtypes: List[np.dtype] = []
        self.h_outputs: List[np.ndarray] = []
        self.d_outputs: List[cuda.DeviceAllocation] = []
        self.output_np_dtypes: List[np.dtype] = []
        self.bindings: List[int] = []
        
        self.input_tensor_names: List[str] = []
        self.output_tensor_names: List[str] = []

        # For get_inputs() / get_outputs() compatibility with ONNXRuntime's NodeArg-like objects
        self._inputs_meta = []
        self._outputs_meta = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = list(self.engine.get_tensor_shape(name))
            trt_engine_dtype = self.engine.get_tensor_dtype(name)
            np_dtype = self.TRT_TO_NP_DTYPE.get(trt_engine_dtype)

            if np_dtype is None:
                raise ValueError(f"Unsupported TensorRT DType {trt_engine_dtype} for tensor '{name}'.")

            if not self.engine.has_implicit_batch_dimension:
                if shape[0] == -1: shape[0] = self.batch_size
                elif shape[0] != self.batch_size :
                     print(f"Warning: Tensor '{name}' (batch_dim={shape[0]}) vs requested batch_size={self.batch_size}.")
                     # Potentially adjust shape[0] = self.batch_size if engine supports dynamic batch for this profile

            # Create a mock NodeArg-like object for get_inputs/get_outputs
            class MockNodeArg:
                def __init__(self, name, shape, dtype_str):
                    self.name = name
                    self.shape = shape # This shape from engine might have -1 for dynamic dims
                    self.type = dtype_str # e.g., "tensor(float)"

            # Convert trt_dtype to a string representation like ONNX runtime
            dtype_str = f"tensor({str(trt_engine_dtype).lower().split('.')[-1]})" # Simplistic conversion
            
            # The actual allocated shape for host/device buffers uses the concrete batch_size
            allocated_shape = shape.copy() # Make a copy to modify for allocation
            if not self.engine.has_implicit_batch_dimension and allocated_shape[0] == -1:
                 allocated_shape[0] = self.batch_size # Use actual batch_size for allocation

            device_mem_size = trt.volume(allocated_shape) * trt_engine_dtype.itemsize

            if is_input:
                self.input_tensor_names.append(name)
                self.h_inputs.append(np.empty(allocated_shape, dtype=np_dtype))
                d_input_buffer = cuda.mem_alloc(device_mem_size)
                self.d_inputs.append(d_input_buffer)
                self.input_np_dtypes.append(np_dtype)
                self.bindings.append(int(d_input_buffer))
                if not self.engine.has_implicit_batch_dimension:
                    self.context.set_tensor_address(name, int(d_input_buffer))
                self._inputs_meta.append(MockNodeArg(name, shape, dtype_str))
            else:
                self.output_tensor_names.append(name)
                self.h_outputs.append(np.empty(allocated_shape, dtype=np_dtype))
                d_output_buffer = cuda.mem_alloc(device_mem_size)
                self.d_outputs.append(d_output_buffer)
                self.output_np_dtypes.append(np_dtype)
                self.bindings.append(int(d_output_buffer))
                if not self.engine.has_implicit_batch_dimension:
                    self.context.set_tensor_address(name, int(d_output_buffer))
                self._outputs_meta.append(MockNodeArg(name, shape, dtype_str))
        
        self.stream = cuda.Stream()
        print(f"TensorRT Runtime Initialized for {self.engine_path}")
        for meta in self._inputs_meta: print(f"  Input: {meta.name}, Shape from Engine: {meta.shape}, Type: {meta.type}")
        for meta in self._outputs_meta: print(f"  Output: {meta.name}, Shape from Engine: {meta.shape}, Type: {meta.type}")

    def get_inputs(self) -> List[Any]: # Return type matches ONNXRuntime's get_inputs()
        return self._inputs_meta

    def get_outputs(self) -> List[Any]: # Return type matches ONNXRuntime's get_outputs()
        return self._outputs_meta

    def run(self, input_data: Dict[str, np.ndarray], output_names: List[str] = None) -> List[np.ndarray]:
        """
        Performs inference. output_names is ignored for TRT as outputs are fetched by order/all.
        Returns a list of numpy arrays in the order of engine outputs.
        """
        if not isinstance(input_data, dict):
            # If YOLOv8Seg._inference provides a single array, wrap it in a dict
            # based on the first input tensor name.
            if isinstance(input_data, np.ndarray) and self.input_tensor_names:
                input_data = {self.input_tensor_names[0]: input_data}
            else:
                raise TypeError("TensorRT run expects input_data as Dict[str, np.ndarray] or single np.ndarray for single-input models.")

        for i, name in enumerate(self.input_tensor_names):
            if name not in input_data:
                raise ValueError(f"Missing data for input tensor: {name}")
            
            data_array = np.asarray(input_data[name], dtype=self.input_np_dtypes[i])
            
            # Basic shape adjustment for batching (more robust logic might be needed)
            if self.h_inputs[i].shape[0] == 1 and data_array.ndim == (self.h_inputs[i].ndim - 1):
                 data_array = np.expand_dims(data_array, axis=0)
            elif self.h_inputs[i].ndim == (data_array.ndim -1) and data_array.shape[0] == 1:
                 data_array = data_array.squeeze(0)

            if data_array.shape != self.h_inputs[i].shape:
                # If engine has dynamic input shapes, this is where set_input_shape would be called on context
                # For now, we assume shapes match after preproc or engine is static for the profile.
                print(f"Warning: Input for '{name}' shape {data_array.shape} vs allocated {self.h_inputs[i].shape}. Ensure consistency or handle dynamic shapes.")
            
            np.copyto(self.h_inputs[i], data_array.ravel())
            cuda.memcpy_htod_async(self.d_inputs[i], self.h_inputs[i], self.stream)

        if self.engine.has_implicit_batch_dimension:
             self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v3(stream_handle=self.stream.handle)

        for i in range(len(self.output_tensor_names)):
            cuda.memcpy_dtoh_async(self.h_outputs[i], self.d_outputs[i], self.stream)
        self.stream.synchronize()
        
        # Return a list of numpy arrays, similar to ONNXRuntime session.run()
        return [out.copy() for out in self.h_outputs]

    def release(self):
        print(f"Releasing TensorRT resources for {self.engine_path}...")
        for d_mem_list in [self.d_inputs, self.d_outputs]:
            for d_mem in d_mem_list:
                if d_mem: 
                    try: d_mem.free()
                    except Exception as e: print(f"Error freeing TRT device_mem: {e}")
        self.d_inputs, self.d_outputs, self.h_inputs, self.h_outputs = [], [], [], []
        if hasattr(self, 'context'): del self.context
        if hasattr(self, 'engine'): del self.engine
        print("TensorRT resources released.")

    def __del__(self):
        self.release()