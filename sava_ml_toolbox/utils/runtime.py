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



# Simple class to mimic onnxruntime.NodeArg structure
class BindingInfo:
    def __init__(self, name: str, shape: tuple, dtype: np.dtype):
        self.name = name
        self.shape = shape
        # Convert numpy dtype to string representation similar to ONNX Runtime
        # This might need refinement based on exact types needed downstream
        if dtype == np.float32:
            self.type = 'tensor(float)'
        elif dtype == np.float16:
             self.type = 'tensor(float16)'
        elif dtype == np.int32:
            self.type = 'tensor(int32)'
        elif dtype == np.int64:
            self.type = 'tensor(int64)'
        elif dtype == np.uint8:
            self.type = 'tensor(uint8)'
        # Add other type mappings as needed
        else:
            self.type = f'tensor({str(dtype)})'
        self._dtype = dtype # Keep original numpy dtype if needed


# TensorRT Runtime implementation of BaseRuntime
class TensorRTRuntime(BaseRuntime):
    """
    TensorRT Runtime implementation adhering to the BaseRuntime interface.
    Loads and runs inference using a TensorRT engine file.
    """
    def __init__(
        self,
        path: str,
        trt_logger_level: trt.Logger.Severity = trt.Logger.WARNING,
        **kwargs
    ) -> None:
        """
        Initializes the TensorRT runtime.

        Args:
            path: Path to the serialized TensorRT engine file (.engine).
            providers: Ignored for TensorRT. Included for potential signature compatibility.
            trt_logger_level: Minimum severity level for the TensorRT logger.
        """
        super(TensorRTRuntime, self).__init__()

        if not path.endswith(".engine"):
             raise ValueError(f"Invalid path: '{path}'. TensorRT runtime requires a '.engine' file.")

        self.engine_path = path
        self.logger = trt.Logger(trt_logger_level)
        self.engine = None
        self.context = None
        self.stream = None

        # Store binding information (inputs/outputs)
        self._input_bindings = [] # List to store BindingInfo objects for inputs
        self._output_bindings = [] # List to store BindingInfo objects for outputs
        self._input_buffers = {} # Dict: {name: {'host': host_mem, 'device': device_mem}}
        self._output_buffers = {} # Dict: {name: {'host': host_mem, 'device': device_mem}}
        self.bindings = [] # List of device pointers (int) for execution context

        self._load_engine()
        self._allocate_buffers()

    def _load_engine(self):
        """Loads the TensorRT engine from the specified path."""
        self.logger.log(trt.Logger.INFO, f"Loading TensorRT engine from: {self.engine_path}")
        try:
            runtime = trt.Runtime(self.logger)
            with open(self.engine_path, "rb") as f:
                serialized_engine = f.read()
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if not self.engine:
                raise RuntimeError("Failed to deserialize the TensorRT engine.")

            # Create execution context AFTER checking for dynamic shapes if needed
            self.context = self.engine.create_execution_context()
            if not self.context:
                raise RuntimeError("Failed to create TensorRT execution context.")

            self.stream = cuda.Stream()
            self.logger.log(trt.Logger.INFO, "TensorRT engine and context loaded successfully.")
        except Exception as e:
            self.logger.log(trt.Logger.ERROR, f"Error loading TensorRT engine: {e}")
            raise

    def _allocate_buffers(self):
        """Allocates memory buffers for inputs and outputs on the device."""
        if not self.engine:
            raise RuntimeError("Engine not loaded before allocating buffers.")

        max_batch_size = 1 # Assuming max batch size is 1 for now

        for i in range(len(self.engine)):
            binding_name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i) # Note: This shape might include -1 for dynamic dimensions
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            is_input = self.engine.binding_is_input(i)

            actual_shape = shape

            # Calculate buffer size (handle potential batch dim if explicit)
            # Assuming max_batch_size is 1 for now
            if self.engine.has_implicit_batch_dimension:
                 # Shape from engine already includes spatial/channel dims for max_batch_size=1
                 size = trt.volume(actual_shape) * max_batch_size # Or use self.engine.max_batch_size
            else:
                 # Explicit batch dim, usually shape[0] == -1 for dynamic batch
                 # For allocation, use max batch size from profile or a predefined one
                 # Using max_batch_size=1 here:
                 size = trt.volume(actual_shape[1:]) # Calculate size without batch dim
                 size *= max_batch_size # Multiply by max batch size we want to support


            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem)) # Add device pointer to bindings list

            # Store buffer info and binding info
            binding_info = BindingInfo(binding_name, tuple(shape), dtype) # Use original shape for info
            buffer_info = {'host': host_mem, 'device': device_mem, 'shape': tuple(shape), 'dtype': dtype}

            if is_input:
                self._input_bindings.append(binding_info)
                self._input_buffers[binding_name] = buffer_info
            else:
                self._output_bindings.append(binding_info)
                self._output_buffers[binding_name] = buffer_info

        self.logger.log(trt.Logger.INFO, "Allocated TensorRT input/output buffers.")

    def run(
        self,
        input_data: Dict[str, np.ndarray],
        output_names: List[str] = None,
    ) -> List[np.ndarray]:
        """
        Perform TensorRT inference.

        Args:
            input_data (Dict[str, np.ndarray]): Dictionary mapping input names to numpy arrays.
                                                Data should be preprocessed and match model requirements.
            output_names (List[str]): Optional. If provided, can be used to verify output order (currently ignored).

        Returns:
            List[np.ndarray]: A list of output numpy arrays in the engine's output binding order.
        """
        if not self.context:
            raise RuntimeError("TensorRT context not initialized.")
        if not self.engine:
            raise RuntimeError("TensorRT engine not initialized.")

        # --- Prepare Inputs ---
        for name, array in input_data.items():
            if name not in self._input_buffers:
                raise ValueError(f"Input name '{name}' not found in engine bindings.")

            buffer_info = self._input_buffers[name]
            expected_shape = buffer_info['shape'] # Shape may include batch dim or -1
            expected_dtype = buffer_info['dtype']

            # --- Basic Input Validation (Adapt as needed) ---
            # Check dtype
            if array.dtype != expected_dtype:
                 self.logger.log(trt.Logger.WARNING, f"Input '{name}' dtype mismatch: Got {array.dtype}, Expected {expected_dtype}. Attempting cast.")
                 array = array.astype(expected_dtype)

            # Check shape (needs refinement for dynamic shapes and explicit/implicit batch)
            # This is a simplified check assuming batch size 1 and matching non-batch dimensions
            if not self.engine.has_implicit_batch_dimension:
                 # Explicit batch dimension (e.g., [-1, 3, 224, 224])
                 # Assuming input array has batch dim = 1
                 if array.shape[1:] != expected_shape[1:]:
                     raise ValueError(f"Input '{name}' shape mismatch: Got {array.shape}, Expected non-batch shape {expected_shape[1:]}")
                 # If dynamic shapes, potentially set context binding shape here
                 # self.context.set_binding_shape(binding_index_for_name, array.shape)
            else:
                 # Implicit batch dimension
                 if array.shape != expected_shape:
                     raise ValueError(f"Input '{name}' shape mismatch: Got {array.shape}, Expected {expected_shape}")


            # Copy data to pagelocked host buffer
            np.copyto(buffer_info['host'], array.ravel()) # Flatten and copy

            # Transfer input data to the GPU asynchronously.
            cuda.memcpy_htod_async(buffer_info['device'], buffer_info['host'], self.stream)

        # --- Execute Inference ---
        # Note: execute_async_v2 is preferred for explicit batch & dynamic shapes
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # --- Retrieve Outputs ---
        results = {}
        for name, buffer_info in self._output_buffers.items():
            # Transfer predictions back from the GPU asynchronously.
            cuda.memcpy_dtoh_async(buffer_info['host'], buffer_info['device'], self.stream)
            # Store the host buffer (reshaping happens after sync)
            results[name] = buffer_info

        # Synchronize the stream to ensure completion
        self.stream.synchronize()

        # Reshape outputs and return in the correct order
        # The order should match self._output_bindings
        final_outputs = []
        for binding_info in self._output_bindings:
            name = binding_info.name
            buffer_info = results[name]
            # Shape needs careful handling for dynamic outputs
            # Assuming output shape is fixed or correctly inferred for batch size 1
            output_shape = buffer_info['shape']
            # If explicit batch, shape might need adjustment (e.g., add batch dim 1)
            if not self.engine.has_implicit_batch_dimension:
                 # Example: If output shape from engine is (C, H, W), reshape to (1, C, H, W)
                 batch_size = 1 # Assuming batch size 1 for now
                 actual_output_shape = (batch_size, ) + output_shape[1:] # Add batch dim
            else:
                 actual_output_shape = output_shape

            # Ensure the buffer has the right number of elements before reshaping
            expected_elements = np.prod(actual_output_shape)
            if buffer_info['host'].size != expected_elements:
                # This might happen with dynamic shapes if allocation was based on max size
                # Need to determine the actual output size based on inference results if possible
                # Or ensure allocation/reshaping logic correctly handles dynamic cases
                 self.logger.log(trt.Logger.WARNING, f"Output '{name}' element count mismatch ({buffer_info['host'].size} vs {expected_elements}). Using available elements.")
                 # Slice the buffer if necessary, though this requires knowing the actual output size
                 reshaped_output = buffer_info['host'][:expected_elements].reshape(actual_output_shape)

            else:
                reshaped_output = buffer_info['host'].reshape(actual_output_shape)

            final_outputs.append(reshaped_output.copy()) # Copy to avoid returning internal buffer


        return final_outputs

    def get_inputs(self) -> List[BindingInfo]:
        """
        Get the input details of the TensorRT engine.

        Returns:
            List[BindingInfo]: List of input binding details (name, shape, type).
        """
        return self._input_bindings

    def get_outputs(self) -> List[BindingInfo]:
        """
        Get the output details of the TensorRT engine.

        Returns:
            List[BindingInfo]: List of output binding details (name, shape, type).
        """
        return self._output_bindings

    def __del__(self):
        """Clean up CUDA resources."""
        # pycuda.autoinit handles context cleanup.
        # Explicitly free allocated memory if needed, though context destruction should handle it.
        # If stream is created manually, it might need destroying.
        pass