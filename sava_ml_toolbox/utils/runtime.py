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

# Updated TensorRT implementation of BaseRuntime
class TensorRTRuntime(BaseRuntime):
    """ TensorRT Runtime Implementation (Updated) """

    def __init__(self, engine_path: str, profile_idx: int = 0) -> None:
        """
        Initializes the TensorRT runtime.

        Args:
            engine_path: Path to the serialized TensorRT engine file (.engine).
            profile_idx: The optimization profile index to use (default: 0).
                         Relevant for engines built with dynamic shapes.
        """
        super().__init__()
        self.engine_path = engine_path
        self.profile_idx = profile_idx
        self.logger = trt.Logger(trt.Logger.WARNING) # Or INFO, ERROR, etc.
        self.runtime = trt.Runtime(self.logger)

        # Load and deserialize the engine
        try:
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
            if not self.engine:
                raise RuntimeError(f"Failed to deserialize engine from {engine_path}")
        except FileNotFoundError:
             raise FileNotFoundError(f"Engine file not found at {engine_path}")
        except Exception as e:
             raise RuntimeError(f"Error loading TensorRT engine: {e}")

        # Verify profile index
        if not (0 <= self.profile_idx < self.engine.num_optimization_profiles):
             raise ValueError(f"Invalid profile_idx {self.profile_idx}. Engine has {self.engine.num_optimization_profiles} profiles.")

        # Create execution context
        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create TensorRT execution context.")

        # Activate optimization profile
        self.context.set_optimization_profile_async(self.profile_idx, cuda.Stream().handle) # Use a temporary stream handle

        # Store tensor specifications and allocate buffers
        self.stream = cuda.Stream()
        self.bindings = []
        self.host_buffers = {} # Store host buffers by name
        self.device_buffers = {} # Store device buffers by name (int address)
        self.tensor_specs = {} # Store name, shape, dtype, is_input

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name) # Engine shape (can have -1)

            # Determine allocation shape (use max profile shape for dynamic dims)
            alloc_shape = shape
            if -1 in shape:
                profile_shapes = self.engine.get_profile_shape(self.profile_idx, name)
                # profile_shapes: (min_shape, opt_shape, max_shape)
                alloc_shape = profile_shapes[2] # Use max shape for allocation
                print(f"Info: Tensor '{name}' is dynamic. Allocating buffer for max shape {alloc_shape}.")

            if tuple(alloc_shape) == (-1,): # Handle scalar outputs if needed, allocate at least 1 element
                alloc_shape = (1,)
                print(f"Warning: Tensor '{name}' has shape (-1,). Allocating buffer for shape (1,).")


            # Allocate memory
            size = trt.volume(alloc_shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            self.host_buffers[name] = host_mem
            self.device_buffers[name] = int(device_mem) # Store address
            self.tensor_specs[name] = {'shape': shape, 'dtype': dtype, 'is_input': is_input, 'index': i}

        # Separate input/output specs for get_inputs/get_outputs compatibility
        self._input_specs = [
            {'name': name, 'shape': spec['shape'], 'dtype': spec['dtype']}
            for name, spec in self.tensor_specs.items() if spec['is_input']
        ]
        self._output_specs = [
            {'name': name, 'shape': spec['shape'], 'dtype': spec['dtype']}
            for name, spec in self.tensor_specs.items() if not spec['is_input']
        ]


    def run(self, input_data: Dict[str, np.array], output_names: List[str] = None) -> Dict[str, np.array]:
        """
        Perform inference using the TensorRT engine.

        Args:
            input_data: Dictionary mapping input tensor names to NumPy arrays.
            output_names: Ignored in this implementation. Kept for interface compatibility.

        Returns:
            Dict[str, np.array]: Dictionary mapping output tensor names to NumPy arrays.
        """
        if not isinstance(input_data, dict):
            raise TypeError(f"Expected input_data to be a Dict[str, np.array], but got {type(input_data)}")

        # --- Input Handling ---
        for name, array in input_data.items():
            spec = self.tensor_specs.get(name)
            if spec is None or not spec['is_input']:
                raise ValueError(f"Input tensor name '{name}' not found or is not an input.")

            # Check and cast dtype
            if array.dtype != spec['dtype']:
                # print(f"Warning: Input '{name}' dtype mismatch ({array.dtype} vs {spec['dtype']}). Casting...")
                array = array.astype(spec['dtype'])

            # Set input shape in context if dynamic
            if -1 in spec['shape']:
                if not self.context.set_input_shape(name, array.shape):
                     raise ValueError(f"Failed to set input shape {array.shape} for dynamic tensor '{name}'. Check profile constraints.")

            # Check if runtime shape matches (after potential context setting)
            runtime_shape = tuple(self.context.get_tensor_shape(name))
            if array.shape != runtime_shape:
                # Allow for broadcasting if batch dim is 1/-1 and input has no batch dim
                if len(runtime_shape) == len(array.shape) + 1 and runtime_shape[0] in [1, -1]:
                    print(f"Info: Input '{name}' automatically adding batch dimension.")
                    array = np.expand_dims(array, axis=0)
                    if array.shape != runtime_shape: # Re-check after adding dim
                         raise ValueError(f"Input '{name}' shape {input_data[name].shape} (original) / {array.shape} (adjusted) is incompatible with expected runtime shape {runtime_shape}")
                else:
                    raise ValueError(f"Input '{name}' shape {array.shape} is incompatible with expected runtime shape {runtime_shape}")

            # Copy data HtoD
            host_buf = self.host_buffers[name]
            # Ensure host buffer is large enough (especially if input size varies dynamically)
            if array.nbytes > host_buf.nbytes:
                 raise ValueError(f"Input data for '{name}' ({array.nbytes} bytes) exceeds allocated host buffer size ({host_buf.nbytes} bytes). Engine might need rebuild with larger max profile shape.")
            # Use slicing to avoid reallocating host_buf if possible
            host_buf_view = host_buf.reshape(host_buf.shape) # Get a writable view
            np.copyto(host_buf_view[:array.size], array.ravel()) # Copy flattened data
            cuda.memcpy_htod_async(self.device_buffers[name], host_buf_view, self.stream)


        # --- Set Binding Addresses in Context ---
        # This is crucial as device buffer addresses might change if reallocated
        for name, addr in self.device_buffers.items():
             self.context.set_tensor_address(name, addr)

        # --- Execute Inference ---
        if not self.context.execute_async_v2(bindings=[], stream_handle=self.stream.handle):
             raise RuntimeError("TensorRT inference failed.")

        # --- Output Handling ---
        outputs = {}
        for name, spec in self.tensor_specs.items():
            if not spec['is_input']:
                host_buf = self.host_buffers[name]
                device_buf_addr = self.device_buffers[name]
                # Get actual output shape for this run
                output_shape = tuple(self.context.get_tensor_shape(name))

                # Calculate required bytes based on actual output shape
                expected_bytes = trt.volume(output_shape) * np.dtype(spec['dtype']).itemsize

                # Check if host buffer is large enough
                if expected_bytes > host_buf.nbytes:
                    raise RuntimeError(f"Output tensor '{name}' size ({expected_bytes} bytes) exceeds allocated host buffer ({host_buf.nbytes} bytes). Max profile shape might be too small.")

                # Copy DtoH
                cuda.memcpy_dtoh_async(host_buf, device_buf_addr, self.stream)

                # Store shaped view of the relevant part of the buffer
                # Important: create a copy so the user gets an independent array
                num_elements = trt.volume(output_shape)
                outputs[name] = host_buf[:num_elements].reshape(output_shape).copy()


        self.stream.synchronize()
        return outputs

    def get_inputs(self) -> List[Dict[str, Any]]:
        """
        Get specifications (name, shape, dtype) of the engine's input tensors.
        Shape reflects the engine's definition (may contain -1 for dynamic).
        """
        return self._input_specs

    def get_outputs(self) -> List[Dict[str, Any]]:
        """
        Get specifications (name, shape, dtype) of the engine's output tensors.
        Shape reflects the engine's definition (may contain -1 for dynamic).
        """
        return self._output_specs

    # Optional: Cleanup method
    def cleanup(self):
        """Manually clean up CUDA memory."""
        print("Cleaning up TensorRT resources...")
        for name, addr in self.device_buffers.items():
            try:
                # Need a way to reconstruct CuDeviceMemory object to call free()
                # This is tricky as we only stored the address (int).
                # pycuda might handle this via context destruction with autoinit,
                # but explicit free is safer if references are held elsewhere.
                # For now, rely on pycuda.autoinit or manual context management.
                pass # cuda.mem_free(addr) # This function expects address, maybe works? Needs testing.
            except Exception as e:
                print(f"Warning: Error freeing device memory for {name}: {e}")
        self.stream = None # Release stream reference
        print("TensorRT resource cleanup finished (Device memory freeing depends on CUDA context management).")

    # Make it usable with 'with' statement for automatic cleanup
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup(