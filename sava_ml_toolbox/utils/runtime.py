from typing import List

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


# TensorRT Runtime implementation of BaseRuntime
class TensorRTRuntime(BaseRuntime):

    def __init__(
        self,
        path:str
    ) -> None:

        super(TensorRTRuntime, self).__init__()

        # Load the TensorRT engine
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)

        with open(path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()


        # Allocate device memory for inputs and outputs
        self.bindings = []
        self.input_shape = None
        self.output_shapes = []
        self.input_idx = None
        self.output_idx = []


        # Prepare input and output bindings
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            # Allocate dvice memory
            mem = cuda.mem_alloc(size * dtype.itemsize)
            self.bindings.append(int(mem))

            if self.engine.binding_is_input(i):
                self.input_shape = shape
                self.input_idx = i
            else:
                self.output_shapes.append(shape)
                self.output_idx.append(i)

        # create cuda stream
        self.stream = cuda.Stream()

    def run(
        self,
        input_data:dict,
        output_names:List[str] = None,
    ) -> np.array:
        """Perform TensorRT inference.

        Args:
            input_data (dict): The input data for inference.
            output_names (List[str], optional): Output names (not used in TensorRT).

        Returns:
            np.array: The output prediction from the TensorRT engine.
        """

        # Get input name
        input name = next(iter(input_data))
        input_data = input_data[input_name]


        # Transfer input data to device
        cuda.memcpy_htod_asyn(
            self.beindings[self.input_idx],
            input_data_np.ravel(),
            self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Transfer outputs from device
        outputs = []
        for i, shape in zip(self.output_idx, self.output_shapes):
            output = np.empty(shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(output, self.bindings[i], self.stream)
            outputs.append(output)

        # Synchronize the stream
        self.stream.synchronize()

        return outputs



    def get_inputs(self) -> List:
        """Get the input details of the TensorRT model.

        Returns:
            List: List of input details.
        """
        inputs = []

        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                input_info = {
                    'name': self.engine.get_binding_name(i),
                    'shape': tuple(self.engine.get_binding_shape(i)),
                    'dtype': trt.nptype(self.engine.get_binding_dtype(i)),
                }
                inputs.append(input_info)

        return inputs


    def get_outputs(self) -> List:
        """Get the output details of the TensorRT model.

        Returns:
            List: List of output details.
        """
        outputs = []
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                output_info = {
                    'name': self.engine.get_binding_name(i),
                    'shape': tuple(self.engine.get_binding_shape(i)),
                    'dtype': trt.nptype(self.engine.get_binding_dtype(i))
                }
                outputs.append(output_info)
        return outputs

