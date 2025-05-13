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



# TensorRT Runtime implementation of BaseRuntime
class TensorRTRuntime(BaseRuntime):

    def __init__(
        self,
        path: str,
    ) -> None:

        super(TensorRTRuntime, self).__init__()

        f = open(path, "rb")
        self.runtime =trt.Runtime(trt.Logger(trt.Logger.WARNING))

        self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def run(
        self,
        input_data: dict,
        output_names: List[str] = None,
    ) -> np.ndarray:
    
