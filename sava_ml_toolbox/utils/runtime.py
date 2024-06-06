from typing import List

import numpy as np
import onnxruntime as rt


# Base class for Runtime
class BaseRuntime:
    def __init__(self) -> None:
        pass

    def run(self) -> None:
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

    def run(self, input_data: np.array) -> np.array:
        """Perform ONNX Runtime inference.

        Args:
            input_data (np.array): The input data for inference.

        Returns:
            np.array: The output prediction from the ONNX Runtime.
        """
        # Prepare input data for the ONNX Runtime session
        ort_inputs = {
            self.ort_session.get_inputs()[0]
            .name: np.array(input_data)
            .astype(np.float32)
        }
        # Run the inference using ONNX Runtime
        ort_outs = self.ort_session.run(None, ort_inputs)
        # Return the output prediction
        return ort_outs
