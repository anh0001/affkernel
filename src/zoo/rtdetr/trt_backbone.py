"""Optional TensorRT backbone runtime.

The dynamic-kernel affordance head cannot be exported to ONNX, but the
*backbone* is plain convolutions and exports cleanly. Running it as a
TensorRT fp16 engine while the encoder / decoder / affordance head stay in
PyTorch cuts the largest single term of the forward pass (~3x on Jetson AGX
Orin) without touching the parts ONNX cannot represent.

TensorRT is an optional dependency; nothing here is imported unless a caller
explicitly asks for the TRT path. Build an engine with
``tools/build_trt_backbone.py`` first.
"""

from __future__ import annotations

import torch

__all__ = ["TRTBackbone", "trt_dtype_to_torch"]


def trt_dtype_to_torch(dtype):
    """Map a TensorRT DataType to the matching torch dtype."""
    import tensorrt as trt

    table = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if dtype not in table:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype}")
    return table[dtype]


class TRTBackbone:
    """Runs a serialized TensorRT backbone engine on the current CUDA stream.

    The returned feature tensors alias persistent device buffers, so a
    downstream ``torch.cuda.CUDAGraph`` can capture the rest of the network
    against them and only the engine has to be re-run per frame.

    Args:
        plan_path: path to a serialized engine (``.plan``).
    """

    def __init__(self, plan_path: str):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        with open(plan_path, "rb") as fh:
            engine = trt.Runtime(logger).deserialize_cuda_engine(fh.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {plan_path}")
        self.engine = engine
        self.context = engine.create_execution_context()

        self.input_name = None
        self.output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if self.input_name is not None:
                    raise RuntimeError("Backbone engine must have exactly one input")
                self.input_name = name
            else:
                self.output_names.append(name)

        def _alloc(name):
            return torch.zeros(
                tuple(engine.get_tensor_shape(name)),
                dtype=trt_dtype_to_torch(engine.get_tensor_dtype(name)),
                device="cuda",
            )

        self.input_buffer = _alloc(self.input_name)
        self.outputs = [_alloc(n) for n in self.output_names]
        self.context.set_tensor_address(self.input_name, self.input_buffer.data_ptr())
        for name, buf in zip(self.output_names, self.outputs):
            self.context.set_tensor_address(name, buf.data_ptr())

    @property
    def input_dtype(self) -> torch.dtype:
        return self.input_buffer.dtype

    @property
    def input_size(self) -> int:
        return int(self.input_buffer.shape[-1])

    def __call__(self, x: torch.Tensor):
        """Run the engine on ``x`` and return the persistent output buffers."""
        self.input_buffer.copy_(x)
        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self.outputs
