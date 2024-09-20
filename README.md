# InferX

# Introduce

With InferX, you can easily deploy your models using TensorRT or ONNX.

# Usage

## 1. build config

```python
from inferx.backend.trt.convert.dynamic_axis import DynamicAxisInfo
from inferx.config import Config, Backend

dynamic_axes = [DynamicAxisInfo("point_coords", min_value=(1, 2, 2), opt_value=(1, 2, 2), max_value=(1, 20, 2)),
                DynamicAxisInfo("point_labels", min_value=(1, 2), opt_value=(1, 2), max_value=(1, 20)),
                DynamicAxisInfo("orig_im_size", min_value=(1000, 1000), opt_value=(2000, 2000), max_value=(5000, 5000))
                ]
config = Config("sam_decoder.onnx", "sam_decoder.trt", backend=Backend.TensorRT, dynamic_axes=dynamic_axes)
```

## 2. build runtime
```python
from inferx.runtime import Runtime

runtime = Runtime(config)
```

## 3. execute
```python
dummy_inputs = {
    "image_embeddings": np.random.randn(1, 256, 64, 64).astype(np.float32),
    "point_coords": np.random.randint(low=0, high=1024, size=(1, 5, 2)).astype(np.float32),
    "point_labels": np.random.randint(low=0, high=4, size=(1, 5)).astype(np.float32),
    "mask_input": np.random.randn(1, 1, 256, 256).astype(np.float32),
    "has_mask_input": np.array([1], dtype=np.float32),
    "orig_im_size": np.array([6000, 6000], dtype=np.int32)
}
outputs = runtime.run(dummy_inputs)
```