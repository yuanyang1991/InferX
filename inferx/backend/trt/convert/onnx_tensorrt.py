from pathlib import Path
from . import dynamic_axis
import tensorrt as trt


class ONNX2TensorRT:

    def __init__(self, onnx_path, dynamic_axes: list[dynamic_axis.DynamicAxisInfo] = None):
        self._onnx_path = onnx_path
        self._dynamic_axes = dynamic_axes

    def convert(self, engine_path):
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        parse_result = parser.parse_from_file(self._onnx_path)

        if not parse_result:
            logger.log(trt.Logger.ERROR, "onnx parse error")
            for err_idx in parser.num_errors:
                logger.log(trt.Logger.ERROR, parser.get_error(err_idx))
            raise RuntimeError("Failed to parse ONNX model")

        config: trt.IBuilderConfig = builder.create_builder_config()
        if self._dynamic_axes:
            profile: trt.IOptimizationProfile = builder.create_optimization_profile()
            for dynamic_axis in self._dynamic_axes:
                profile.set_shape(dynamic_axis.input_name, dynamic_axis.min_shape, dynamic_axis.opt_shape,
                                  dynamic_axis.max_shape)
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        parent_dir = Path(engine_path).parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
