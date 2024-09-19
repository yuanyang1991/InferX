from pathlib import Path

import tensorrt as trt

from . import dynamic_axis


class ONNX2TensorRT:

    def __init__(self, onnx_path, dynamic_axes: list[dynamic_axis.DynamicAxisInfo] = None):
        self._onnx_path = onnx_path
        self._dynamic_axes = dynamic_axes

    def convert(self, engine_path):

        def is_dim_dynamic(dim):
            is_dim_str = not isinstance(dim, int)
            return dim is None or is_dim_str or dim < 0

        def is_dynamic_tensor(tensor: trt.ITensor) -> bool:
            return len([dim for dim in tensor.shape if is_dim_dynamic(dim)]) > 0

        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network: trt.INetworkDefinition = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        parse_result = parser.parse_from_file(self._onnx_path)

        num_outputs = network.num_outputs
        for idx in range(num_outputs):
            tensor = network.get_output(idx)
            print(f"output:{tensor.name}")

        if not parse_result:
            logger.log(trt.Logger.ERROR, "onnx parse error")
            for err_idx in parser.num_errors:
                logger.log(trt.Logger.ERROR, parser.get_error(err_idx))
            raise RuntimeError("Failed to parse ONNX model")

        config: trt.IBuilderConfig = builder.create_builder_config()
        self._dynamic_axes = {item.input_name: item for item in self._dynamic_axes}
        if self._dynamic_axes:
            logger.log(trt.Logger.INFO, "start to bind profile config")
            profile: trt.IOptimizationProfile = builder.create_optimization_profile()
            num_inputs = network.num_inputs
            for idx in range(num_inputs):
                tensor: trt.ITensor = network.get_input(idx)
                if tensor.is_shape_tensor:
                    if tensor.name in self._dynamic_axes:
                        dynamic_axis = self._dynamic_axes[tensor.name]
                        profile.set_shape_input(dynamic_axis.input_name, dynamic_axis.min_shape, dynamic_axis.opt_shape,
                                                dynamic_axis.max_shape)
                    else:
                        raise RuntimeError(f"{tensor.name} is shape tensor, must provide profile values")
                elif is_dynamic_tensor(tensor):
                    if tensor.name in self._dynamic_axes:
                        dynamic_axis = self._dynamic_axes[tensor.name]
                        profile.set_shape(dynamic_axis.input_name, dynamic_axis.min_shape, dynamic_axis.opt_shape,
                                          dynamic_axis.max_shape)
                    else:
                        raise RuntimeError(f"{tensor.name} is dynamic tensor, must provide profile shapes")

                else:
                    logger.log(trt.Logger.INFO, f"tensor: {tensor.name} is static tensor")
            result_code = config.add_optimization_profile(profile)
            if result_code == -1:
                raise RuntimeError("bind profile error, input is not valid, check that")
            logger.log(trt.Logger.INFO, "complete to bind profile config")

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("build network error")
        parent_dir = Path(engine_path).parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        logger.log(trt.Logger.INFO, "complete build trt engine")
