from onnx_tensorrt_help.miotf2onnx_util import export_unipredict_fused_json_to_onnx

if __name__ == "__main__":
    export_unipredict_fused_json_to_onnx(
        json_path="./test/dynamic_json_config.json",
        is_export_onnx=True,
    )
