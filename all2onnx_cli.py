#!/usr/bin/env python3
# coding: utf-8

"""all2onnx 命令行工具

- export-json: 从 dynamic json 批量导出 uni_predict_fused* -> onnx base64，可选导出 .onnx
- convert-b64: 单个 mio graph(base64://...) -> onnx base64，可选导出 .onnx
- to-onnx: 使用 graph.pb + yaml 配置转换为 onnx（可选导出 .onnx / 或输出 base64）

说明：
- 导出 onnx 时：onnx 文件与 external data（如 {name}.bin）会落在同一个 output_dir 下。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from typing import Optional

from onnx_tensorrt_help.miotf2onnx_util import (
    export_unipredict_fused_json_to_onnx,
    miotfb64_to_onnxb64,
    miotf_to_onnxb64,
)
from onnx_tensorrt_help.miotf_util import get_inputs_outputs_params_from_yaml


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def dump_text(s: str, out: Optional[str]) -> None:
    if out:
        ensure_dir(os.path.dirname(out) or ".")
        with open(out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        sys.stdout.write(s + "\n")


def cmd_export_json(a: argparse.Namespace) -> int:
    ret = export_unipredict_fused_json_to_onnx(
        json_path=a.json,
        output_dir=a.output_dir,
        opset=a.opset,
        is_export_onnx=a.export_onnx,
        overwrite=True,
    )

    if a.output_json:
        # 输出“替换 graph 字段后的完整 json”
        from onnx_tensorrt_help.miotf2onnx_util import replace_unipredict_fused_graph_in_json_obj

        with open(a.json, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        new_cfg = replace_unipredict_fused_graph_in_json_obj(raw_cfg, ret)

        ensure_dir(os.path.dirname(a.output_json) or ".")
        with open(a.output_json, "w", encoding="utf-8") as f:
            json.dump(new_cfg, f, ensure_ascii=False, indent=2)

    if a.output_map_json:
        ensure_dir(os.path.dirname(a.output_map_json) or ".")
        with open(a.output_map_json, "w", encoding="utf-8") as f:
            json.dump(ret, f, ensure_ascii=False, indent=2)

    if a.print:
        for name, b64 in ret.items():
            sys.stdout.write(f"{name}\t{b64[:80]}...\n")

    return 0


def cmd_convert_b64(a: argparse.Namespace) -> int:
    output_path = (
        os.path.join(a.output_dir, f"{a.onnx_name}.onnx") if a.export_onnx else None
    )
    if a.export_onnx:
        ensure_dir(a.output_dir)

    onnx_b64 = miotfb64_to_onnxb64(
        mio_graph_b64=a.mio_b64,
        inputs=a.inputs,
        params=a.params,
        output_tensor_names=a.outputs,
        opset=a.opset,
        is_export_onnx=a.export_onnx,
        output_path=output_path,
    )
    dump_text(onnx_b64, a.output)
    return 0


def cmd_to_onnx(a: argparse.Namespace) -> int:
    from tensorflow.core.framework import graph_pb2

    graph_def = graph_pb2.GraphDef()
    with open(a.graph, "rb") as f:
        graph_def.ParseFromString(f.read())

    params, outputs, inputs = get_inputs_outputs_params_from_yaml(a.yaml)

    output_path = (
        os.path.join(a.output_dir, f"{a.onnx_name}.onnx") if a.export_onnx else None
    )
    if a.export_onnx:
        ensure_dir(a.output_dir)

    onnx_model = miotf_to_onnxb64(
        graph_def,
        inputs=inputs,
        params=params,
        output_tensor_names=outputs,
        opset=a.opset,
        is_export_onnx=a.export_onnx,
        output_path=output_path,
    )

    if not a.export_onnx:
        dump_text(
            "base64://" + base64.b64encode(onnx_model.SerializeToString()).decode("ascii"),
            a.output,
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="all2onnx")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("export-json", help="dynamic json -> onnx base64（可选导出 .onnx）")
    sp.add_argument("json", help="dynamic json 配置路径")
    sp.add_argument("--export-onnx", action="store_true", help="导出 .onnx 到 output_dir")
    sp.add_argument("--output-dir", default="./onnx_out", help="导出目录")
    sp.add_argument("--opset", type=int, default=None, help="onnx opset")
    sp.add_argument("--output-json", default=None, help="输出替换 graph 后的完整 json 到文件")
    sp.add_argument("--output-map-json", dest="output_map_json", default=None, help="额外输出 onnx_name->base64 映射到文件")
    sp.add_argument("--print", action="store_true", help="打印模型名和 base64 预览")
    sp.set_defaults(func=cmd_export_json)

    sp = sub.add_parser("convert-b64", help="mio base64 -> onnx base64（可选导出 .onnx）")
    sp.add_argument("--mio-b64", required=True, help="mio graph base64（base64://...）")
    sp.add_argument("--onnx-name", required=True, help="导出 onnx 名称（不含扩展名）")
    sp.add_argument("--inputs", nargs="*", default=[], help="输入名（不含 :0）")
    sp.add_argument("--params", nargs="*", default=[], help="参数名（不含 :0）")
    sp.add_argument("--outputs", nargs="*", default=[], help="输出名（通常带 :0）")
    sp.add_argument("--export-onnx", action="store_true", help="导出 .onnx 到 output_dir")
    sp.add_argument("--output-dir", default="./onnx_out", help="导出目录")
    sp.add_argument("--opset", type=int, default=None, help="onnx opset")
    sp.add_argument("--output", default=None, help="输出 base64 到文件（默认 stdout）")
    sp.set_defaults(func=cmd_convert_b64)

    sp = sub.add_parser("to-onnx", help="graph.pb + yaml -> onnx（可选导出 .onnx）")
    sp.add_argument("--graph", required=True, help="TensorFlow GraphDef pb 文件")
    sp.add_argument("--yaml", required=True, help="yaml 配置（同 tf_graph_tool.py）")
    sp.add_argument("--onnx-name", required=True, help="导出 onnx 名称（不含扩展名）")
    sp.add_argument("--export-onnx", action="store_true", help="导出 .onnx 到 output_dir")
    sp.add_argument("--output-dir", default="./onnx_out", help="导出目录")
    sp.add_argument("--opset", type=int, default=None, help="onnx opset")
    sp.add_argument("--output", default=None, help="不导出 onnx 时，base64 输出文件")
    sp.set_defaults(func=cmd_to_onnx)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    raise SystemExit(main())
