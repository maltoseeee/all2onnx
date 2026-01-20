# all2onnx

## 环境安装（Conda + pip）

```bash
conda create -n your_env_name python=3.7
conda activate your_env_name

pip3 install tensorflow==1.15
pip3 install tf2onnx
pip3 install pyyaml
```

> 说明：
> - 本仓库部分流程依赖 TensorFlow；`to-onnx` 子命令会读取 TF 的 `GraphDef` 并转换。
> - 如需导出 onnx 文件（含 external data），还需安装 `onnx`（通常会被 `tf2onnx` 间接安装；如缺失可手动 `pip3 install onnx`）。

## 命令行（CLI）

本仓库提供命令行入口脚本：`all2onnx_cli.py`

### 1) 从 dynamic json 批量导出 onnx

- 导出到 `--output-dir` 下，文件名为 `{onnx_name}.onnx`
- 同时会生成 external data（如 `{onnx_name}.bin`）并放在同一目录

```bash
python3 all2onnx_cli.py export-json ./test/dynamic_json_config.json --export-onnx --output-dir ./output_dir
```

如需输出“替换 graph 字段后的完整 json”（把原本 mio 的 base64 graph 替换为 onnx base64）：

```bash
python3 all2onnx_cli.py export-json ./test/dynamic_json_config.json --output-json ./output_dir/new_config.json
```

如仍需要额外输出 `onnx_name -> onnx_base64` 映射：

```bash
python3 all2onnx_cli.py export-json ./test/dynamic_json_config.json --output-map-json ./output_dir/onnx_b64_map.json
```

### 2) graph.pb + yaml 转 onnx

```bash
python3 all2onnx_cli.py to-onnx \
  --graph ./your_graph.pb \
  --yaml ./your_config.yaml \
  --onnx-name your_model \
  --export-onnx --output-dir ./output_dir
```

### 3) 单个 mio base64 转 onnx base64（可选导出 onnx）

```bash
python3 all2onnx_cli.py convert-b64 \
  --mio-b64 "base64://..." \
  --onnx-name demo_model \
  --inputs inputA inputB \
  --params param1 param2 \
  --outputs out:0 \
  --export-onnx --output-dir ./output_dir
```

## TODO

- [ ] 将 `onnx_tensorrt_help/tensorflow_custom_ops/*.so` 对应的 **tf custom op 编译源码**与构建脚本引入本仓库，并补充不同 TF 版本（1.15/2.4/2.15）的构建说明与产物路径。

