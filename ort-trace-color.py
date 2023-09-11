import argparse
import json
import logging
import onnx
import re
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument('--input', help='input')
    parser.add_argument('--output', help='output')
    parser.add_argument('--webgpu', help='webgpu kernel timestamps')
    args = parser.parse_args()
    return args


def load_json(profile_path, webgpu_timestamps):
    entries = {}
    with open(profile_path, "r") as f:
        data = json.load(f)

    if type(data) == dict:
        data = data['traceEvents']

    for item in data:
        dur = item.get("dur")
        if dur is None:
            continue
        cat = item.get("cat")
        if cat not in ["Node", "Op"]:
            continue
        arg = item.get('args')
        if not arg:
            continue
        provider = arg.get("provider")
        provider = str(provider).replace("ExecutionProvider", "")
        op = arg.get("op_name")
        if op:
            name = item['name']
            if not name.endswith("_kernel_time"):
                continue
            dur = item['dur']
            name = name.replace("_kernel_time", "")
            # parameter_size = float(arg.get('parameter_size'))
            # activation_size = float(arg.get('activation_size'))
            # output_size = float(arg.get('output_size'))
            input_type_shape = arg.get('input_type_shape')
            input_dtype = str(list(input_type_shape[0].keys())[0])
            input_type_shape = str([list(i.values())[0] for i in input_type_shape])[1:-1]
            
            if provider == "Js" and webgpu_timestamps:
                w = webgpu_timestamps.get(name)
                if w:
                    dur = w['time']

            if op in ["MemcpyFromHost", "MemcpyToHost"]:
                provider = "CPY"

            e = {
                "dur": dur, "provider": provider, "dtype": input_dtype
            }
            entries[name] = e
    return entries


def main():
    args = get_args()
    webgpu_timestamps = None
    if args.webgpu:
        with open(args.webgpu, "r") as f:
            webgpu_timestamps = json.load(f)
            webgpu_timestamps = {i["name"]: i for i in webgpu_timestamps}

    try:
        entries = load_json(args.input, webgpu_timestamps)
        with open(args.output, "w") as f:
            json.dump(entries, f, indent=2)
    except Exception as ex:
        print(f"{args.input}: {ex}")
        sys.exit(1)


if __name__ == '__main__':
    main()
