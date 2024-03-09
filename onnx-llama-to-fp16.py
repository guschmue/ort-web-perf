
#
# convert an onnx model to fp16
#

import onnx
from onnxruntime.transformers.onnx_model import OnnxModel
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='fp32 input')
    parser.add_argument('--output', required=True, help='fp16 output')
    parser.add_argument('--keep-io', action='store_true', help='keep io types')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = OnnxModel(onnx.load_model(args.input, load_external_data=True))
    model.convert_float_to_float16(keep_io_types=False)
    model.save_model_to_file(args.output, use_external_data_format=True)


if __name__ == '__main__':
    main()
