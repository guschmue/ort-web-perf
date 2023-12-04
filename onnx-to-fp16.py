
import onnx
from onnxconverter_common import float16
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument('--input', required=True, help='fp32 input')
    parser.add_argument('--output', required=True, help='fp16 output')
    parser.add_argument('--keep-io', action='store_true', help='keep io types')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = onnx.load(args.input)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=args.keep_io)
    onnx.save_model(model_fp16, args.output, save_as_external_data=False, all_tensors_to_one_file=False)


if __name__ == '__main__':
    main()
