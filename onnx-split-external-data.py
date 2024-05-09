import argparse

import onnx
import os


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime tool')
    parser.add_argument("--input", required=True, help='input')
    parser.add_argument("--output", help='output')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    max_size = 10 * 1024 * 1024
    model = onnx.load(args.input)

    if args.output:
        location = os.path.basename(args.output) + ".data"

        onnx.save_model(model, args.output, save_as_external_data=True, all_tensors_to_one_file=True, location=location, size_threshold=max_size, convert_attribute=False)


if __name__ == '__main__':
    main()
