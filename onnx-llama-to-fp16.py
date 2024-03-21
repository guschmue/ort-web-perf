
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
    parser.add_argument('--external-data', action='store_true', help='keep io types')
    parser.add_argument('--block', required=False, help='op block list')
    parser.add_argument('--block-nodes', required=False, help='node block list')
    
    args = parser.parse_args()
    if (args.block):
        args.block = args.block.split(',')
    if (args.block_nodes):
        args.block_nodes = args.block_nodes.split(',')
    return args


def main():
    args = get_args()

    model = OnnxModel(onnx.load_model(args.input, load_external_data=True))
    model.convert_float_to_float16(keep_io_types=args.keep_io, op_block_list=args.block, node_block_list=args.block_nodes)
    model.save_model_to_file(args.output, use_external_data_format=args.external_data)


if __name__ == '__main__':
    main()
