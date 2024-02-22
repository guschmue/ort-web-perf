import argparse
import onnx
import re


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument("--output", required=True, help='output')
    parser.add_argument("--input", required=True, help='input')
    parser.add_argument("--op", default="", help='op-type to add to output')
    parser.add_argument("--name", default="", help='node name to add to output')
    parser.add_argument("--purge", action='store_true', help="remove original outputs")
    parser.add_argument("--all", action='store_true', help="add all nodes as output")
    parser.add_argument("--count", type=int, default=0, help='number of ops to take')
    parser.add_argument("--external_data", action='store_true', help='use external data')
    args = parser.parse_args()
    args.op = args.op.split(",")
    return args


def add_output(model, output_name, dtype):
    output = onnx.helper.make_tensor_value_info(output_name, dtype, None)
    model.graph.output.extend([output])
    print(f"add output {output_name}")


EXCLUDE = ['ConstantOfShape', 'Identity', "Constant", "Squeeze", "Unsqueeze", "Shape", "Cast", "Slice", "Reshape"]


def main():
    args = get_args()

    model = onnx.load(args.input)
    pat = None
    if args.name:
        pat = re.compile(args.name)

    taken = 0
    if args.purge:
        model.graph.ClearField("output")

    found = {i.name: 1 for i in model.graph.output}

    for i in model.graph.node:
        match = pat and pat.search(i.name)
        if i.op_type in args.op or match or args.all:
            if i.op_type in EXCLUDE:
                continue
            idx = 0
            if i.output[idx] not in found:
                add_output(model, i.output[idx], onnx.TensorProto.UNDEFINED)
                taken += 1
            if args.count > 0 and taken >= args.count:
                break

    print(f"{taken} taken")

    if args.output:
        model_with_shapes = onnx.shape_inference.infer_shapes(model)
        types = {}
        for i in model_with_shapes.graph.value_info:
            types[i.name] = i.type.tensor_type.elem_type

        for i in model.graph.output:
            if i.type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
                i.type.tensor_type.elem_type = types[i.name]
        onnx.save(model, args.output, save_as_external_data=args.external_data)
        print(f"output in {args.output}")


if __name__ == '__main__':
    main()
