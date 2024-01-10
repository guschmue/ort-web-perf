import argparse
import onnx


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument("--output", required=True, help='output')
    parser.add_argument("--input", required=True, help='input')
    parser.add_argument("--op", default="", help='op-type to add to output')
    parser.add_argument("--name", default="", help='node name to add to output')
    parser.add_argument("--purge", action='store_true', help="remove original outputs")
    parser.add_argument("--all", action='store_true', help="add all nodes as output")
    # parser.add_argument("--unused_nodes", action='store_true', help="remove unused nodes")
    # parser.add_argument("--unused_const", action='store_true', help="remove unused initializers")
    parser.add_argument("--count", type=int, default=0, help='number of ops to take')
    args = parser.parse_args()
    args.op = args.op.split(",")
    args.name = args.name.split(",")
    return args


def add_output(model, output_name):
    print(f"adding {output_name}")
    output = model.graph.output.add()
    output.name = output_name
    print(f"add output {output_name}")


def main():
    args = get_args()

    model = onnx.load(args.input)
    taken = 0
    if args.purge:
        model.graph.ClearField("output")

    for i in model.graph.node:
        if i.op_type in args.op or i.name in args.name or args.all:
            add_output(model, i.output[0])
            taken += 1
            if args.count > 0 and taken >= args.count:
                break

    if args.output:
        model_with_shapes = onnx.shape_inference.infer_shapes(model)
        model.graph.ClearField("output")
        for i in model_with_shapes.graph.output:
            model.graph.output.add().CopyFrom(i)
        onnx.save(model, args.output)
        print(f"output in {args.output}")


if __name__ == '__main__':
    main()
