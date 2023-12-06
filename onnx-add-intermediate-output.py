import argparse
import onnx


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument("--output", required=True, help='output')
    parser.add_argument("--input", required=True, help='input')
    parser.add_argument("--op", help='op-type to add to output')
    parser.add_argument("--name", help='node name to add to output')
    parser.add_argument("--purge", action='store_true', help="remove original outputs")
    parser.add_argument("--count", type=int, default=0, help='number of ops to take')
    args = parser.parse_args()
    args.op = args.op.split(",")
    args.name = args.name.split(",")
    return args


def add_output(model, output_name):
    output = model.graph.output.add()
    output.name = output_name


def main():
    args = get_args()

    model = onnx.load(args.input)
    taken = 0
    if args.purge:
        model.graph.ClearField("output")

    for i in model.graph.node:
        if i.op_type in args.op:
            add_output(model, i.output[0])
            taken += 1
            if args.count > 0 and taken >= args.count:
                break
        if i.name in args.name:
            add_output(model, i.output[0])
            taken += 1
            if args.count > 0 and taken >= args.count:
                break

    if args.output:
        onnx.save(model, args.output)
        print(f"output in {args.output}")


if __name__ == '__main__':
    main()
