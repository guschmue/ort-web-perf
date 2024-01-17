import argparse

import onnx


def get_args():
    parser = argparse.ArgumentParser(description='make const op in onnx an initializer')
    parser.add_argument("--input", required=True, help='input')
    parser.add_argument("--output", help='output')
    args = parser.parse_args()
    return args


def add_output(model, output_name):
    output = model.graph.output.add()
    output.name = output_name
    tensor_type_proto = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [1, 64, 56, 56])
    output.type.CopyFrom(tensor_type_proto)


def remove_const(model):
    graph = model.graph
    to_remove = []
    for n in graph.node:
        if n.op_type == 'Constant':
            print(f"removeing {n.name}")
            to_remove.append(n)
            t = n.attribute[0].t
            t.name = n.output[0]
            graph.initializer.append(t)
    for n in to_remove:
        graph.node.remove(n)
    print(f"removed {len(to_remove)} nodes, total initializers {len(graph.initializer)}")


def main():
    args = get_args()

    model = onnx.load(args.input)
    remove_const(model)

    if args.output:
        onnx.save(model, args.output)


if __name__ == '__main__':
    main()
