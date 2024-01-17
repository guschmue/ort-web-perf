import argparse
from collections import defaultdict
import onnx


def get_args():
    parser = argparse.ArgumentParser(description='remove float64 from onnx model')
    parser.add_argument("--output", required=False, help='output')
    parser.add_argument("--input", required=True, help='input')
    args = parser.parse_args()
    return args


def remove_node(n, i_map, o_map):
    # replace children inputs
    name = n.output[0]
    for p in i_map[name]:
        for i in range(len(p.input)):
            if p.input[i] == name:
                p.input[i] = n.input[0]


def create_io_mapping(graph, i_map, o_map):
    for n in graph.node:
        for i in n.input:
            i_map[i].append(n)
    for n in graph.node:
        for o in n.output:
            assert o not in o_map[o]
            o_map[o] = [n]


def remove_float64_cast_1(graph, i_map, o_map):
    new_type = onnx.TensorProto.FLOAT16
    changed = 0
    for n in graph.node:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.i == onnx.TensorProto.DOUBLE:
                    a.i = new_type
                    changed += 1
    for o in graph.output:
        if o.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
            o.type.tensor_type.elem_type = new_type

    for i in range(0, len(graph.initializer)):
        it = graph.initializer[i]
        if it.data_type == onnx.TensorProto.DOUBLE:
            val = onnx.numpy_helper.to_array(it)
            val = val.astype("float16")
            nit = onnx.numpy_helper.from_array(val, name=it.name)
            it.CopyFrom(nit)
            changed += 1

    print(f"changed: {changed}")


def main():
    args = get_args()

    model = onnx.load(args.input)

    i_map = defaultdict(list)
    o_map = defaultdict(list)
    create_io_mapping(model.graph, i_map, o_map)

    remove_float64_cast_1(model.graph, i_map, o_map)

    if args.output:
        onnx.save(model, args.output)
        print(f"output in {args.output}")


if __name__ == '__main__':
    main()
