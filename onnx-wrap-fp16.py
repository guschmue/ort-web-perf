import argparse
import onnx

from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(description='make fp16 input/outputs fp32')
    parser.add_argument("--input", required=True, help='input')
    parser.add_argument("--output", help='output')
    args = parser.parse_args()
    return args


def create_io_mapping(graph, i_map, o_map):
    for n in graph.node:
        for i in n.input:
            i_map[i].append(n)
    for n in graph.node:
        for o in n.output:
            assert o not in o_map[o]
            o_map[o] = [n]


def wrap_inputs(graph, i_map):
    inputs = [n for n in graph.input if n.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16]
    for i in inputs:
        print(f"input {i.name} from fp32")
        for n in i_map[i.name]:
            for j, o in enumerate(n.input):
                if o == i.name:
                    n.input[j] = i.name + "_fp16"
        cast = onnx.helper.make_node(
            "Cast",
            inputs=[i.name],
            outputs=[i.name + "_fp16"],
            to=onnx.TensorProto.FLOAT16,
        )
        graph.node.insert(0, cast)
        i.type.tensor_type.elem_type = onnx.TensorProto.FLOAT


def wrap_outputs(graph, o_map):
    outputs = [n for n in graph.output if n.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16]
    for o in outputs:
        print(f"output {o.name} to fp32")
        for n in o_map[o.name]:
            for j, i in enumerate(n.output):
                if i == o.name:
                    n.output[j] = o.name + "_fp16"
        cast = onnx.helper.make_node(
            "Cast",
            inputs=[o.name + "_fp16"],
            outputs=[o.name],
            to=onnx.TensorProto.FLOAT,
        )
        graph.node.append(cast)
        o.type.tensor_type.elem_type = onnx.TensorProto.FLOAT


def main():
    args = get_args()

    model = onnx.load(args.input)
    i_map = defaultdict(list)
    o_map = defaultdict(list)

    create_io_mapping(model.graph, i_map, o_map)

    wrap_inputs(model.graph, i_map)
    wrap_outputs(model.graph, o_map)

    if args.output:
        onnx.save(model, args.output)


if __name__ == '__main__':
    main()
