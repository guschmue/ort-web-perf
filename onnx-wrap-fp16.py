import argparse
import onnx
import re

from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument("--input", required=True, help='input')
    parser.add_argument("--output", help='output')
    parser.add_argument("--name", help='name')
    parser.add_argument("--external_data", action='store_true', help='use external data')
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


def wrap_inputs(graph, i_map, names):
    # 1. find fp16 inputs
    # 2. rewrite all consumers
    # 3. insert cast
    # 4. rewrite graph inputs
    inputs = [n for n in graph.input if n.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16]
    for i in inputs:
        if names:
            match = names.search(i.name)
            if not match:
                continue
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


def wrap_outputs(graph, i_map, o_map, names):
    # 1. find fp16 outputs
    # 2. rewrite all providers
    # 3. append cast
    # 4. rewrite graph outputs
    outputs = [n for n in graph.output if n.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16]
    for o in outputs:
        if names:
            match = names.search(o.name)
            if not match:
                continue
        print(f"output {o.name} to fp32")
        for n in o_map[o.name]:
            for j, i in enumerate(n.output):
                if i == o.name:
                    n.output[j] = o.name + "_fp16"
        for n in i_map[o.name]:
            for j, i in enumerate(n.input):
                if i == o.name:
                    n.input[j] = o.name + "_fp16"

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

    pat = None
    if args.name:
        pat = re.compile(args.name)

    wrap_inputs(model.graph, i_map, pat)
    wrap_outputs(model.graph, i_map, o_map, pat)

    if args.output:
        onnx.save(model, args.output, save_as_external_data=args.external_data)


if __name__ == '__main__':
    main()
