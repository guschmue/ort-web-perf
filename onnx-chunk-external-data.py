##
## tool to chunk onnx external data files
## 
## for example: python onnx-chunk-external-data.py -i ../gemma-2-2b-web.0/onnx/model_q4f16.onnx -o onnx/model_q4f16.onnx --threshhold 1 --maxchunks 1
##
##

import argparse

import onnx
import onnx.external_data_helper as ext_data
import os
import itertools

MB = 1024 * 1024
DEFAULT_MAX_SIZE = 2000 # 2048 fails to fetch in chrome


def recursive_attribute_processor(attribute, func):
    if attribute.type == onnx.AttributeProto.GRAPH:
        yield from func(attribute.g)
    if attribute.type == onnx.AttributeProto.GRAPHS:
        for graph in attribute.graphs:
            yield from func(graph)


def get_attribute_tensors_from_graph(graph_or_function):
    for node in graph_or_function.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            yield from recursive_attribute_processor(attribute, get_attribute_tensors_from_graph)


def get_attribute_tensors(model_proto):
    yield from get_attribute_tensors_from_graph(model_proto.graph)
    for function in model_proto.functions:
        yield from get_attribute_tensors_from_graph(function)


def get_initializer_tensors_from_graph(graph_or_function):
    if isinstance(graph_or_function, onnx.GraphProto):
        yield from graph_or_function.initializer
    for node in graph_or_function.node:
        for attribute in node.attribute:
            yield from recursive_attribute_processor(attribute, get_initializer_tensors_from_graph)


def get_initializer_tensors(model_proto):
    yield from get_initializer_tensors_from_graph(model_proto.graph)
    for function in model_proto.functions:
        yield from get_attribute_tensors_from_graph(function)


def get_all_tensors(model_proto):
    return itertools.chain(get_initializer_tensors(model_proto), get_attribute_tensors(model_proto))


def save_external(model_proto, external_data_name, max_size, threshhold, maxchunks):
    idx = 0
    file_name = os.path.basename(external_data_name)

    def open_segment():
        if idx >= maxchunks:
            # no more datafiles, stick initializers into onnx file
            return None, "model"
        if idx == 0:
            name = f"{external_data_name}_data"
        else:
            name = f"{external_data_name}_data_{idx}"
        return open(name, "wb"), os.path.basename(name)

    offset = 0
    f, name = open_segment()

    for tensor in get_all_tensors(model_proto):
        tensor_data = tensor.raw_data
        tensor_size = len(tensor_data)
        assert tensor_data and tensor_size > 0
        if tensor_size < threshhold:
            # small tensors are stored in the model file
            continue
        if not f:
            # no more datafiles, stick initializers into onnx file
            continue
        if offset + tensor_size > max_size:
            # start new chunk
            f.close()
            print(f"{name}, size: {offset // MB} MB)")
            idx += 1
            offset = 0
            f, name = open_segment()
            if not f:
                continue
        f.write(tensor_data)
        tensor.ClearField("raw_data")
        del tensor.external_data[:]
        tensor.data_location = onnx.TensorProto.EXTERNAL
        for k, v in {
            "location": name,
            "offset": offset,
            "length": tensor_size,
        }.items():
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)
        offset += tensor_size
    if f:
        f.close()
        print(f"{name}, size: {offset // MB} MB)")
    onnx.save_model(model_proto, external_data_name)


def get_args():
    parser = argparse.ArgumentParser(description='tool to chunk onnx external data files')
    parser.add_argument("--input", "-i", required=True, help='input')
    parser.add_argument("--output", "-o", help='output')
    parser.add_argument("--size", default=DEFAULT_MAX_SIZE, type=int, help='max weight size')
    parser.add_argument("--threshhold", default=0, type=int, help='threshhold in MB to be external data')
    parser.add_argument("--maxchunks", default=99, type=int, help='maximum number of datafiles')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = onnx.load_model(args.input)
    if args.output:
        save_external(model, args.output, args.size * MB, args.threshhold * MB, args.maxchunks)


if __name__ == '__main__':
    main()
