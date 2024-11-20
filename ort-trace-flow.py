import argparse
import json
import logging
import onnx
import re
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument('strings', metavar='N', type=str, nargs='+', help='strings')
    parser.add_argument('--name', help='filter list')
    parser.add_argument('--model', help='onnx model')
    parser.add_argument('--csv', help='save intermidiate data to csv')
    parser.add_argument('--exclude', help='ops to exclude, ie. If')
    parser.add_argument('-l', type=int, default=20, help='list top N items, default=20')
    parser.add_argument('-v', action='store_true', help='verbose')
    parser.add_argument('--nodes', action='store_true', help='show top N nodes')
    parser.add_argument('--shapes', action='store_true', help='group by shapes')
    parser.add_argument('--provider', action='store_true', help='group by provider')
    parser.add_argument('--mem', action='store_true', help='sort by memory usage')
    args = parser.parse_args()

    if args.exclude:
        args.exclude = args.exclude.split(",")
    return args


def clean_json(s):
    s = re.sub(",[ \t\r\n]*}", "}", s)
    s = re.sub(",[ \t\r\n]*\]", "]", s)
    return s


def json_to_df(profile_path, exclude, verbose, node_order):
    entries = []
    with open(profile_path, "r") as f:
        # data = json.load(f)
        data = json.loads(clean_json(f.read()))

    if type(data) == dict:
        data = data['traceEvents']

    last_order_id = 0
    for item in data:
        dur = item.get("dur")
        if dur is None:
            continue
        cat = item.get("cat")
        if cat not in ["Node", "Op"]:
            continue
        arg = item.get('args')
        if not arg:
            continue
        provider = arg.get("provider")
        provider = str(provider).replace("ExecutionProvider", "")
        op = arg.get("op_name")
        if op:
            if exclude and op in exclude:
                continue
            name = item['name']
            if not name.endswith("_kernel_time"):
                continue
            dur = item['dur']
            name = name.replace("_kernel_time", "")
            order_id = -1
            if node_order:
                order_id = node_order.get(name)
                if order_id is None:
                    print(f"WARNING: node_order not found for {name}")
                    order_id = last_order_id
                last_order_id = order_id
            if op in ["MemcpyFromHost"]:
                provider = "CPU"

            # graph_index = arg.get('graph_index')
            parameter_size = float(arg.get('parameter_size'))
            activation_size = float(arg.get('activation_size'))
            output_size = float(arg.get('output_size'))
            input_type_shape = arg.get('input_type_shape')
            input_dtype = str(list(input_type_shape[0].keys())[0])
            input_type_shape = str([list(i.values())[0] for i in input_type_shape])[1:-1]
            # output_type_shape = arg.get('output_type_shape')
            op = op + "." + input_dtype

            e = {
                "name": name, "dur": dur, "op_type_org": op, "provider": provider,
                "parameter_size": parameter_size, "activation_size": activation_size,
                "output_size": output_size, "shape": input_type_shape,
                "dtype": input_dtype, "op_type": f"{op}",
                "order_id": order_id,
            }
            entries.append(e)

    # entries = sorted(entries, key=lambda x: x['order_id'])

    flow = 0
    cprovider = "CPU"
    flow_map = {}
    for e in entries:
        provider = e["provider"]
        if provider != cprovider:
            flow += 1
            cprovider = provider
        flow_item = flow_map.get(flow)
        if not flow_item:
            flow_item = ([], [], [])
        flow_item[0].append(e["name"])
        flow_item[1].append(e["op_type"])
        flow_item[2].append(provider)
        flow_map[flow] = flow_item
        e["flow"] = flow

    # dedupe flows
    flow_map_reverse = {}
    idx = 0
    flow_map_ids = {}
    for k, v in flow_map.items():
        names = ",".join(v[0])
        ops = ",".join(v[1])
        flow_item = flow_map_reverse.get(names)
        if flow_item:
            assert ops == flow_item[1]
            flow_map_ids[k] = flow_item[1]
            continue
        idx += 1
        flow_map_reverse[ops] = (idx, ops, names, v[2])
        flow_map_ids[k] = ops

    for e in entries:
        e['flow'] = flow_map_ids.get(e['flow'])
    df = pd.DataFrame([f for f in entries])
    df['count'] = 1
    return df, flow_map_reverse


def load_model(model_path):
    node2input = {}
    node2output = {}
    model = onnx.load(model_path)
    graph_queue = [model.graph]
    node_order = {}
    idx = 0
    while len(graph_queue):
        graph = graph_queue.pop(0)
        for node in graph.node:
            node_order[node.name] = idx
            idx += 1
            for attr in node.attribute:
                if attr.graphs:
                    graph_queue.extend(attr.graphs)
                if attr.HasField("g"):
                    graph_queue.append(attr.g)
            for i in node.input:
                node2input[i] = node.name
            for i in node.output:
                node2output[i] = node.name

    return node_order, node2input, node2output


def main():
    args = get_args()

    node_order = None
    node2input = {}
    node2output = {}
    if args.model:
        node_order, node2input, node2output = load_model(args.model)

    df_list = []
    for fname in args.strings:
        try:
            df, flow_map_reverse = json_to_df(fname, args.exclude, args.v, node_order)
            if args.v:
                print(fname, len(df))
            df_list.append(df)
        except Exception as ex:
            print(f"{fname}: {ex}")
            sys.exit(1)

    df = pd.concat(df_list)
    df_list = None
    digits = 1
    top = args.l
    pd.set_option('display.max_colwidth', 180)
    df2 = df[['dur', 'count']].sum()
    df['pct'] = (100 * df['dur'] / df2['dur'])
    if args.nodes:
        fields = ["op_type", "shape", "provider", "dur", "pct", "count", "name"]
        df1 = df[fields]
    else:
        sort_by = "dur"
        fields = ["flow", "op_type", "dur", "pct", "count"]
        groups = ['flow']
        df1 = df[fields].groupby(groups).sum()
        df1 = df1.sort_values(by=sort_by, ascending=False)[:top]
        df1['provider'] = df1.index.to_series().apply(lambda x: flow_map_reverse[x][3][0])
        df1['csum'] = df1['pct'].cumsum()
        df1['avg'] = df1['dur'] / df1['count']
        print(f"\n--Top flows by total runtime, {len(flow_map_reverse)} flows")

    print(df1.round(digits).to_string(index=True))

    if args.csv:
        if args.shapes:
            df1 = df1.reset_index().set_index('op_type')
        df1.to_csv(args.csv)


if __name__ == '__main__':
    main()
