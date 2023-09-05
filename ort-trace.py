import argparse
import json
import logging
import re
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument('strings', metavar='N', type=str, nargs='+', help='strings')
    parser.add_argument('--name', help='filter list')
    parser.add_argument('--csv', help='save intermidiate data to csv')
    parser.add_argument('--webgpu', help='webgpu kernel timestamps')
    parser.add_argument('--cov', help='tag coverage list')
    parser.add_argument('--exclude', help='ops to exclude, ie. If')
    parser.add_argument('--exclude-provider', help='providers to exclude')
    parser.add_argument('-l', type=int, default=20, help='list top N items, default=20')
    parser.add_argument('-v', action='store_true', help='verbose')
    parser.add_argument('--nodes', action='store_true', help='show top N nodes')
    parser.add_argument('--shapes', action='store_true', help='group by shapes')
    parser.add_argument('--dtypes', action='store_true', help='group by dtypes')
    parser.add_argument('--type-in-name', action='store_true', help='add dtype to op_type')
    parser.add_argument('--provider', action='store_true', help='group by provider')
    parser.add_argument('--mem', action='store_true', help='sort by memory usage')
    args = parser.parse_args()
    if args.exclude:
        args.exclude = args.exclude.split(",")
    if args.exclude_provider:
        args.exclude_provider = "|".join(args.exclude_provider.split(","))
    return args


def clean_json(s):
    s = re.sub(",[ \t\r\n]*}", "}", s)
    s = re.sub(",[ \t\r\n]*\]", "]", s)
    return s


def json_to_df(profile_path, exclude, webgpu_timestamps, verbose):
    entries = []
    op_index = -1
    webgpu_acc = 0

    with open(profile_path, "r") as f:
        # data = json.load(f)
        data = json.loads(clean_json(f.read()))

    if type(data) == dict:
        data = data['traceEvents']
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
            name = item['name']
            if not name.endswith("_kernel_time"):
                continue
            name = name.replace("_kernel_time", "")
            op_index += 1
            if exclude and op in exclude:
                continue
            dur = item['dur']
            parameter_size = float(arg.get('parameter_size'))
            activation_size = float(arg.get('activation_size'))
            output_size = float(arg.get('output_size'))
            input_type_shape = arg.get('input_type_shape')
            input_dtype = str(list(input_type_shape[0].keys())[0])
            input_shape = str([list(i.values())[0] for i in input_type_shape])[1:-1]
            # output_type_shape = arg.get('output_type_shape')

            if provider == "Js" and webgpu_timestamps:
                if op == "MemcpyToHost":
                    dur -= webgpu_acc
                    webgpu_acc = 0
                    pass
                else:
                    w = webgpu_timestamps.get(name)
                    if w:
                        # should this be dur += w['time'] ?
                        dur = w['time']
                        # next line will substract the kernel time from memcopytohost
                        # webgpu_acc += w['time']
                        # print(f"{name}, {op}, {provider} not a kernel")

            e = {
                "name": name, "dur": dur, "op_type": op, "provider": provider,
                "parameter_size": parameter_size, "activation_size": activation_size,
                "output_size": output_size, "shape": input_shape, "dtype": input_dtype,
            }
            entries.append(e)
    df = pd.DataFrame([f for f in entries])
    df['count'] = 1
    return df


def main():
    args = get_args()

    webgpu_timestamps = None
    if args.webgpu:
        with open(args.webgpu, "r") as f:
            webgpu_timestamps = json.load(f)
            webgpu_timestamps = {i["name"]: i for i in webgpu_timestamps}
    df_list = []
    for fname in args.strings:
        try:
            print(fname)
            df = json_to_df(fname, args.exclude, webgpu_timestamps, args.v)
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
    pd.set_option('display.max_colwidth', 120)
    df2 = df[['dur', 'count']].sum()
    df['pct'] = (100 * df['dur'] / df2['dur'])
    mem_field = "output_size"
    sort_by = mem_field if args.mem else "dur"
    extra_fields = [mem_field] if args.mem else []
    if args.provider:
        extra_fields.append("provider")
    if args.exclude_provider:
        df = df[~df['provider'].str.contains(args.exclude_provider)]
    if args.type_in_name:
        df['op_type'] = df['op_type'] + "." + df['dtype']
    if not args.nodes:
        fields = ["op_type", "dur", "pct", "count", "shape", "dtype"] + extra_fields
        groups = ['op_type']
        if args.dtypes:
            groups.append('dtype')
        if args.shapes:
            groups.append('shape')
        if args.provider:
            groups.append('provider')
        df1 = df[fields].groupby(groups).sum()
        if args.mem:
            df1[mem_field] = df1[mem_field] / df1['count'] / 1024 / 1024
        df1 = df1.sort_values(by=sort_by, ascending=False)[:top]
        df1['csum'] = df1['pct'].cumsum()
        df1['avg'] = df1['dur'] / df1['count']
        print("\n--Top ops by total runtime")
        print(df1.round(digits).to_string(index=True))
    else:
        fields = ["name", "op_type", "dur", "pct", "count"] + extra_fields
        df1 = df[fields].groupby(['name', "op_type"]).sum()
        if args.mem:
            df1[mem_field] = df1[mem_field] / df1['count'] / 1024 / 1024
        df1 = df1.sort_values(by=sort_by, ascending=False)[:top]
        df1['csum'] = df1['pct'].cumsum()
        df1['avg'] = df1['dur'] / df1['count']
        print("\n--Top nodes by total runtime")
        print(df1.round(digits).to_string(index=True))

    if args.csv:
        if args.shapes:
            df1 = df1.reset_index().set_index('op_type')
        if True or args.cov:
            # impl = {}
            # with open(args.cov, "r") as f:
            #    for i in f:
            #        impl[i.strip()] = 1
            df1['aggname'] = df1.index.map(lambda x: x[0] + "." + x[1])
            # df1['aggimpl'] = df1["aggname"].apply(lambda x: impl.get(x, 0))
            df1['aggsrc'] = os.path.basename(args.strings[0])
            df1['aggcnt'] = 1
        df1.to_csv(args.csv)


if __name__ == '__main__':
    main()
