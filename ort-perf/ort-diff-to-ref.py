import argparse
import os
import shutil
import sys
import time
import json

import numpy as np
import onnxruntime as rt
import csv
from ortdata import gen_data
from onnx import helper, numpy_helper, TensorProto, ModelProto, onnx_pb


class Status:

    @staticmethod
    def csv_header(writer):
        writer.writerow(
            [
                "model",
                "setup",
                "shape",
                "dtype",
                "correct",
                "first_run",
                "latency",
                "profile",
                "error",
            ]
        )

    def __init__(self, name):
        self.name = name
        self.setup = False
        self.shape = False
        self.dtype = False
        self.val = False
        self.err = "ok"
        self.first_run = 0
        self.latency = 0
        self.profile = False

    def info(self):
        self.err = self.err.replace("\n", " ")
        return f"{self.name} {self.setup} {self.shape} {self.dtype} {self.val} {self.first_run:.2f} {self.latency:.2f} {self.profile} {self.err}"

    def to_csv(self, writer):
        self.err = self.err.replace("\n", " ")
        writer.writerow(
            [
                self.name,
                self.setup,
                self.shape,
                self.dtype,
                self.val,
                self.first_run,
                self.latency,
                self.profile,
                self.err,
            ]
        )


def get_args():
    parser = argparse.ArgumentParser(description="onnxruntime bench tool")
    parser.add_argument("--models", help="run for specifc models")
    parser.add_argument("--cat", default="default", help="model category to run")
    parser.add_argument("--root", default="models", help="path to models")
    parser.add_argument("--rtol", type=float, default="0.005", help="default rtol")
    parser.add_argument("--atol", type=float, default="0.005", help="default atol")
    parser.add_argument("--config", default="ort-models.json", help="test config")
    parser.add_argument("--csv", help="csv")
    parser.add_argument(
        "-e",
        type=str,
        default="webgpu",
        choices=["cpu", "cuda", "dml", "webgpu"],
        help="EP",
    )
    parser.add_argument("--profile", help="enable profiling")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("--test-data", help="generate test data for onnx_test_runner")
    parser.add_argument("--perf", type=int, default=0, help="take some perf numbers")
    arg = parser.parse_args()
    return arg


def save_protobuf(path, message):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "wb") as f:
        f.write(message.SerializeToString())


def save_protos(inputs, outputs, output_names, name):
    dir = os.path.join(name, "test_data_set_0")
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

    for i, data_key in enumerate(inputs):
        data = inputs[data_key]
        t = numpy_helper.from_array(data)
        t.name = data_key
        data_full_path = os.path.join(
            name, "test_data_set_0", "input_" + str(i) + ".pb"
        )
        save_protobuf(data_full_path, t)

    for i, (data, data_key) in enumerate(zip(outputs, output_names)):
        t = numpy_helper.from_array(data)
        t.name = data_key
        data_full_path = os.path.join(
            name, "test_data_set_0", "output_" + str(i) + ".pb"
        )
        save_protobuf(data_full_path, t)


def make_session(model_path, ep, profile, verbose):
    opt = rt.SessionOptions()
    if profile:
        opt.enable_profiling = True
    if verbose:
        opt.log_verbosity_level = 3
        opt.log_severity_level = 1

    providers = ["CPUExecutionProvider"]
    if ep == "cuda":
        providers = ["CUDAExecutionProvider"]
    elif ep == "opengl":
        providers = ["OpenglExecutionProvider"]
    elif ep == "dml":
        providers = ["DmlExecutionProvider"]
        opt.enable_mem_pattern = False
    elif ep == "webgpu":
        providers = ["WebGpuExecutionProvider"]

    sess = rt.InferenceSession(model_path, sess_options=opt, providers=providers)
    return sess


def run_one_model(model, args):
    name = model["name"]
    status = Status(name)

    model_path = os.path.join(args.root, model["path"])
    if not os.path.exists(model_path):
        status.err = f"not found in {model_path}"
        return status

    if model.get("skip"):
        status.err = "skipped"
        return status

    print(f"# running: {name}, {model['path']}")

    try:
        # load model to cpu and get the reference output
        cpu_sess = make_session(model_path, "cpu", False, False)
        feed = gen_data(cpu_sess, model["gen"], False)
        output_names = [o.name for o in cpu_sess.get_outputs()]

        ref_outputs = cpu_sess.run(output_names, feed)

        if args.test_data:
            # save model and test data to a new directory
            dst_dir = os.path.join(args.test_data, name)
            save_protos(feed, ref_outputs, output_names, dst_dir)
            # copy the model
            shutil.copy(model_path, dst_dir)
            if os.path.exists(model_path + ".data"):
                shutil.copy(model_path + ".data", dst_dir)
            if os.path.exists(model_path + "_data"):
                shutil.copy(model_path + "_data", dst_dir)

        del cpu_sess
    except Exception as e:
        status.err = str(e)
        return status

    status.setup = True

    if args.verbose:
        print("input:")
        for k, v in feed.items():
            print(f"  {k}: {v.shape} {v.dtype}")

    rtol = model.get("rtol", args.rtol)
    atol = model.get("atol", args.atol)
    nocheck = model.get("nocheck", False)
    status.val = False
    status.dtype = False
    status.shape = False
    try:
        # load model to the target EP and get the output
        ep_sess = make_session(model_path, args.e, False, args.verbose)
        start = time.time()
        outputs = ep_sess.run(output_names, feed)
        status.first_run = time.time() - start
        if len(outputs) == len(ref_outputs):
            status.val = True
            status.dtype = True
            status.shape = True
            for i, (vr, v) in enumerate(zip(ref_outputs, outputs)):
                if vr.dtype != v.dtype:
                    status.dtype = False
                if vr.shape != v.shape:
                    status.shape = False
                if "argmax" in model and output_names[i] in model["argmax"]:
                    # llm - only check argmax
                    v = np.argmax(v)
                    vr = np.argmax(vr)
                results_ok = nocheck or np.allclose(vr, v, rtol, atol)
                if not results_ok:
                    status.val = False
                if args.debug and not results_ok:
                    # print the difference
                    np.testing.assert_allclose(vr, v, rtol, atol)
                if not results_ok:
                    status.err = f"{status.err},{output_names[i]}" if status.err != "ok" else f"missmatch: {output_names[i]}"

        # take some perf numbers
        if args.perf > 0:
            start = time.time()
            for i in range(args.perf):
                _ = ep_sess.run(output_names, feed)
            status.latency = (time.time() - start) / args.perf

        del ep_sess
    except Exception as e:
        status.err = str(e)

    if args.profile and status.err == "ok":
        # take some profiler numbers
        try:
            os.makedirs(args.profile, exist_ok=True)
            profile_file = os.path.join(args.profile, name + ".json")
            ep_sess = make_session(model_path, args.e, True, args.verbose)
            for i in range(10):
                _ = ep_sess.run(output_names, feed)
            trace_file = ep_sess.end_profiling()
            shutil.move(trace_file, profile_file)
            status.profile = True
        except Exception as e:
            status.err = str(e)

    return status


def main(args):
    config = json.load(open(args.config, encoding="utf-8"))
    models = []
    if args.models:
        names = args.models.split(",")
        models = [c for c in config if c["name"] in names]
    else:
        models = [c for c in config if c.get("category") == args.cat]

    start = time.time()
    # makes sure we got the models installed
    for model in models:
        path = os.path.join(args.root, model["path"])
        if not os.path.exists(path):
            print(f"model {model['name']} not found in {path}")

    if args.csv:
        f = open(args.csv, "w", encoding="utf-8")
        writer = csv.writer(f)
        Status.csv_header(writer)

    cnt = 0
    cnt_setup = 0
    cnt_pass = 0
    for model in models:
        status = run_one_model(model, args)
        print(status.info())
        if args.csv:
            status.to_csv(writer)
        cnt += 1
        if status.setup:
            cnt_setup += 1
        if status.val:
            cnt_pass += 1
    print(
        f"total time: {time.time() - start:.2f} sec, total: {cnt}, setup: {cnt_setup}, pass: {cnt_pass}"
    )
    if args.csv:
        f.close()

    return 0


if __name__ == "__main__":
    args = get_args()
    ret = main(args)
    sys.exit(0)
