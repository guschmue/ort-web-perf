#
# simple llm benchmark using onnxruntime.
# models use nick names, like llama3.2-1b, They are mapped we assume models in the MODELS dictionary few lines down
# and prefixed with $PWD/models/.
#
# prep:
#  pip install transformers onnxruntime
#
# run:
#  python ort-llm.py --model llama3.2-1b --provider webgpu --use-iob
#  python ort-llm.py --model llama3.2-1b --fp32 --provider webgpu --use-iob
#

import argparse
import json
import time
from transformers import AutoTokenizer, AutoConfig
import onnxruntime as ort
import numpy as np
import os
import sys
import shutil


def get_args():
    parser = argparse.ArgumentParser(description='tool')
    parser.add_argument('--model', required=True, help='model')
    parser.add_argument('--config', default="ort-llm.json", help='config')
    parser.add_argument('--N', "-N", type=int, default=1, help='measure N times')
    parser.add_argument('--max_tokens', type=int, default=9999, help='max_tokens')
    parser.add_argument('--max_prompt', type=int, default=10 * 1024, help='chunk prompt at max_prompt')
    parser.add_argument('--provider', default="webgpu", choices=["cpu", "cuda", "webgpu", "dml"], help='provider')
    parser.add_argument('--profile', help='profile')
    parser.add_argument('--metal-profile', help='take a metal profile, needs MTL_CAPTURE_ENABLED=1')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--fp32', action='store_true', help='use fp32, fp16 is the default')
    parser.add_argument('--use-iob', action='store_true', help='use io-bindings')
    parser.add_argument('--static', action='store_true', help='use static kv_cache')
    parser.add_argument('--pinemb', action='store_true', help='pin embeddings to cpu')
    parser.add_argument('--quiet', action='store_true', help='no output duing run')
    parser.add_argument('--stats', action='store_true', help='print output stats')
    parser.add_argument('--gc', action='store_true', help='graph capture')
    parser.add_argument('--system-prompt', action='store_true', help='use a system prompt')
    parser.add_argument('--csv', help='csv')
    parser.add_argument('--tag', default="main", help='tag')
    parser.add_argument('--platform', default="", help='platform')
    parser.add_argument('--task', default="prefill-500", help='task')
    parser.add_argument('--postfix', help='postfix for path')
    parser.add_argument('--values', help='dump values')

    parser.description = "Example: python ort-llm.py  --provider webgpu --model llama3.2-1b --task generation"
    args = parser.parse_args()
    args.use_iob = True
    return args


def stats(arr):
    arr = arr.astype(np.float32)
    mean = np.mean(arr)
    median = np.median(arr)
    minimum = np.min(arr)
    maximum = np.max(arr)
    variance = np.var(arr)
    stddev = np.std(arr)
    return {"min": minimum, "max": maximum, "mean": mean, "median": median, "stddev": stddev, "var": variance}


class LLM(object):
    def __init__(self):
        pass

    def setup_session(
        self,
        model_path,
        ep,
        verbose,
        profile=False,
        fp32=False,
        use_iob=False,
        stats=False,
        gc=False
    ):
        self.stats = stats
        self.use_iob = use_iob
        self.ep = ep
        self.profile = profile
        if stats:
            self.use_iob = False
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)

        self.config = config
        opt = ort.SessionOptions()

        providers = ["CPUExecutionProvider"]
        if ep == "cuda":
            providers = ["CUDAExecutionProvider"]
        elif ep == "dml":
            providers = ["DmlExecutionProvider"]
        elif ep == "webgpu":
            providers = ["WebGpuExecutionProvider"]

        if verbose:
            opt.log_verbosity_level = 3
            opt.log_severity_level = 1

        if profile:
            opt.enable_profiling = True

        opt.add_session_config_entry("ep.webgpuexecutionprovider.enableGraphCapture", "1")

        model_file = os.path.join(model_path, "model.onnx")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_path, "onnx", "model_q4f16.onnx")
            if not os.path.exists(model_file):
                raise RuntimeError(f"model file {model_file} not found")
        self.sess = ort.InferenceSession(
            model_file, sess_options=opt, providers=providers
        )
        self.device_id = (
            int(self.sess.get_provider_options()[providers[0]]["device_id"])
            if ep in ["cuda", "dml"]
            else 0
        )
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.need_position_id = "position_ids" in self.input_names
        self.logits_idx = self.output_names.index("logits")
        self.eos = self.config.eos_token_id
        if not isinstance(self.eos, list):
            self.eos = [self.eos]
        return self.sess

    def setup_feed(self):
        self.feed = {}
        try:
            dim_kv = self.config.head_dim
        except:
            dim_kv = int(self.config.hidden_size / self.config.num_attention_heads)
        kv_dims = [1, self.config.num_key_value_heads, 0, dim_kv]
        for i, input_meta in enumerate(self.sess.get_inputs()):
            if input_meta.name.startswith("past"):
                dtype = np.float32 if input_meta.type == "tensor(float)" else np.float16
                self.feed[input_meta.name] = np.array([], dtype=dtype).reshape(kv_dims)
        self.output_tokens = []
        if self.use_iob:
            self.iob = self.sess.io_binding()
            self.iob.bind_output("logits")
            for i, name in enumerate(self.output_names):
                if name.startswith("present"):
                    self.iob.bind_output(
                        name, device_type=self.ep, device_id=self.device_id
                    )
            for i, name in enumerate(self.input_names):
                if name.startswith("past"):
                    v = self.feed[name]
                    self.iob.bind_cpu_input(name, v)

    def update_kv_cache_iob(self):
        for i, name in enumerate(self.output_names):
            if name.startswith("present"):
                new_name = name.replace("present", "past_key_values")
                ort_output = self.iob.get_outputs()[i]
                self.iob.bind_ortvalue_input(new_name, ort_output)
                self.iob.bind_output(
                    name, device_type=self.ep, device_id=self.device_id
                )

    def update_kv_cache(self, outputs):
        for i, name in enumerate(self.output_names):
            if name.startswith("present"):
                new_name = name.replace("present", "past_key_values")
                self.feed[new_name] = outputs[i]

    def generate(self, tokens, **kw_args):
        keep_cache = kw_args.get("keep_cache", 0)
        max_tokens = kw_args.get("max_tokens", 256) + len(tokens)
        values = kw_args.get("values", None)
        cb = kw_args.get("cb", None)
        input_names = ["input_ids", "attention_mask"]

        if keep_cache:
            self.output_tokens.extend(tokens)
        else:
            self.setup_feed()
            self.output_tokens = tokens.tolist()

        feed = self.feed
        feed["input_ids"] = tokens.reshape(1, len(tokens))

        last_token = 0
        seqlen = len(self.output_tokens)
        if self.need_position_id:
            input_names.append("position_ids")
            if keep_cache:
                feed["position_ids"] = np.arange(
                    seqlen - len(tokens), seqlen, dtype=np.int64
                ).reshape([1, len(tokens)])
            else:
                feed["position_ids"] = np.arange(seqlen, dtype=np.int64).reshape(
                    [1, seqlen]
                )

        first_token_time = 0
        start_time = time.time()
        while last_token not in self.eos and seqlen < max_tokens:
            feed["attention_mask"] = np.empty([1, seqlen], dtype=np.int64)
            feed["attention_mask"].fill(1)
            if self.use_iob:
                for name in input_names:
                    self.iob.bind_cpu_input(name, feed[name])
                self.iob.bind_output("logits")
                self.sess.run_with_iobinding(self.iob)
                logits = self.iob.get_outputs()[0].numpy()
                self.update_kv_cache_iob()
            else:
                outputs = self.sess.run([], feed)
                logits = outputs[self.logits_idx]
                self.update_kv_cache(outputs)

            if not first_token_time:
                first_token_time = time.time()

            if values:
                j = {}
                for idx, name in enumerate(self.output_names):
                    if name.startswith("present."):
                        continue
                    t = outputs[idx]
                    v = t.flatten().tolist()[:10000000]
                    j[self.output_names[idx]] = {
                        "dtype": str(t.dtype),
                        "shape": list(t.shape),
                        "output-avg": float(t.mean()),
                        "output-val": list(v),
                    }
                with open(values, "w") as f:
                    f.write(json.dumps(j, indent=2))
                return 0

            if self.stats:
                for idx, name in enumerate(self.output_names):
                    if name.startswith("present.") or name.startswith("past."):
                        continue
                    if not (name.startswith("/model/layers.0/attn/GroupQueryAttention/") or name.startswith("/model/layers.0/input_layernorm")):
                        continue
                    t = outputs[idx]
                    print(f"{name}:: {stats(t)}")
                return

            last_token = np.argmax(logits[0, -1, :])
            self.output_tokens.append(last_token)
            feed["input_ids"] = np.array([last_token], dtype=np.int64).reshape(1, 1)
            if self.need_position_id:
                feed["position_ids"] = np.array([seqlen], dtype=np.int64).reshape(1, 1)
            seqlen += 1
            if cb:
                cb(self.output_tokens, seqlen)

        end_time = time.time()

        return (
            self.output_tokens,
            end_time - start_time,
            end_time - first_token_time,
            first_token_time - start_time,
            seqlen,
        )


def main():
    args = get_args()

    if args.metal_profile:
        import mlx.core as mx

    model_root = os.environ.get("ORT_LLM_MODELS", "models")

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = config["models"][args.model]
    model_path = os.path.join(model_root, model["path"])
    if not args.postfix and args.provider in {"cpu", "cuda"}:
        args.postfix = args.provider
    if args.postfix:
        model_path = model_path + "-" + args.postfix
    print(model_path)

    task = config["tasks"][args.task]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
    llm = LLM()
    llm.setup_session(model_path, args.provider, args.verbose, args.profile, fp32=args.fp32, use_iob=args.use_iob, stats=args.stats, gc=args.gc)
    llm.setup_feed()

    # warmup
    message = [{"role": "user", "content": "hello"}]
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=False, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="np", return_attention_mask=False)  # , return_tensor: false, padding: true, truncation: true })
    input_ids = input_ids['input_ids'][0].astype(np.int64)
    prompt_tokens = len(input_ids)
    _ = llm.generate(input_ids, keep_cache=False, max_tokens=1)

    # timed run
    message = [{"role": "user", "content": task}]
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=False, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="np", return_attention_mask=False)  # , return_tensor: false, padding: true, truncation: true })
    input_ids = input_ids['input_ids'][0].astype(np.int64)
    prompt_tokens = len(input_ids)

    def cb(output_tokens, seqlen):
        return print(tokenizer.decode(output_tokens[seqlen - 1:], skip_special_tokens=True), end='', flush=True)

    cb1 = None if args.quiet else cb
    if args.metal_profile:
        mx.metal.start_capture(args.metal_profile)
    output_tokens, took, gen_time, prompt_time, seqlen = llm.generate(input_ids, keep_cache=False, max_tokens=args.max_tokens, values=args.values, cb=cb1)
    if args.metal_profile:
        mx.metal.stop_capture()
    if args.profile:
        trace_file = llm.sess.end_profiling()
        shutil.move(trace_file, args.profile)

    txt = tokenizer.decode(output_tokens[prompt_tokens:], skip_special_tokens=True)
    if not args.quiet:
        print(txt)
    if gen_time == 0:
        gen_time = 1
    new_tokens_length = seqlen - prompt_tokens
    e2e_tps = new_tokens_length / took
    prompt_tps = prompt_tokens / prompt_time
    gen_tps = new_tokens_length / gen_time
    print(f'{new_tokens_length} tokens in {took:.1f}sec, e2e:{e2e_tps:.1f} tps, prompt: {prompt_tps:.1f} tps, gen: {gen_tps:.1f} tps, ttft: {prompt_time:.2f} sec')

    if args.csv:
        with open(args.csv, "a") as f:
            precision = "q4fp32" if args.fp32 else "q4fp16"
            f.write(f"{args.model},{took:.1f},{e2e_tps:.1f},{new_tokens_length},{prompt_tokens},{prompt_tps:.1f},{gen_tps:.1f},{prompt_time:.2f},{args.task},{precision},{args.tag},{args.platform},{args.provider},ort-llm.py,1\n")
    return 0


if __name__ == '__main__':
    main()
