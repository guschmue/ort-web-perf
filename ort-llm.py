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


MODELS = {
    "llama3.2-1b": {"name": "llama3.2-1B", "path": "tjs/Llama-3.2-1B-Instruct"},
    "llama3.2-1b-gqa": {"name": "llama3.2-1B-gqa", "path": "tjs/Llama-3.2-1B-Instruct-gqa"},
    "llama3.2-3b": {"name": "llama3.2-3B", "path": "tjs/Llama-3.2-3B-Instruct"},
    "phi3.5": {"name": "phi3.5", "path": "microsoft/Phi-3.5-mini-instruct-onnx-web"},
    "phi3.5-gqa": {"name": "phi3.5-gqa", "path": "microsoft/Phi-3.5-mini-instruct-gqa"},
    "phi3.5-128": {"name": "phi3.5-128", "path": "microsoft/Phi-3.5-mini-instruct-128"},
    "phi3": {"name": "phi3", "path": "microsoft/Phi-3-mini-4k-instruct-onnx-web"},
    "phi3-dml": {"name": "phi3", "path": "microsoft/Phi-3-mini-4k-instruct-onnx-directml"},
    "gemma-2b": {"name": "gemma-2b", "path": "Xenova/gemma-2-2b-it-onnx-web"},
    "smollm": {"name": "smollm", "path": "Xenova/smollm-360M-instruct-add-basics"},
    "smollmv2": {"name": "smollmv2", "path": "tjs/SmolLM2-135M-Instruct-onnx-web"},
    "tinyllama": {"name": "tinyllama", "path": "schmuell/TinyLlama-1.1B-Chat-v1.0"},
    "qwen2.5-0.5b": {"name": "qwen2.5-0.5b", "path": "tjs/Qwen2.5-0.5B-Instruct-onnx-web"},
    "foo": {"name": "foo", "path": "tjs/foo"},
}

SUM = """
Summarize:
Constantinople, now known as Istanbul in modern Turkey, was a historically significant city that served as the capital of both the Roman/Byzantine 
Empire and the Ottoman Empire. Its rich history spans over 2,500 years, with its strategic location at the crossroads between Europe and Asia contributing 
to its prominence throughout various periods. The city was originally founded by Greek colonists from Megara as Byzantium around 657 BC. It became a 
significant center of trade due to its position on the Bosphorus, controlling passage between the Black Sea and the Mediterranean. However, it gained 
even greater importance after Emperor Constantine I (Constantinus) relocated his capital there in 324 AD, thus renaming it Constantinople.
The Byzantine Empire developed into a major hub of Christian culture and religion, with Hagia Sophia being one of its most iconic structures 
built during the reign of Emperor Justinian I. The city flourished as an artistic and intellectual center until the 12th century when it faced 
significant challenges from various invaders, including Arabs, Bulgarians, Crusaders, and Venetians.
In 1453, Constantinople fell to the Ottoman Empire after a protracted siege led by Sultan Mehmed II. The city was renamed Istanbul as part of the empire's 
policy of Islamization, but it retained much of its Greek Byzantine culture and architecture under the new rule. Today, Istanbul is Turkey's largest city and 
an important cultural, economic, and transportation hub. The historical significance of Constantinople/Istanbul lies in its architectural landmarks 
such as Hagia Sophia, The Hippodrome (now Sultanahmet Square), the Chora Church, and many more that showcase a blend of Byzantine, Roman, and Ottoman influences.
"""

SUM_LONG = """
Summarize:
As of now, **NVIDIA Nsight Graphics** does not natively support **WebGPU**, because WebGPU is a relatively new API designed to 
offer more modern graphics capabilities directly from web browsers, whereas Nsight focuses primarily on low-level graphics APIs like Vulkan, Direct3D, and OpenGL. However, there are still ways to profile **WebGPU shaders**, either by using WebGPU-specific tools or by leveraging alternative strategies in conjunction with Nsight.
### Here's how you can approach profiling WebGPU shaders:
### 1. **Browser's Developer Tools**
WebGPU runs inside web browsers, so the primary way to profile WebGPU shaders is through the browser's built-in developer tools.
- **Google Chrome** (and other Chromium-based browsers) have a WebGPU implementation. The **Performance tab** can give some general insight into GPU workload and execution times.
- However, this is not as detailed as using tools like Nsight for native APIs like Vulkan.
You can follow these steps for basic profiling:
   - Open **Chrome DevTools** (F12 or right-click and choose Inspect).
   - Go to the **Performance** tab.
   - Start recording while running your WebGPU workload.
   - Stop the recording to analyze the GPU time and function calls.
For **WebGPU shader debugging**, WebGPU currently doesn't provide as many sophisticated tools for low-level shader profiling as Vulkan or DirectX.
### 2. **Emulate WebGPU with Vulkan and Use Nsight**
To use **NVIDIA Nsight Graphics** or other advanced GPU profiling tools (which are generally more mature than WebGPU's current ecosystem), you can take an indirect approach:
- **WGPU** is a popular Rust-based implementation of WebGPU, which can compile WebGPU code to Vulkan, Metal, or Direct3D 12.
   - By targeting **Vulkan** in a WebGPU project (using WGPU), you can capture the Vulkan API traces in Nsight Graphics.
   - This way, you can benefit from Nsight's advanced GPU profiling and shader analysis tools by profiling the Vulkan backend that your WebGPU project uses.
   Steps:
   1. Set up **WGPU** (or another WebGPU-to-Vulkan translation layer).
   2. Run your WebGPU code, but have it target the Vulkan backend.
   3. Open **NVIDIA Nsight Graphics** and capture a frame from the Vulkan-based WebGPU app.
   4. Analyze shader execution times, memory usage, and other GPU performance metrics using Nsight's tools.
This approach allows you to leverage Vulkan profiling tools on WebGPU projects indirectly.
### 3. **Shader Profiling with Spirv-Cross**
WebGPU shaders are written in **WGSL** or can be translated from HLSL/GLSL into SPIR-V using tools like **SPIRV-Cross**. If you're using SPIR-V shaders, Nsight Graphics can profile them once you've translated the WebGPU pipeline to Vulkan or other supported APIs.
- **SPIR-V** shader code can be compiled using tools like **glslang** or **SPIRV-Tools** to ensure it's compatible with Vulkan.
- Profiling SPIR-V shaders is supported by Nsight, so once you have your WebGPU shaders translated to SPIR-V, you can take advantage of Nsight's shader analysis capabilities.
### 4. **Monitor GPU Performance via Chrome's Internals**
Although Nsight Graphics doesn't support WebGPU directly, you can monitor GPU usage through **Chrome’s Task Manager** (Shift + Esc) to get rough insights into GPU memory usage and execution.
Additionally, Chrome flags like `--enable-gpu-benchmarking` and `--enable-webgpu` might give you more low-level insight into how WebGPU commands are being dispatched.
### 5. **Wait for WebGPU Toolchain Maturity**
As WebGPU matures, tools specifically designed to profile and debug WebGPU shaders will become more common. For example, upcoming features in **Google Chrome DevTools** and other WebGPU-focused browser tools could make shader profiling easier and more accessible without relying on Vulkan backends.
### Conclusion
1. **Direct Nsight support for WebGPU** is currently not available.
2. You can **use browser developer tools** (like Chrome's Performance tab) for high-level profiling.
3. **Convert WebGPU to Vulkan** via **WGPU** or similar projects to profile using Nsight.
4. Use **SPIR-V shaders** for more direct shader profiling via Nsight in Vulkan-based projects.
While the tools for WebGPU shader profiling are not as mature as those for Vulkan or DirectX, the combination of browser tools and Vulkan translation layers can provide insights for performance tuning.
"""

TASK = {
    "easy": "Tell me about Constantinople.",
    "sum": SUM,
    "sum_long": SUM_LONG,
    "lha": "Tell me about the lighthouse of alexandria with details.",
}


def get_args():
    parser = argparse.ArgumentParser(description='tool')
    parser.add_argument('--model', required=True, choices=MODELS.keys(), help='model')
    parser.add_argument('--provider', default="cpu", choices=["cpu", "cuda", "webgpu", "dml"], help='provider')
    parser.add_argument('--max_tokens', type=int, default=9999, help='max_tokens')
    parser.add_argument('--profile', help='profile')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--fp32', action='store_true', help='use fp32, fp16 is the default')
    parser.add_argument('--use-iob', action='store_true', help='use io-bindings')
    parser.add_argument('--quiet', action='store_true', help='no output duing run')
    parser.add_argument('--values', help='dump values')
    parser.add_argument('--csv', help='csv')
    parser.add_argument('--tag', default="test", help='tag')
    parser.add_argument('--platform', default="", help='platform')
    parser.add_argument('--task', default="easy", help='task')
    parser.add_argument('--root', default="models", help='root of models')

    args = parser.parse_args()
    args.model = MODELS.get(args.model, args.model)
    return args


class LLM(object):
    def __init__(self):
        pass

    def setup_session(self, model_path, ep, verbose, profile=False, fp32=False, use_iob=False):
        self.use_iob = use_iob
        self.ep = ep
        self.profile = profile
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)

        self.config = config
        opt = ort.SessionOptions()

        providers = ['CPUExecutionProvider']
        if ep == "cuda":
            providers = ['CUDAExecutionProvider']
        elif ep == 'dml':
            providers = ['DmlExecutionProvider']
        elif ep == 'webgpu':
            providers = ['WebGpuExecutionProvider']

        if verbose:
            opt.log_verbosity_level = 3
            opt.log_severity_level = 1

        if profile:
            opt.enable_profiling = True

        model_file = os.path.join(model_path, "onnx", "model_q4.onnx") if fp32 else os.path.join(model_path, "onnx", "model_q4f16.onnx")
        self.sess = ort.InferenceSession(model_file, sess_options=opt, providers=providers)
        self.device_id = int(self.sess.get_provider_options()[providers[0]]['device_id']) if ep in ['cuda', 'dml'] else 0
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.need_position_id = 'position_ids' in self.input_names
        self.logits_idx = self.output_names.index('logits')
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
            self.iob.bind_output('logits')
            for i, name in enumerate(self.output_names):
                if name.startswith('present'):
                    self.iob.bind_output(name, device_type=self.ep, device_id=self.device_id)
            for i, name in enumerate(self.input_names):
                if name.startswith('past'):
                    v = self.feed[name]
                    self.iob.bind_cpu_input(name, v)

    def update_kv_cache_iob(self):
        for i, name in enumerate(self.output_names):
            if name.startswith('present'):
                new_name = name.replace('present', 'past_key_values')
                ort_output = self.iob.get_outputs()[i]
                # print(f"name: {name} -> {new_name} {ort_output.device_name()}, {ort_output.shape()}, {ort_output.element_type()}")
                self.iob.bind_ortvalue_input(new_name, ort_output)
                self.iob.bind_output(name, device_type=self.ep, device_id=self.device_id)

    def update_kv_cache(self, outputs):
        for i, name in enumerate(self.output_names):
            if name.startswith('present'):
                new_name = name.replace('present', 'past_key_values')
                self.feed[new_name] = outputs[i]

    def generate(self, tokens, **kw_args):
        keep_cache = kw_args.get("keep_cache", 0)
        max_tokens = kw_args.get("max_tokens", 256) + len(tokens)
        values = kw_args.get("values", None)
        cb = kw_args.get("cb", None)
        input_names = ['input_ids', 'attention_mask']

        if keep_cache:
            self.output_tokens.extend(tokens)
        else:
            self.setup_feed()
            self.output_tokens = tokens.tolist()

        feed = self.feed
        feed['input_ids'] = tokens.reshape(1, len(tokens))

        last_token = 0
        seqlen = len(self.output_tokens)
        if self.need_position_id:
            input_names.append('position_ids')
            if keep_cache:
                feed['position_ids'] = np.arange(seqlen - len(tokens), seqlen, dtype=np.int64).reshape([1, len(tokens)])
            else:
                feed['position_ids'] = np.arange(seqlen, dtype=np.int64).reshape([1, seqlen])

        first_token_time = 0
        start_time = time.time()
        while last_token not in self.eos and seqlen < max_tokens:
            feed['attention_mask'] = np.empty([1, seqlen], dtype=np.int64)
            feed['attention_mask'].fill(1)
            if self.use_iob:
                for name in input_names:
                    self.iob.bind_cpu_input(name, feed[name])
                self.iob.bind_output('logits')
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
                    if name.startswith('present.'):
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

            last_token = np.argmax(logits[0, -1, :])
            self.output_tokens.append(last_token)
            feed['input_ids'] = np.array([last_token], dtype=np.int64).reshape(1, 1)
            if self.need_position_id:
                feed['position_ids'] = np.array([seqlen], dtype=np.int64).reshape(1, 1)
            seqlen += 1
            if cb:
                cb(self.output_tokens, seqlen)

        end_time = time.time()

        return self.output_tokens, end_time - start_time, end_time - first_token_time, first_token_time - start_time, seqlen


def main():
    args = get_args()

    model_path = os.path.join(args.root, args.model['path'])
    # if args.provider == "cpu":
    #    model_path = model_path + "-cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)

    llm = LLM()
    llm.setup_session(model_path, args.provider, args.verbose, args.profile, fp32=args.fp32, use_iob=args.use_iob)
    llm.setup_feed()

    query = TASK[args.task]

    # warmup
    message = [{"role": "user", "content": "hello"}]
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=False, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="np", return_attention_mask=False)  # , return_tensor: false, padding: true, truncation: true })
    input_ids = input_ids['input_ids'][0].astype(np.int64)
    prompt_tokens = len(input_ids)
    _ = llm.generate(input_ids, keep_cache=False, max_tokens=1)

    # timed run
    message = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=False, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="np", return_attention_mask=False)  # , return_tensor: false, padding: true, truncation: true })
    input_ids = input_ids['input_ids'][0].astype(np.int64)
    prompt_tokens = len(input_ids)

    def cb(output_tokens, seqlen):
        return print(tokenizer.decode(output_tokens[seqlen - 1:], skip_special_tokens=True), end='', flush=True)

    cb1 = None if args.quiet else cb
    output_tokens, took, gen_time, prompt_time, seqlen = llm.generate(input_ids, keep_cache=False, max_tokens=args.max_tokens, values=args.values, cb=cb1)
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
            f.write("name,took,token_per_sec,tokens,input_tokens,prompt_token_per_sec,gen_token_per_sec,ttft,task,precision,tag,platform,provider,generator,filter,\n")
            f.write(f"{args.model['name']},{took:.1f},{e2e_tps:.1f},{new_tokens_length},{prompt_tokens},{prompt_tps:.1f},{gen_tps:.1f},{prompt_time:.2f},{args.task},{precision},{args.tag},{args.platform},{args.provider},ort-llm.py,1\n")
    return 0


if __name__ == '__main__':
    main()
