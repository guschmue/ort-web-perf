import onnxruntime_genai as og
import argparse
import time
import os

MODELS = {
    "llama3.2-1b": {"name": "llama3.2-1B", "path": "tjs/Llama-3.2-1B-Instruct"},
    "llama3.2-1b-gqa": {"name": "llama3.2-1B-gqa", "path": "tjs/Llama-3.2-1B-Instruct-gqa"},
    "llama3.2-3b": {"name": "llama3.2-3B", "path": "tjs/Llama-3.2-3B-Instruct"},
    "llama3.2-3b-gqa": {"name": "llama3.2-3B-gqa", "path": "tjs/Llama-3.2-3B-Instruct-gqa"},
    "phi3.5": {"name": "phi3.5", "path": "microsoft/Phi-3.5-mini-instruct-onnx-web"},
    "phi3.5-gqa": {"name": "phi3.5-gqa", "path": "microsoft/Phi-3.5-mini-instruct-gqa"},
    "phi3.5-128": {"name": "phi3.5-128", "path": "microsoft/Phi-3.5-mini-instruct-128"},
    "phi3": {"name": "phi3", "path": "microsoft/Phi-3-mini-4k-instruct-onnx-web"},
    "phi3-dml": {"name": "phi3", "path": "microsoft/Phi-3-mini-4k-instruct-onnx-directml"},
    "gemma-2b": {"name": "gemma-2b", "path": "Xenova/gemma-2-2b-it-onnx-web"},
    "smollm": {"name": "smollm", "path": "Xenova/smollm-360M-instruct-add-basics"},
    "tinyllama": {"name": "tinyllama", "path": "schmuell/TinyLlama-1.1B-Chat-v1.0"},
    "smollmv2": {"name": "smollmv2", "path": "tjs/SmolLM2-135M-Instruct-onnx-web"},
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
    parser.add_argument('--max_tokens', type=int, default=9999, help='max_tokens')
    parser.add_argument('--provider', default="cpu", choices=["cpu", "cuda", "webgpu", "dml"], help='provider')
    parser.add_argument('--profile', action='store_true', help='profile')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--fp32', action='store_true', help='use fp32, fp16 is the default')
    parser.add_argument('--use-iob', action='store_true', help='use io-bindings')
    parser.add_argument('--quiet', action='store_true', help='no output duing run')
    parser.add_argument('--csv', help='csv')
    parser.add_argument('--tag', default="main", help='tag')
    parser.add_argument('--platform', default="", help='platform')
    parser.add_argument('--task', default="easy", help='task')
    parser.add_argument('--root', default="models", help='root of models')

    args = parser.parse_args()
    args.model = MODELS.get(args.model, args.model)
    return args


def main():
    args = get_args()

    model_path = os.path.join(args.root, args.model['path'])
    if args.provider == "cpu":
        model_path = model_path + "-cpu"
    if not os.path.exists(model_path):
        model_path = model_path.replace("-onnx-web", "")
    print(model_path)
    model = og.Model(model_path)
    # print(f"device_type={model.device_type}")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    search_options = {name: getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}

    # Set the max length to something sensible by default, unless it is specified by the user,
    # since otherwise it will be set to the entire context length
    if 'max_length' not in search_options:
        search_options['max_length'] = 2048

    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

    # Keep asking for input prompts in a loop
    text = TASK[args.task]
    prompt = f'{chat_template.format(input=text)}'
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)

    # warmup
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    generator.generate_next_token()
    del generator

    new_tokens = []
    first_token_time = 0
    start = time.time()
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    try:
        while not generator.is_done():
            generator.generate_next_token()
            if not first_token_time:
                first_token_time = time.time()
            new_token = generator.get_next_tokens()[0]
            if not args.quiet:
                print(tokenizer_stream.decode(new_token), end='', flush=True)
            new_tokens.append(new_token)
            if len(new_tokens) >= args.max_tokens:
                break
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")

    end_time = time.time()
    print()
    took = end_time - start
    prompt_time = first_token_time - start
    gen_time = end_time - first_token_time
    new_tokens_length = len(new_tokens)
    prompt_tokens = len(input_tokens)
    e2e_tps = new_tokens_length / took
    prompt_tps = prompt_tokens / prompt_time
    gen_tps = new_tokens_length / gen_time

    print(f'{new_tokens_length} tokens in {took:.1f}sec, e2e:{e2e_tps:.1f} tps, prompt: {prompt_tps:.1f} tps, gen: {gen_tps:.1f} tps, ttft: {prompt_time:.2f} sec')

    if args.csv:
        with open(args.csv, "a") as f:
            precision = "q4fp32" if args.fp32 else "q4fp16"
            f.write("name,took,token_per_sec,tokens,input_tokens,prompt_token_per_sec,gen_token_per_sec,ttft,task,precision,tag,platform,provider,generator,filter,\n")
            f.write(f"{args.model['name']},{took:.1f},{e2e_tps:.1f},{new_tokens_length},{prompt_tokens},{prompt_tps:.1f},{gen_tps:.1f},{prompt_time:.2f},{args.task},{precision},{args.tag},{args.platform},{args.provider},ort-genai.py,1\n")


if __name__ == '__main__':
    main()
