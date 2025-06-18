import onnxruntime_genai as og
import argparse
import time
import os
import json
from transformers import AutoTokenizer, AutoConfig


# python ort-genai.py --model phi4-8qe --task generation --provider webgpu --use-iob  --static --system-prompt --do_sample --verbose --temperature 1.0 --top_k 1


MAX_LEN = 2048

SYSTEM_PROMPT = "You are a helpful assistant. Give short and concise answers."


def get_args():
    parser = argparse.ArgumentParser(description='tool')
    parser.add_argument('--model', required=True, help='model')
    parser.add_argument('--config', default="ort-llm.json", help='config')
    parser.add_argument('--N', "-N", type=int, default=1, help='measure N times')
    parser.add_argument('--max_tokens', type=int, default=9999, help='max_tokens')
    parser.add_argument('--max_prompt', type=int, default=10*1024, help='chunk prompt at max_prompt')
    parser.add_argument('--provider', default="webgpu", choices=["cpu", "cuda", "webgpu", "dml"], help='provider')
    parser.add_argument('--profile', help='profile')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--fp32', action='store_true', help='use fp32, fp16 is the default')
    parser.add_argument('--use-iob', action='store_true', help='use io-bindings')
    parser.add_argument('--static', action='store_true', help='use static kv_cache')
    parser.add_argument('--pinemb', action='store_true', help='pin embeddings to cpu')
    parser.add_argument('--quiet', action='store_true', help='no output duing run')
    parser.add_argument('--system-prompt', action='store_true', help='use a system prompt')
    parser.add_argument('--csv', help='csv')
    parser.add_argument('--tag', default="main", help='tag')
    parser.add_argument('--platform', default="", help='platform')
    parser.add_argument('--task', default="prefill-500", help='task')
    parser.add_argument('--root', default="models", help='root of models')
    parser.add_argument('--postfix', help='postfix for path')
    parser.add_argument('-ds', '--do_sample', action='store_true', help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-re', '--repetition_penalty', type=float, help='Repetition penalty to sample with')

    parser.description = "Example: python ort-genai.py --provider webgpu --model llama3.2-1b --task generation"
    args = parser.parse_args()
    args.use_iob = True
    return args


def main():
    args = get_args()

    model_root = os.environ.get("ORT_LLM_MODELS", "models")

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = config["models"][args.model]
    model_path = os.path.join(model_root, model["path"])
    if not args.postfix and args.provider in {"cpu", "cuda"}:
        args.postfix = args.provider
    if args.postfix:
        if os.path.exists(model_path + "-" + args.postfix):
            model_path = model_path + "-" + args.postfix
    print(model_path)

    task = config["tasks"][args.task]

    # patch config file
    config_path = os.path.join(model_path, "genai_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        gen_config = json.load(f)
    opt = { 
        "ep.webgpuexecutionprovider.validationMode": "0" 
    }
    if "pinemb" in args.model:
        args.pinemb = args.model["pinemb"]
    if args.pinemb:
        opt["ep.webgpuexecutionprovider.forceCpuNodeNames"] = "/model/embed_tokens/Gather\n/model/embed_tokens/QEGatherScales"
    if "provider_options" in gen_config["model"]["decoder"]["session_options"]:
        del gen_config["model"]["decoder"]["session_options"]["provider_options"]
    if "enable_profiling" in gen_config["model"]["decoder"]["session_options"]:
        del gen_config["model"]["decoder"]["session_options"]["enable_profiling"]
    if args.provider != "cpu":
        gen_config["model"]["decoder"]["session_options"]["provider_options"] = [
            {args.provider: opt}
        ]
    if args.profile:
        gen_config["model"]["decoder"]["session_options"]["enable_profiling"] = args.profile
    gen_config["search"]["max_length"] = MAX_LEN
    gen_config["search"]["past_present_share_buffer"] = args.static
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=4)

    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    search_options = {
        name: getattr(args, name)
        for name in [
            "do_sample",
            "max_length",
            "min_length",
            "top_p",
            "top_k",
            "temperature",
            "repetition_penalty",
        ]
        if name in args and getattr(args, name) is not None
    }
    if args.verbose:
        print(search_options)

    if "max_length" not in search_options:
        search_options["max_length"] = MAX_LEN

    # Keep asking for input prompts in a loop
    tokenizera = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
    message = [{"role": "user", "content": task}]
    prompt = tokenizera.apply_chat_template(message, add_generation_prompt=True, return_dict=False, tokenize=False)
    if args.verbose:
        print(prompt)

    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)

    # warmup
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    generator.generate_next_token()
    del generator

    timestamps = []

    for i in range(args.N):
        new_tokens = []
        generator = og.Generator(model, params)

        if args.system_prompt:
            message = [{"role": "system", "content": SYSTEM_PROMPT}]
            prompt = tokenizera.apply_chat_template(
                message, add_generation_prompt=False, return_dict=False, tokenize=False
            )
            system_prompt_tokens = tokenizer.encode(prompt)
            if args.verbose:
                print(f"system prompt: {prompt}")
            generator.append_tokens(system_prompt_tokens)

        first_token_time = 0
        start = time.time()
        pos = 0
        while pos < len(input_tokens):
            l = min(args.max_prompt, len(input_tokens) - pos)
            if args.verbose:
                print(f"input: {len(input_tokens)}")
            generator.append_tokens(input_tokens[pos : pos + l])
            pos += l
        try:
            while not generator.is_done():
                generator.generate_next_token()
                if not first_token_time:
                    first_token_time = time.time()
                new_token = generator.get_next_tokens()[0]
                if not args.quiet:
                    print(tokenizer_stream.decode(new_token), end="", flush=True)
                new_tokens.append(new_token)
                if len(new_tokens) >= args.max_tokens:
                    break
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")

        end_time = time.time()
        took = end_time - start
        prompt_time = first_token_time - start
        gen_time = end_time - first_token_time
        if gen_time == 0:
            gen_time = 1
        new_tokens_length = len(new_tokens)
        timestamps.append((took, prompt_time, gen_time, new_tokens_length))

    print()

    # sum all entries in timestamps
    # print(timestamps)
    took = sum([t[0] for t in timestamps])
    prompt_time = sum([t[1] for t in timestamps])
    gen_time = sum([t[2] for t in timestamps])
    new_tokens_length = sum([t[3] for t in timestamps])

    prompt_tokens = len(input_tokens)
    e2e_tps = new_tokens_length / took
    took = took / args.N
    prompt_time = prompt_time / args.N
    prompt_tps = prompt_tokens / prompt_time
    gen_tps = new_tokens_length / gen_time

    print(
        f"{new_tokens_length // args.N} tokens in {took:.1f}sec, e2e:{e2e_tps:.1f} tps, prompt: {prompt_tps:.1f} tps, gen: {gen_tps:.1f} tps, ttft: {prompt_time:.2f} sec"
    )

    if args.csv:
        with open(args.csv, "a") as f:
            precision = "q4fp32" if args.fp32 else "q4fp16"
            provider = args.provider
            if args.static:
                provider += "-static"
            f.write(
                f"{args.model},{took:.1f},{e2e_tps:.1f},{new_tokens_length},{prompt_tokens},{prompt_tps:.1f},{gen_tps:.1f},{prompt_time:.2f},{args.task},{precision},{args.tag},{args.platform},{provider},ort-genai.py,1\n"
            )


if __name__ == "__main__":
    main()
