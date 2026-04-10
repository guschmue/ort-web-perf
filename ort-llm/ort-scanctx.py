# python ort-scanctx.py --model qwen3-0.6b --static

import onnxruntime_genai as og
import argparse
import time
import os
import json
from transformers import AutoTokenizer, AutoConfig


MAX_LEN = 31 * 1024


def get_args():
    parser = argparse.ArgumentParser(description="tool")
    parser.add_argument("--model", required=True, help="model")
    parser.add_argument(
        "--text", default="war_of_the_worlds.txt", help="text file with input data"
    )
    parser.add_argument("--config", default="ort-llm.json", help="config")
    parser.add_argument("--max_tokens", type=int, default=100, help="max_tokens")
    parser.add_argument("--step", type=int, default=1024, help="steps")
    parser.add_argument("--max_chunk", type=int, default=4*1024, help="max_chunk")
    parser.add_argument(
        "--provider",
        default="webgpu",
        choices=["cpu", "cuda", "webgpu"],
        help="provider",
    )
    parser.add_argument("--profile", help="profile")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--static", action="store_true", help="use static kv_cache")
    parser.add_argument("--quiet", action="store_true", help="no output duing run")
    parser.add_argument("--csv", help="csv")
    parser.add_argument("--tag", default="main", help="tag")
    parser.add_argument("--platform", default="test", help="platform")
    parser.add_argument("--root", default="models", help="root of models")
    parser.add_argument("--postfix", help="postfix for path")
    parser.add_argument("--max_length", type=int, default=8*1024, help="max generation length")

    parser.description = "Example: python ort-genai.py --provider webgpu --model llama3.2-1b --task generation"
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model_root = args.root

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = config["models"][args.model]
    model_path = os.path.join(model_root, model["path"])
    if not args.postfix and args.provider in {"cpu", "cuda"}:
        if os.path.exists(model_path + "-" + args.provider):
            args.postfix = args.provider
    if args.postfix:
        if os.path.exists(model_path + "-" + args.postfix):
            model_path = model_path + "-" + args.postfix
    print(model_path)

    # patch config file
    config_path = os.path.join(model_path, "genai_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        gen_config = json.load(f)
    opt = {}
    gc = model.get("graph_capture", 0)
    if gc:
        opt["enableGraphCapture"] = "1"
        opt["validationMode"] = "disabled"
    pinemb = model.get("pinemb", False)
    if pinemb:
        opt["forceCpuNodeNames"] = pinemb
    if "provider_options" in gen_config["model"]["decoder"]["session_options"]:
        del gen_config["model"]["decoder"]["session_options"]["provider_options"]
    if "enable_profiling" in gen_config["model"]["decoder"]["session_options"]:
        del gen_config["model"]["decoder"]["session_options"]["enable_profiling"]
    if args.provider != "cpu":
        gen_config["model"]["decoder"]["session_options"]["provider_options"] = [
            {args.provider: opt}
        ]
    if args.profile:
        gen_config["model"]["decoder"]["session_options"][
            "enable_profiling"
        ] = args.profile
    gen_config["search"]["max_length"] = MAX_LEN
    gen_config["search"]["past_present_share_buffer"] = args.static
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=4)

    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.split(" ")
    print(len(text))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, local_files_only=True
    )

    model = og.Model(model_path)
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

    # if "max_length" not in search_options:
    #    search_options["max_length"] = MAX_LEN


    def create_messages(text, size):
        messages = []
        messages.append({"role": "system", "content": "Create a short summary."})
        messages.append({"role": "user", "content": " ".join(text[:size])})
        return tokenizer.encode(
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_dict=False, tokenize=False
            )
        )

    input_tokens = create_messages(text, 512)
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)

    # warmup
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    generator.generate_next_token()
    del generator

    # ---

    output = []

    token_send = 0
    token_received = 0
    token_total = 0

    for i in range(512, args.max_length, args.step):
        input_tokens = []
        token_in = 0
        if i <= args.max_chunk:
            input_tokens.append(create_messages(text, i))
        else:
            j = 0
            while j < i:
                this_time = min(args.max_chunk, i - j)
                input_tokens.append(create_messages(text, this_time))
                j += this_time
        token_in = sum(len(tokens) for tokens in input_tokens)
        iter_start = time.time()
        ttft = 0
        gen = 0
        prefill = 0
        token_out = 0
        try:
            generator = og.Generator(model, params)
            start = time.time()
            for tokens in input_tokens:
                generator.append_tokens(tokens)

            while not generator.is_done():
                generator.generate_next_token()
                if token_out == 0:
                    # First token
                    ttft = time.time() - start
                    start = time.time()
                new_token = generator.get_next_tokens()[0]
                token_out += 1
                if not args.quiet:
                    print(tokenizer.decode(new_token), end="", flush=True)
                if token_out > args.max_tokens:
                    break
            gen_time = time.time() - start
            prefill = token_in / ttft if ttft > 0 else 0
            gen = token_out / gen_time if gen_time > 0 else 0
            token_received += token_out
            token_total += token_in + token_out
            total_time = time.time() - iter_start

            print(f"context-length: {token_in}, ttft: {ttft:.2f}s, prefill: {prefill:.1f} token/s, gen: {gen:.1f} token/s")
            txt = f"{token_in},{ttft:.2f},{prefill:.1f},{gen:.1f},{token_send},{token_received},{total_time:.2f},{args.tag},{args.model},{args.platform},{args.provider}\n"
            output.append(txt)
            if ttft > 60:
                print("ttft > 60, stop testing")
                break

            token_total += token_in + token_out
            if args.profile:
                break

        except KeyboardInterrupt:
            print("break by user")
            break
        except Exception as e:
            print(e)
            break
        generator = None
    print()

    if args.csv:
        need_header = True
        if os.path.exists(args.csv):
            need_header = False
        with open(args.csv, "a", encoding="utf-8") as f:
            if need_header:
                txt = "context_length,ttft,prefill,gen,token_send,token_received,total_time,tag,model,platform,provider\n"
                f.write(txt)
            for i in output:
                f.write(i)


if __name__ == "__main__":
    main()
