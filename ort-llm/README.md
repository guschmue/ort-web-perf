# Simple benchmark tool for llm's using onnxruntime-web

# Install

```
npm install
pip install transformers
```

# Models

Models are expected under the **models** directory in the parent directory.
Links where to download the models are for most models included in [ort-llm.json](ort-llm.json).

# onnxruntime + webgpu native
```
export ORT_LLM_MODELS=$PWD/models
```

Using [ort-llm.py](ort-llm.py). This uses the onnxruntime api directly and is convinient for debugging and profiling.
```
python ort-llm.py --model qwen3-0.6b --task prefill-500
```

Using [ort-genai.py](ort-genai.py). This uses the onnxruntime-genai api, recommended for performance measurements.
```
python ort-llm.py --model qwen3-0.6b --static --task prefill-500
```

ort-llm.py and ort-genai.py have identical command line argument.

# onnxruntime-web + webgpu

## Run manual

Start a http server in the parent directory. For example use [http-server.py](../http-server.py).

Visit http://localhost:8888/ort-llm/ort-llm.html?model=qwen3-0.6b.

You can pick one of the models defined in [ort-llm.json](ort-llm.json).


## Run using playwright

Start a http server in the parent directory. For example use [http-server.py](../http-server.py).

```
npx playwright test ort-llm.spec.js
```
