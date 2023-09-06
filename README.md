# ort-perf.html - simple benchmark tool to automate perf testing for onnxruntime-web


## Install
### Dependencies
```
npm install
```

### Models
We use mostly models provided by [transformers.js](https://github.com/xenova/transformers.js/).

The steps to download the models we use:

Set ORT_PERF_DIR is set to root of this project.

Set TJS_DIR to the root of the transformers.js repo.

```
git clone https://github.com/xenova/transformers.js/
```
```
pip install optimum
```
```
cd $TJS_DIR
$ORT_PERF_DIR/download-models.sh
```
ort-perf assumes all models are in the models directory.
Copy or link the transformers.js models to that models directory:
```
cd $ORT_PERF_DIR
mkdir models/
ln -s $TJS_DIR/models models/tjs
```

## Interactive Run
```
npx light-server -s . -p 8888
```

point your browser to http://localhost:8888/ort-perf.html

## Automated run
```
npx light-server -s . -p 8888
npx playwright test
```

The playwright configuration currently is somewhat hard coded. Take a look at [playwright.config.js](playwright.config.js). Also take a look at [ort-perf.spec.js](ort-perf.spec.js) and change settings to your needs.

## Options
[ort-perf](ort-perf.html) is configured by arguments on the url. Currently supported:
#### model
the model path, ie. tjs/t5-small/onnx/encoder_model.onnx
#### name
the model name. 
#### filter
filter pre-configured models to the given set. Currently supported is default and tjs-demo. The later has all models used by https://xenova.github.io/transformers.js/ with similar parameters.
#### provider
wasm|webgpu|webnn

#### device
Only valid for webnn: cpu|gpu

#### threads
number of threads to use

#### profiler
1 = capture onnxruntime profile output for 20 runs

#### gen
How the input data for the model is generated. See [ort-perf.html](ort-perf.html) for supported types.

#### verbose
1 = verbose onnxruntime output

#### min_query_count
Mimimum numbers of queries to run (default=30).

#### min_query_time
Minimum time to run (default=10sec)

#### seqlen
Sequence length if model supports it (default=128)

#### enc_seqlen
Encoder sequence length if models supports it (default=128)

#### go
1 = start benchmark directly

#### csv
1 = generate csv output
