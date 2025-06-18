# Simple benchmark tool for onnxruntime-web

# Install

```
npm install
```

# Models

Models are expected under the **models** directory in the parent directory.
Links where to download the models are for most models included in [ort-models.json](ort-models.json).

# Run manual

Start a http server in the parent directory. For example use [http-server.py](../http-server.py).

Visit http://localhost:8888/ort-perf/ort-perf.html and pick the model you want to benchmark.

# Run using playwright

Start a http server in the parent directory. For example use [http-server.py](../http-server.py).

```
npx playwright test ort-perf.spec.js
```
