# ort-perf.html - simple benchmark tool to automate perf testing for onnxruntime-web


## Install
### Dependencies
```
npm install
```

### Models
We use mostly models provided by [transformers.js](https://github.com/xenova/transformers.js/).

The steps to download the models we use:
```
git clone https://github.com/xenova/transformers.js/
```
```
pip install optimum
```
from the transformers.js directory call 
```
download-models.sh
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
