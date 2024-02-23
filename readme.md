# Intel LLM

This repository provides FastChat support for MiniCPM optimized with BigDL-LLM

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Optimizing and Saving the Model to a Low Bit Version

To optimize the model to a low bit version and save it for later use, run the following command:

```bash
python src/optimize_and_save_model.py
```

## Serving using BigDL-LLM and FastChat

First, launch the controller
```bash
python -m fastchat.serve.controller
```

Then, launch the model worker:
```bash
python src/model_worker.py --model-path models/minicpm-2b-bigdl-lowbit --device cpu
```

Finally, launch the Gradio web server
```bash
python src/gradio_web_server.py
```
