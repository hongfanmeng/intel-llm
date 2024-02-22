# Intel LLM

Run controller, worker and web server

```bash
python -m fastchat.serve.controller
python model_worker.py --model-path openbmb/MiniCPM-2B-dpo-bf16 --device cpu
python gradio_web_server.py
```

Or use local model to run

```bash
python src/save_low_bit.py
python -m fastchat.serve.controller
python src/model_worker.py --model-path minicpm-2b-bigdl-lowbit --device cpu
python src/gradio_web_server.py
```


