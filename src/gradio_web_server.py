import runpy

from register_minicpm import register_minicpm

if __name__ == "__main__":
    register_minicpm()
    runpy.run_module("fastchat.serve.gradio_web_server", run_name="__main__")
