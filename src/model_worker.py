import runpy

from register_minicpm import register_minicpm

if __name__ == "__main__":
    register_minicpm()
    runpy.run_module("bigdl.llm.serving.model_worker", run_name="__main__")
