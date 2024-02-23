import argparse

from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="Convert the original model to low bit model.")
parser.add_argument(
    "--model-path",
    type=str,
    default="openbmb/MiniCPM-2B-dpo-bf16",
    help="The model path of the original model.",
)
parser.add_argument(
    "--output-path",
    type=str,
    default="models/minicpm-2b-bigdl-lowbit",
    help="The output path to save low bit model.",
)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_4bit=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

model.save_low_bit(args.output_path)
tokenizer.save_pretrained(args.output_path)
