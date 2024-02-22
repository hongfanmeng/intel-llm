from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


model_path = "openbmb/MiniCPM-2B-sft-bf16"
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


save_path = "minicpm-2b-bigdl-lowbit"
model.save_low_bit(save_path)
tokenizer.save_pretrained(save_path)
