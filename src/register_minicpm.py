from fastchat.conversation import Conversation, SeparatorStyle, get_conv_template, register_conv_template
from fastchat.model.model_adapter import BaseModelAdapter, register_model_adapter
from transformers import AutoTokenizer


class MiniCPMAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "minicpm" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        print("Customized MiniCPM loader")

        from bigdl.llm.transformers import AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if "bigdl-lowbit" in model_path:
            model = AutoModelForCausalLM.load_low_bit(
                model_path,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,
                trust_remote_code=True,
            )

        return model, tokenizer

    def get_default_conv_template(self, model_path) -> Conversation:
        return get_conv_template("MiniCPM")


def register_minicpm():
    register_conv_template(
        Conversation(
            name="MiniCPM",
            system_template="<s> You are a helpful assistant.",
            roles=("<用戶>", "<AI>"),
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            sep="",
            stop_token_ids=[2],  # </s>
            stop_str="</s>",
        )
    )
    register_model_adapter(MiniCPMAdapter)
