from transformers import GPT2LMHeadModel, BertForMaskedLM, GPT2Config, BertConfig, GPTNeoConfig, GPTNeoForCausalLM, \
    GPT2ForSequenceClassification, GPTNeoForSequenceClassification, GPT2Model, GPTNeoModel, \
    BertForSequenceClassification, BloomForCausalLM, GPTNeoForQuestionAnswering, BertForQuestionAnswering, \
    DebertaV2ForSequenceClassification, \
    DebertaV2Config, DebertaV2ForQuestionAnswering, GPTNeoXConfig, GPTNeoXForQuestionAnswering, \
    GPTNeoXForSequenceClassification

from evaluation.assin.BERTimbauMultiAssin import BERTimbauMultiAssin
from evaluation.assin.DeBERTaMultiAssin import DeBERTaMultiAssin
from evaluation.assin.GPTNeoMultiAssin import GPTNeoMultiAssin
from evaluation.assin.GPTNeoXMultiAssin import GPTNeoXMultiAssin


def select_model_checkpoint(baseModel: str):
    if baseModel == "GPT2-124":
        return GPT2Model.from_pretrained('gpt2')
    if baseModel == "GPT2-355":
        return GPT2LMHeadModel.from_pretrained('gpt2-medium')
    if baseModel == "GPT2-774":
        return GPT2LMHeadModel.from_pretrained('gpt2-large')
    if baseModel == "GPT2-1.5B":
        return GPT2LMHeadModel.from_pretrained('gpt2-xl')
    if baseModel == "GPTNEO-125":
        return GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
    if baseModel == "GPTNEO-350":
        # Bless this person who saved the last available checkpoint for this
        # model size!
        # https://huggingface.co/xhyi/PT_GPTNEO350_ATG
        # model = AutoModelForCausalLM.from_pretrained("xhyi/PT_GPTNEO350_ATG")
        return GPTNeoForCausalLM.from_pretrained('xhyi/PT_GPTNEO350_ATG')
    if baseModel == "GPTNEO-1.3B":
        return GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    if baseModel == "GPTNEO-2.7B":
        return GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
    if baseModel == "glorIA-1.3B":
        return GPTNeoForCausalLM.from_pretrained('rvlopes/glorIA-1.3B')
    if baseModel == "glorIA-2.7B":
        return GPTNeoForCausalLM.from_pretrained('rvlopes/glorIA-2.7B')
    if baseModel == "BLOOM-560":
        return BloomForCausalLM.from_pretrained("bigscience/bloom-560m")


def select_model(baseModel: str):
    if baseModel == "BERT":
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=512
        )
        return BertForMaskedLM(config), config
    if baseModel == "GPT2-124":
        config = GPT2Config(
            vocab_size=50257,  # -> change to 50257 and retrain tokenizer
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_positions=1024
        )
        return GPT2LMHeadModel(config), config
    if baseModel == "GPT2-355":
        config = GPT2Config(
            vocab_size=50257,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_positions=1024,
            n_ctx=1024
        )
        return GPT2LMHeadModel(config), config
    if baseModel == "GPT2-774":
        config = GPT2Config(
            vocab_size=50257,
            n_embd=1280,
            n_layer=36,
            n_head=20,
            n_positions=1024,
            n_ctx=1024
        )
        return GPT2LMHeadModel(config), config
    if baseModel == "GPT2-1.5B":
        config = GPT2Config(
            vocab_size=50257,
            n_embd=1600,
            n_layer=48,
            n_head=25,
            n_positions=1024
        )
        return GPT2LMHeadModel(config), config
    if baseModel == "GPTNEO-125":
        config = GPTNeoConfig(
            vocab_size=50257,
            max_position_embeddings=2048,
            attention_types=[
                [[
                    "global",
                    "local"
                ],
                    6
                ]],
            hidden_size=768,
            num_layers=12,
            num_heads=12
        )
        return GPTNeoForCausalLM(config), config
    if baseModel == "GPTNEO-350":
        config = GPTNeoConfig(
            vocab_size=50257,
            max_position_embeddings=2048,
            attention_types=[
                [[
                    "global",
                    "local"
                ],
                    12
                ]],
            hidden_size=1024,
            num_layers=24,
            num_heads=16
        )
        return GPTNeoForCausalLM(config), config
    if baseModel == "GPTNEO-1.3B":
        config = GPTNeoConfig(
            vocab_size=50257,
            max_position_embeddings=2048,
            attention_types=[
                [[
                    "global",
                    "local"
                ],
                    12
                ]],
            hidden_size=2048,
            num_layers=24,
            num_heads=16
        )
        return GPTNeoForCausalLM(config), config

# mrpc = 2 | rte = 2 | stsb = 1 | wnli = 2
def load_from_checkpoint(baseModel: str, ft_task: str, targetCheck: str):
    if "NEO" in baseModel:
        config = GPTNeoConfig.from_pretrained(targetCheck + "/config.json")
        # ASSIN
        if ft_task == "assin":
            model = GPTNeoMultiAssin.from_pretrained(targetCheck)
        if ft_task == "assin_similarity": # old head - GPTNeoForSequenceRegression
            model = GPTNeoForSequenceClassification.from_pretrained(targetCheck, num_labels=1)
        elif ft_task == "assin_entailment":
            model = GPTNeoForSequenceClassification.from_pretrained(targetCheck, num_labels=2)
        # SQUAD
        elif "squad" in ft_task:
            model = GPTNeoForQuestionAnswering.from_pretrained(targetCheck)
        # GLUE
        elif "glue_mrpc" in ft_task or "glue_rte" in ft_task or "glue_wnli" in ft_task:
            model = GPTNeoForSequenceClassification.from_pretrained(targetCheck, num_labels=2)
        elif "glue_stsb" in ft_task:
            model = GPTNeoForSequenceClassification.from_pretrained(targetCheck, num_labels=1)
        model.config.pad_token_id = model.config.eos_token_id
        return model, config
    elif "GPT2" in baseModel:
        config = GPT2Config.from_pretrained(targetCheck + "/config.json")
        if ft_task == "assin_similarity":
            model = GPT2ForSequenceClassification.from_pretrained(targetCheck, num_labels=1)
        elif ft_task == "assin_entailment":
            model = GPT2ForSequenceClassification.from_pretrained(targetCheck, num_labels=2)
        elif "squad" in ft_task:
            model = GPT2LMHeadModel.from_pretrained(targetCheck)
        model.config.pad_token_id = model.config.eos_token_id
        return model, config
    elif "bertimbau" in baseModel.lower():
        model_name = "neuralmind/bert-large-portuguese-cased"
        if targetCheck is not None:
            model_name = targetCheck
        config = BertConfig.from_pretrained(model_name)
        if ft_task == "assin":
            model = BERTimbauMultiAssin.from_pretrained(model_name)
        if ft_task == "assin_similarity":
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
        elif ft_task == "assin_entailment":
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        # SQUAD
        elif "squad" in ft_task:
            model = BertForQuestionAnswering.from_pretrained(model_name)
        # GLUE
        elif "glue_mrpc" in ft_task or "glue_rte" in ft_task or "glue_wnli" in ft_task:
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        elif "glue_stsb" in ft_task:
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
        #model.config.pad_token_id = model.config.eos_token_id
    elif "albertina" in baseModel.lower():
        model_name = "PORTULAN/albertina-ptpt"
        if targetCheck is not None:
            model_name = targetCheck
        config = DebertaV2Config.from_pretrained(model_name)
        if ft_task == "assin":
            model = DeBERTaMultiAssin.from_pretrained(model_name)
        if ft_task == "assin_similarity":
            model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=1)
        elif ft_task == "assin_entailment":
            model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
        # SQUAD
        elif "squad" in ft_task:
            model = DebertaV2ForQuestionAnswering.from_pretrained(model_name)
        # GLUE
        elif "glue_mrpc" in ft_task or "glue_rte" in ft_task or "glue_wnli" in ft_task:
            model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
        elif "glue_stsb" in ft_task:
            model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=1)
    elif "gervasio" in baseModel.lower():
        model_name = "PORTULAN/gervasio-ptpt-base"
        if targetCheck is not None:
            model_name = targetCheck
        config = GPTNeoXConfig(model_name)
        if ft_task == "assin":
            model = GPTNeoXMultiAssin.from_pretrained(model_name)
        # SQUAD
        elif "squad" in ft_task:
            model = GPTNeoXForQuestionAnswering.from_pretrained(model_name)
        # GLUE
        elif "glue_mrpc" in ft_task or "glue_rte" in ft_task or "glue_wnli" in ft_task:
            model = GPTNeoXForSequenceClassification.from_pretrained(model_name, num_labels=2)
        elif "glue_stsb" in ft_task:
            model = GPTNeoXForSequenceClassification.from_pretrained(model_name, num_labels=1)
    elif "BERT" in baseModel:
        config = BertConfig.from_pretrained(targetCheck + "/config.json")
        return None, config  # BERT NOT BEING USED ATM
    return model, config
