from transformers import DataCollatorForLanguageModeling


def get_data_collator(baseModel, loaded_tokenizer):
    if baseModel == "BERT":
        return DataCollatorForLanguageModeling(loaded_tokenizer, mlm=True)
    else:
        return DataCollatorForLanguageModeling(loaded_tokenizer, mlm=False)
