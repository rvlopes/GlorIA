import argparse
import os.path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from evaluation.squad.squad_utils import compute_squad_metrics
from tokenization.tokenizer_loader import TokenizerLoader
from utils.dataset_loader import DatasetLoaderAccelerate
from utils.model_selector import load_from_checkpoint
import json

def build_squad_checkpoint(targetChkDir, bestMetricCheckpoint):
    inner_ft_dir = "finetune/squad1-1/ftcheckpoints"

    return targetChkDir + "/" + inner_ft_dir + "/" + bestMetricCheckpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that loads a model's finetune checkpoint and evaluates it for ASSIN")
    parser.add_argument('-run_name', dest='runName', help='Pretrain run to choose assin ft checkpoint from')
    parser.add_argument('-base_model', dest='baseModel', help='Base Model - GPTNEO-350, GPTNEO-1.3B, etc')
    parser.add_argument('-run_checkpoint', dest='runCheckpoint',
                        help='Pretrain run checkpoint to choose assin ft checkpoint from')

    args = parser.parse_args()
    print(args)

    # Load tokenizer
    tokenizerLoader = TokenizerLoader("gptuga-tk-512")
    maxLength = 512
    stride=20
    loaded_tokenizer = tokenizerLoader.loadTokenizer(maxLength, args.baseModel)

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = loaded_tokenizer(
            questions,
            examples["context"],
            max_length=maxLength,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    # The following is just logic to load the target checkpoint
    # Feel free to change this according to your own folder organization
    model_dir = os.path.join("/data/rv.lopes/models", args.runName)  # ../model rv.lopes
    target_model_checkpoints_dir = os.path.join(model_dir, "checkpoints")  # ../model/checkpoints
    target_chk = os.path.join(target_model_checkpoints_dir, args.runCheckpoint)  # ../model/checkpoints/checkpoint-1000

    data_collator = DataCollatorWithPadding(tokenizer=loaded_tokenizer, max_length=maxLength)

    squad_loader = DatasetLoaderAccelerate("squadpt1-fixed", benchmark=True)
    squad_dev = squad_loader.loadDataset(streaming=False, benchmarkSplit="test")

    squad_dev_dataset = squad_dev.map(preprocess_validation_examples, batched=True,
                                      remove_columns=squad_dev.column_names)

    # Eval dataloader
    squad_dev_dataset_for_model = squad_dev_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        squad_dev_dataset_for_model, collate_fn=data_collator, batch_size=32
    )

    metrics_best_chks = ["checkpoint-best-f1","checkpoint-best-exact_match"]
    for metric_best_chk in metrics_best_chks:
        final_dir = build_squad_checkpoint(target_chk, metric_best_chk)
        print("Evaluating "+metric_best_chk)
        # Load model checkpoint
        model, config = load_from_checkpoint(baseModel=args.baseModel, ft_task="squad",
                                             targetCheck=final_dir)

        metrics = {}
        start_logits = []
        end_logits = []

        model.to("cuda")
        model.eval()
        for batch in eval_dataloader:
            batch.to("cuda")
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'])
                # Accumulate start and end logits
                start_logits += outputs.start_logits.cpu().tolist()
                end_logits += outputs.end_logits.cpu().tolist()

        computed_metrics = compute_squad_metrics(start_logits, end_logits, squad_dev_dataset, squad_dev)
        metrics['f1'] = computed_metrics['f1']
        metrics['exact_match'] = computed_metrics['exact_match']
        print(metrics)