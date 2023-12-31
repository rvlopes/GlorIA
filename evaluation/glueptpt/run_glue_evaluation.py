import argparse
import os.path

import jsonlines
import numpy
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from evaluation.assin.Assin2Processor import preprocess_assin2
from evaluation.assin.AssinDataset import AssinDataset
from evaluation.glueptpt.GLUEGenDataset import GLUEGenDataset
from tokenization.tokenizer_loader import TokenizerLoader
from utils.model_selector import load_from_checkpoint
import json


def waterdown_task(glueTaskArg):
    if "rte" in glueTaskArg:
        return "rte"
    if "mrpc" in glueTaskArg:
        return "mrpc"
    if "stsb" in glueTaskArg:
        return "stsb"
    if "wnli" in glueTaskArg:
        return "wnli"


def get_metrics_checkpoint(glueTaskArg):
    if "stsb" in glueTaskArg:
        return ["checkpoint-best-pearson"]
    if "rte" in glueTaskArg or "wnli" in glueTaskArg:
        return ["checkpoint-best-accuracy"]
    if "mrpc" in glueTaskArg:
        return ["checkpoint-best-accuracy", "checkpoint-best-f1"]


def build_glue_checkpoint(targetChkDir, glueTaskArg, bestMetricCheckpoint):
    inner_ft_dir = ""
    if "rte" in glueTaskArg:
        inner_ft_dir = "finetune/glue_rte/ftcheckpoints"
    if "mrpc" in glueTaskArg:
        inner_ft_dir = "finetune/glue_mrpc/ftcheckpoints"
    if "stsb" in glueTaskArg:
        inner_ft_dir = "finetune/glue_stsb/ftcheckpoints"
    if "wnli" in glueTaskArg:
        inner_ft_dir = "finetune/glue_wnli/ftcheckpoints"

    return targetChkDir + "/" + inner_ft_dir + "/" + bestMetricCheckpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that loads a model's finetune checkpoint and evaluates it for ASSIN")
    parser.add_argument('-run_name', dest='runName', help='Pretrain run to choose assin ft checkpoint from')
    parser.add_argument('-base_model', dest='baseModel', help='Base Model - GPTNEO-350, GPTNEO-1.3B, etc')
    parser.add_argument('-run_checkpoint', dest='runCheckpoint',
                        help='Pretrain run checkpoint to choose assin ft checkpoint from')
    parser.add_argument('-glue_task', dest='glueTask', default="glue_rte",
                        help='GLUE Task to eval')
    args = parser.parse_args()

    print(args)

    # Load tokenizer
    tokenizerLoader = TokenizerLoader("gptuga-tk-512")
    loaded_tokenizer = tokenizerLoader.loadTokenizer(128, args.baseModel)

    # Load glue task new test split
    glueDir = "/data/rv.lopes/benchmarks/glueptpt/" + waterdown_task(args.glueTask)
    data = []
    with jsonlines.open(glueDir + "/" + waterdown_task(args.glueTask)+"_new_test.json") as f:
        for doc in f:
            data.append(doc)

    # The following is just logic to load the target checkpoint
    # Feel free to change this according to your own folder organization
    model_dir = os.path.join("/data/rv.lopes/models", args.runName)  # ../model rv.lopes
    target_model_checkpoints_dir = os.path.join(model_dir, "checkpoints")  # ../model/checkpoints
    target_chk = os.path.join(target_model_checkpoints_dir, args.runCheckpoint)  # ../model/checkpoints/checkpoint-1000

    metrics_best_chks = get_metrics_checkpoint(args.glueTask)
    for metric_best_chk in metrics_best_chks:
        final_dir = build_glue_checkpoint(target_chk, args.glueTask, metric_best_chk)
        print("Evaluating "+metric_best_chk)
        # Load model checkpoint
        model, config = load_from_checkpoint(baseModel=args.baseModel, ft_task=args.glueTask,
                                             targetCheck=final_dir)

        data_collator = DataCollatorWithPadding(tokenizer=loaded_tokenizer, max_length=128)

        testDataToLoad = GLUEGenDataset(loaded_tokenizer=loaded_tokenizer,
                                        batchSize=64,
                                        data=data,
                                        seq_len=128)

        # Train dataloader
        test_dataloader = DataLoader(
            testDataToLoad, collate_fn=data_collator, batch_size=64,
        )

        gold_similarity = []
        sys_similarities = numpy.array([])  # stsb
        sys_entailments = numpy.array([])  # rte, mrpc, wnli
        gold_labels = []

        model.to("cuda")
        model.eval()
        for batch in test_dataloader:
            batch.to("cuda")
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                labels=batch['labels'])

            if "rte" in args.glueTask or "mrpc" in args.glueTask or "wnli" in args.glueTask:
                gold_labels += batch['labels'].cpu().tolist()
                preds = outputs.logits.cpu().argmax(dim=1).numpy()
                sys_entailments = numpy.concatenate((sys_entailments, preds))

            if "stsb" in args.glueTask:
                gold_labels += batch['labels'].cpu().tolist()
                preds = outputs.logits.cpu().squeeze().tolist()
                sys_similarities = numpy.concatenate((sys_similarities, preds))

        if "stsb" in args.glueTask:
            assert len(gold_labels) == len(sys_similarities)
            # Eval Similarity
            pearson = pearsonr(gold_labels, sys_similarities)[0]
            corr = spearmanr(gold_labels, sys_similarities)[0]
            print()
            print('STSB evaluation')
            print('Pearson\t\tSpearman')
            print('-------\t\t-------')
            print('{:7.3f}\t\t{:7.3f}'.format(pearson, corr))
        else:
            assert len(gold_labels) == len(sys_entailments)
            # accuracy = (gold_labels == sys_entailments).sum() / len(gold_labels)
            accuracy = sum([1 for gold, pred in zip(gold_labels, sys_entailments) if gold == pred]) / float(len(gold_labels))
            # accuracy = (gold_labels == sys_entailments).sum() / len(gold_labels)
            print()
            print(waterdown_task(args.glueTask) + ' evaluation')
            print('Accuracy')
            print('--------')
            print('{:8.2%}'.format(accuracy))
            if "mrpc" in args.glueTask:
                f1 = f1_score(gold_labels, sys_entailments)
                print('F1')
                print('--------')
                print('{:8.2%}'.format(f1))
