import argparse
import os
import re
import numpy
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig, TextGenerationPipeline, GPTNeoXForCausalLM, GPTNeoForCausalLM, \
    AutoModelForCausalLM
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from evaluation.assin.Assin2Processor import preprocess_assin2
import random

from tokenization.tokenizer_loader import TokenizerLoader

# DEPRECATED NOT USED
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument('-beams', dest='num_beams', default=4, type=int, help='Number of beams', required=False)
    parser.add_argument('-top_k', dest='top_k', default=50, type=int, help='Top k tokens', required=False)
    parser.add_argument('-temp', dest='temperature', default=1.0, type=float, help='Temperature', required=False)
    parser.add_argument('-max_new_tokens', dest='max_new_tokens', type=int, default=50, help='Max new tokens',
                        required=False)
    parser.add_argument('-rep_pen', dest='rep_pen', default=2.0, type=float, help='Repetition Penalty', required=False)
    parser.add_argument('-early_stopping', dest='early_stopping', default=False, action=argparse.BooleanOptionalAction,
                        help='Early stopping to stop when reaching EOS')
    parser.add_argument('-sample', dest='sample', default=False, action=argparse.BooleanOptionalAction,
                        help='Enable sampling')
    parser.add_argument('-save_file_suffix', dest='save_file_suffix', default="v1", help='Save file suffix',
                        required=False)
    parser.add_argument('-model', dest='model', default="gptuga", help='Model to Inference',
                        required=False)
    parser.add_argument('-assin_ver', dest='assinVersion', default=2, type=int, help='Assin version')

    args = parser.parse_args()
    print(args)

    # Load tokenizer
    seq_len = 1024
    tokenizerLoader = TokenizerLoader("gptuga-tk-512")  # "gptuga-tk-512"
    loaded_tokenizer = tokenizerLoader.loadTokenizer(seq_len, "GPTNEO-1.3B")

    # Load gervasio tokenizer
    if "gervasio" in args.model:
        loaded_tokenizer = AutoTokenizer.from_pretrained("PORTULAN/gervasio-ptpt-base", max_len=seq_len)
        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
    elif "mgpt" in args.model.lower():
        loaded_tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")

    # Set global seed
    random.seed(42)

    # Model Directory Setup
    wandbRun = "gptuganeo-1.3B-2M"  # model folder
    modelDir = "/data/rv.lopes/models/" + wandbRun  # model dir
    checkpoint = "checkpoint-3000000"  # gptuga exclusive
    wandbRun = wandbRun if "gptuga" in args.model else "gervasio"  # change wandb if using gervasio
    # saveDir = "/user/home/rv.lopes/thesis_training/text-generation/" + wandbRun # save dir for text gen
    # save_file_suffix = args.save_file_suffix

    # Only applicable to GPTuga
    checkpointsDir = os.path.join(modelDir, "checkpoints")
    targetCheckpointDir = os.path.join(checkpointsDir, checkpoint)

    # Load assin - version 1 comes from huggingface, while the second needs to
    # be locally stored and loaded, since the huggingface version is not correct
    if args.assinVersion == 1:
        # Load assin1 https://huggingface.co/datasets/assin/viewer/full/train
        print("LOADING ASSIN-1 PTPT")
        data = load_dataset("assin", name="ptpt")
    else:
        # Load assin2
        data = preprocess_assin2("assin2_pickled")

    # Load model
    if "gptuga" in args.model:
        model = GPTNeoForCausalLM.from_pretrained(targetCheckpointDir)
    elif "gervasio" in args.model:
        model = GPTNeoXForCausalLM.from_pretrained("PORTULAN/gervasio-ptpt-base")
    elif "mgpt" in args.model.lower():
        model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT")
    model.to("cuda")

    # GENERATION / DECODER PARAMETERS & STRATEGIES
    # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig
    print("Setting up parameters...")
    num_beams = args.num_beams
    num_return_sequences = num_beams
    max_new_tokens = args.max_new_tokens
    top_k = args.top_k
    repetition_penalty = args.rep_pen
    temperature = args.temperature
    early_stopping = args.early_stopping
    output_scores = True
    sample = args.sample

    # Tokens ids that make up the target generation string
    # isto servia para o forçar a gerar os tokens que representam "entailment" ou "similaridade"
    force_words_ids = [[15, 360, 371, 604, 718, 741, 889, 936, 1033, 1109, 11531, 759, 1651, 5157, 347, 2323, 25, 13]]

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=sample, top_k=top_k, eos_token_id=model.config.eos_token_id,
        no_repeat_ngram_size=0, num_beams=num_beams, repetition_penalty=repetition_penalty, temperature=temperature,
        output_scores=output_scores, early_stopping=early_stopping, force_words_ids=force_words_ids
    )
    generator = TextGenerationPipeline(model=model, task="text-generation",
                                       tokenizer=loaded_tokenizer, device=0)

    num_context_examples = 5
    wrong_generations = 0
    generations_to_process = []
    # Ground truth
    gold_similarity = []
    gold_entailment = []
    # Predictions
    sys_similarity = []
    sys_entailment = []

    # For every test sample
    for test_pair in data['test']:

        # Prepare string for build
        initial_string = "Exemplos Classificados:\n"
        chosen_examples = []

        # We pick X train examples
        for i in range(num_context_examples):
            # Pick one example
            new_example = random.choice(data['train'])
            # Add it to list of examples
            chosen_examples.append(new_example)
            # Prepare string with new example
            example_string = "Premissa: {prem}\nHipotese: {hyp}\nSemelhança: {sim}\nConsequência: {con} \n\n".format(
                prem=new_example['premise'], hyp=new_example['hypothesis'],
                sim=new_example['relatedness_score'], con=new_example['entailment_judgment'])

            initial_string += example_string

        # Add test sample to predict
        initial_string += "Para classificar a Semelhança e Consequência:\n"
        test_string = "Premissa: {prem}\nHipotese: {hyp}".format(
                prem=test_pair['premise'], hyp=test_pair['hypothesis'])
        initial_string += test_string
        out = generator(initial_string, generation_config=generation_config)[0]['generated_text']
        print(out)
        print("######################################################")
        generations_to_process.append(out)

    print("FINISHED INFERENCE STEP")
    # Now that we have generations for each test set, we'll try to process them and retrieve
    # the systems predictions

    # Regular expressions to match "Similarity" and "Consequence" lines
    similarity_pattern = r"Semelhança: ([\d.]+)"
    consequence_pattern = r"Consequência: (\d+)"

    # Regular expressions to match the pattern
    pattern = r"Semelhança: ([\d.]+)\s+Consequência: (\d+)"

    for idx, gen in enumerate(generations_to_process):
        # Find all occurrences of the pattern in the input string
        matches = re.findall(pattern, gen)

        # If the number of matches equals the total pairs of sim+entail that should appear
        # in the string, then the model was able to correctly perform the task and we
        # count it as a good generation. otherwise, we discard it
        if len(matches) == num_context_examples + 1:
            # Save groundtruth
            gold_similarity.append(data['test'][idx]['relatedness_score'])
            gold_entailment.append(data['test'][idx]['entailment_judgment'])

            last_similarity, last_consequence = map(float, matches[-1])
            print(f"Last Similarity: {last_similarity}")
            print(f"Last Consequence: {last_consequence}")
            sys_similarity.append(last_similarity)
            sys_entailment.append(int(last_consequence))

        else:
            print("Pattern not found in the input string")
            wrong_generations += 1

    print(len(gold_entailment))
    print(len(sys_entailment))

    macro_f1 = f1_score(gold_entailment, sys_entailment, average='macro',
                        labels=list(gold_entailment))
    correct_predictions = sum(1 for pred, gt in zip(sys_entailment, gold_entailment) if pred == gt)
    print("correct rte predictions", correct_predictions)
    print("wrong gens", wrong_generations)
    #accuracy = (gold_entailment == sys_entailment).sum() / len(gold_entailment)
    accuracy = correct_predictions / len(gold_entailment)
    pearson = pearsonr(gold_similarity, sys_similarity)[0]
    absolute_diff = numpy.array(gold_similarity) - numpy.array(sys_similarity)
    mse = (absolute_diff ** 2).mean()

    print()
    print('RTE evaluation')
    print('Accuracy\tMacro F1')
    print('--------\t--------')
    print('{:8.2%}\t{:8.3f}'.format(accuracy, macro_f1))

    print()
    print('Similarity evaluation')
    print('Pearson\t\tMean Squared Error')
    print('-------\t\t------------------')
    print('{:7.3f}\t\t{:18.2f}'.format(pearson, mse))

    print("EVAL FINISHED")
