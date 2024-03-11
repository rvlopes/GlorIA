import argparse
import os
import re
import torch
from jsonlines import jsonlines
from transformers import AutoTokenizer, GPTNeoForCausalLM, GPTNeoXForCausalLM, GenerationConfig, TextGenerationPipeline, \
    AutoModelForCausalLM
from accelerate import init_empty_weights
from tokenization.tokenizer_loader import TokenizerLoader
from unidecode import unidecode

# Setup load dataset
def load_dataset(datasetName):
    dataset = []
    original_doc_count = 0
    dataset_file_paths = []
    dataset_dir = "/data/rv.lopes/datasets/" + datasetName
    for file in os.listdir(dataset_dir):
        if file.endswith('.json'):
            dataset_file_paths.append(dataset_dir + "/" + file)

    print("Found " + str(len(dataset_file_paths)) + " files for current dataset.")
    for f in dataset_file_paths:
        with jsonlines.open(f) as reader:
            for obj in reader:
                dataset.append(obj)
                original_doc_count += 1
    print("Read a total of " + str(original_doc_count) + " documents.")
    return dataset


def extract_first_word(input_string):
    # Define a regular expression pattern to match the first word
    pattern = r'\b\w+\b'

    # Use the findall function from the re module to find all matches
    matches = re.findall(pattern, input_string)

    if matches:
        return matches[0]
    else:
        return ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calame eval')
    parser.add_argument('-beams', dest='num_beams', default=1, type=int, help='Number of beams', required=False)
    parser.add_argument('-top_k', dest='top_k', default=50, type=int, help='Top k tokens', required=False)
    parser.add_argument('-temp', dest='temperature', default=1.0, type=float, help='Temperature', required=False)
    parser.add_argument('-max_new_tokens', dest='max_new_tokens', type=int, default=5, help='Max new tokens',
                        required=False)
    parser.add_argument('-rep_pen', dest='rep_pen', default=2.0, type=float, help='Repetition Penalty', required=False)
    parser.add_argument('-early_stopping', dest='early_stopping', default=False, action=argparse.BooleanOptionalAction,
                        help='Early stopping to stop when reaching EOS')
    parser.add_argument('-sample', dest='sample', default=False, action=argparse.BooleanOptionalAction,
                        help='Enable sampling')
    parser.add_argument('-model', dest='model', default="gptuga", help='Model to Inference',
                        required=False)
    parser.add_argument('-calame_set', dest='calame_set', default="calame_reviewed", help='calame set to use',
                        required=False)
    parser.add_argument('-gptuganeo_checkpoint', dest='checkpoint', default="checkpoint-3000000", help='GPTugaNeo Checkpoint to use',
                        required=False)
    args = parser.parse_args()

    max_seq_len = 512

    # Load calame
    dataset_name = args.calame_set
    print("Evaluating calame using set ", dataset_name)
    calame = load_dataset(dataset_name)

    # Load tokenizer
    tokenizerLoader = TokenizerLoader("gptuga-tk-512")
    loaded_tokenizer = tokenizerLoader.loadTokenizer(max_seq_len, "GPTNEO-1.3B")

    if "gervasio" in args.model.lower():
        loaded_tokenizer = AutoTokenizer.from_pretrained("PORTULAN/gervasio-ptpt-base", max_len=max_seq_len)
        #loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
    elif "mGPT-13B" in args.model:
        loaded_tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT-13B", device_map="auto", torch_dtype=torch.float16)
    elif "mgpt" in args.model.lower():
        loaded_tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT", max_len=max_seq_len)

    # args.model = wandbrun (only for gptuga)
    modelDir = "/data/rv.lopes/models/" + args.model
    checkpointsDir = os.path.join(modelDir, "checkpoints")
    # For gptuga only
    checkpoint = args.checkpoint

    # GENERATION / DECODER PARAMETERS & STRATEGIES
    # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig
    num_beams = args.num_beams
    num_return_sequences = num_beams
    max_new_tokens = 5
    top_k = args.top_k
    repetition_penalty = args.rep_pen
    temperature = args.temperature
    early_stopping = args.early_stopping
    output_scores = True
    sample = args.sample

    print("Starting calame evaluation for model " + args.model)
    # Load model
    if "gptuga" in args.model:
        test = "/data/rv.lopes/models/GPTugaNeo-1.3B/"
        targetCheckpointDir = os.path.join(checkpointsDir, checkpoint) #checkpointsDir
        model = GPTNeoForCausalLM.from_pretrained(targetCheckpointDir)
        model.to("cuda")
    elif "gervasio" in args.model:
        model = GPTNeoXForCausalLM.from_pretrained("PORTULAN/gervasio-ptpt-base")
        model.to("cuda")
    elif "mGPT-13B" in args.model:
        model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT-13B", device_map="auto", torch_dtype=torch.bfloat16)
    elif "mGPT" in args.model:
        model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT")
        model.to("cuda")

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=sample, top_k=top_k, eos_token_id=model.config.eos_token_id,
        no_repeat_ngram_size=0, num_beams=num_beams, repetition_penalty=repetition_penalty, temperature=temperature,
        output_scores=output_scores, early_stopping=early_stopping
    )

    # Temp fix
    if "mGPT-13B" not in args.model:
        generator = TextGenerationPipeline(model=model, task="text-generation",
                                       tokenizer=loaded_tokenizer, device=0)

    total_samples = len(calame)
    scores = []
    correct_exact_match = 0
    for doc in calame:
        prompt = doc['sentence']
        target_last_word = doc['last_word'].strip()
        if "mGPT-13B" in args.model:
            out = model.generate(**loaded_tokenizer(prompt, return_tensors="pt"), generation_config=generation_config)
            out = loaded_tokenizer.decode(out.tolist()[0]).replace(prompt, "").strip()
            #print(out)
            generated_text = out.replace("\n", "")
        else:
            out = generator(prompt, generation_config=generation_config, return_full_text=False)
            generated_text = out[0]['generated_text'].replace("\n","")
        # Extract first predicted word
        first_word_pattern = r'\b\w+\b'
        predicted_last_word = extract_first_word(generated_text).strip()

        if unidecode(predicted_last_word.lower()) == unidecode(target_last_word.lower()): #unidecode(predicted_last_word.lower()) == unidecode(target_last_word.lower())
            scores.append(1)
            correct_exact_match += 1
        else:
            scores.append(0)
        #print("PROMPT:", prompt)
        #print("TARGET LAST WORD:", target_last_word)
        #print("PREDICTED LAST WORD:", predicted_last_word)
        #print("OUTPUT:", generated_text)
        #print("###########################################################")

    acc = correct_exact_match / total_samples * 100
    print("ACCURACY \ EXACT_MATCH:", acc)
    print("###########################################################")

