# GlórIA 1.3B - A Portuguese European-focused Large Language Model
 
 **GlorIA** is a a large generative language model, with special **focus on European Portuguese**.  It is a 1.3B parameters model, based on [GPTNeo](https://huggingface.co/EleutherAI/gpt-neo-1.3B), which has 24 layers and a hidden size of 2048.


 **Model Resources**:

- **[Paper]** You can check our [paper](https://arxiv.org/abs/2402.12969), accepted in PROPOR 2024.
- **[Pre-trained Model]** The pre-trained model is available in HuggingFace: [GlórIA 1.3B](https://huggingface.co/rvlopes/glorIA-1.3B).


If you find our work usefull, please cite our paper:
```
@InProceedings{gloria_ptpt_propor2024,
  author="Lopes, Ricardo
          and Magalhães, João
          and Semedo, David",
  title="GlórIA: A Generative and Open Large Language Model for Portuguese",
  booktitle="Computational Processing of the Portuguese Language (PROPOR 2024)",
  year="2024",
}
```



## Introduction
The code in this repository implements pre-training of GlórIA using either a multi-sourced dataset/corpora, or a single text dataset. It also contains the code to finetune such a model for the following tasks/benchmarks: ASSIN-2, GLUE-PTPT and SquadPT. The code in this repository was ran and tested on a cluster using Slurm and several NVIDIA A100s (PCIe).


## Training Data
**GlorIA 1.3B** was trained on a large corpora, with aproximately 35B billion tokens. This corpora was built by gathering multiple Portuguese sources:
- [ArquivoPT News PT-PT Dataset](): A collection of 1.4M European Portuguese archived news and periodicals from [Arquivo.pt](https://arquivo.pt/).
- [ClueWeb-Large PT-PT](https://lemurproject.org/clueweb22.php/): Multilingual Corpus, similar to OSCAR. Again, metadata was used to filter only PT-PT webpages.
- [Europarl PT-PT](https://www.statmt.org/europarl/): A parallel corpus with documents such as transcripts from the European Parliament (we only used the PT-PT documents).
- [OpenSubtitles PT-PT](https://opus.nlpl.eu/OpenSubtitles.php): A corpus containing PT-PT subtitles from [OpenSubtitles](http://www.opensubtitles.org/).
- [OSCAR PT-PT](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201): Multilingual Corpus obtained from filtering the Common Crawl corpus. We used metadata to filter only PT-PT webpages.
- [PT WIKI](): The Portuguese Wikipedia. 2022/06/20 Dump.

## Running Scripts

To pre-train, you need to run *run-pretrain.py*. For any specific benchmark, you need to run their respective *run-finetune-task.py*. E.g: To finetune on ASSIN-2, you'd run *run-finetune-assin.py*. 

### Running Pre-Train
In your conda environment, you can run the following to launch the pre-train script. This example would pretrain a GPTNeo-1.3B, with a batch size of 128 p/ GPU, using 16 gradient accumulation steps - meaning the model will use an inner batch size of 128 / 16 = 8. Uses a learning rate of 1e-4 and a  sequence length of 512 (padded to max len). We specify both 1 epoch and 200k max steps (MS) - however, **since we're specifying the max steps, the number of epochs is ignored**. 
```PowerShell
python -m torch.distributed.launch --nproc_per_node 4 --master_port $master_port \
--use_env run-pretrain.py -b 128 -lr 1e-4 -ml 512 -ga 16 -e 1 -wd 0.01 -ws 10000 -ms 2000000 -ls 100 -ss 250000 \
-fp16 bf16 -hrr 4 -deepspeed -t gptuga-tk-512 -m GPTNEO-1.3B -wandb gptuganeo-1.3B-2M -scheduler cosine_hard
```
The **nproc_per_node** is a torch.distributed param, but it is important here, since it represents the number of DISTRIBUTED PROCESSES that will be launched - **this value corresponds usually to the number of GPUs you are using**. You can view the detailed arguments for pre-training further below.

The *-t* flag is used to load our own produced tokenizer (gptuga-tk-512), and *-m* indicates the base model we want to use - which was GPTNeo-1.3B initially.

### Running in SLURM
You can run the following command in a Slurm-supported environment.
```bash
sbatch pretrain_model.sbatch
```
The contents of the *sbatch* file contain the required resources (CPUs, RAM, GPUs, job name, etc) and the actual code to run.
```bash
#!/bin/bash
#SBATCH --job-name=my-pretrain-job
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id
#SBATCH --nodelist=mynode
# You must manually create the folder to store slurm output logs
#SBATCH --output=slurmlogs/%x-%j.out
#SBATCH -e slurmlogs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 16  # number cpus (threads) per task
#SBATCH --mem=320000
#SBATCH --time=0 # No time limit
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:4 #or gpu:4

eval "$(conda shell.bash hook)"

#Activate your anaconda environment
conda activate thesis

#Change dir to where you want to run scripts
cd $PWD

# For pytorch distributed
master_port=$(shuf -i 10000-15000 -n 1)

#Run program 
python -m torch.distributed.launch --nproc_per_node 4 --master_port $master_port \
--use_env run-pretrain.py -b 128 -be 8 -lr 1e-4 -ml 512 -ga 16 -e 1 -wd 0.01 -ws 10000 -ms 2000000 -ls 100 -ss 250000 \
-fp16 bf16 -hrr 4 -deepspeed -t gptuga-tk-512 -m GPTNEO-1.3B -wandb gptuganeo-1.3B-2M -scheduler cosine_hard
```
You will also find examples of slurm scripts that were used during development in this repository. They contain the training arguments aswell.


### Pre-Train (*run-pretrain*) Arguments:
- **-b [INT]** : Batch size PER GPU
- **-ga [INT]** : Number of graddient accumulation steps. If specifying, the inner batch size per GPU will be BS / GA -> this is useful to know when checking OOMs.
- **-lr [STR]** : learning rate, represented as '1e-4' for example.
- **-ml [INT]** : maximum sequence length. shorter sequences will be padded to this length.
- **-e [INT]** : Number of epochs.
- **-ms [INT]** : Maximum number of steps. Will override the 'epochs' param and will run training until it reaches the given number of steps.
- **-wd [FLOAT]** : Weight decay (default 0.01).
- **-ws [INT]** : Warmup Steps for scheduler.
- **-ss [INT]** : Save steps. Will save a checkpoint every X save steps. Overrides save_on_epoch behavior.
- **-scheduler [STR]** : Which scheduler to use. Supports "cosine_hard", "cosine", "linear" and "constant".
- **-hrr [INT]**: Number of hard restarts for scheduler. Only used with "cosine_hard".
- **-ls [INT]** : Log every X steps.
- **-fp16 [STR]** : MISLEADING ARG NAME! Enable mix precision. Supports "fp16" and "bf16" as arguments.
- **-deepspeed** : Enables Deepspeed
- **-wandb [STR]**: Wandb Run Name
- **-m [STR]** : Base Model to start pre-training from. Supports "GPTNEO-1.3B", "GPTNEO-2.7B", "GPT2-124", GPT2-355", "GPT2-774", "GLORIA-1.3B" and "GLORIA-2.7B".
- **-t [STR]** : Used to load custom tokenizer. **If using existing tokenizer from HF for example, you would need to change the code in run-pretrain.py to remove my custom tokenizer loader and simply load one from HF instead.**
- **-resume** : Flag to indicate that we're resuming training.
- **-checkpoint [STR]** : Indicate which checkpoint to resume from. E.g "checkpoint-1000000".


### Misc
- The Trainer was built as "steps oriented", meaning it was programmed with the concept of total steps in mind instead of epochs, due to the incredibly large amounts of text we had (35M documents or 35B tokens!), so measuring training in "epochs" was not very helpful. **When possible, launch pre-training with a given number of max steps for simplicity**;
- Specifying max steps WILL override *number of epochs* and run training until it reaches the specified number;
- Specifying save steps WILL override *save on epoch* behavior and will save a checkpoint every X steps;
- The Trainer uses a custom multidataset object to leverage multiple sources and arbitrary weights, but internally supports "regular" datasets as long as they can be used in a PyTorch Dataloader;


## Datasets Pre-Processing
The *datasets_preprocess* folder contains code used for the pre-processing of our multiple sources of data. Some code was adapted from the [BLOOM](https://arxiv.org/abs/2211.05100) ([Data Prep Github](https://github.com/bigscience-workshop/data-preparation)) preprocessing pipeline and the idea behind the pre-processing steps was inspired by [Gopher's](https://arxiv.org/abs/2112.11446).

Some details:
- No deduplication was performed for ClueWeb, since it was already performed by the authors;
- We filtered for ".pt" domain documents using metadata when available (OSCAR and ClueWeb);
- A handwritten bad-word list was created in PT to filter the OpenSubtitle dataset, avoiding polluting the data with toxic content.


## Dependencies
Please check the **environment.yml** file that contains the list of packages and their versions, which can then be used to create and import the Conda environment to a new machine.


## Contacts
For any question feel free to use the following email: rv.lopes[at]campus.fct.unl.pt
If the previous email does not work, please use: ri cardo val verde 2000[a](g)mail.com


## Acknowledgements
We would like to thank Arquivo.pt's team for their content preservation efforts, and for all the help and guidance in accessing the archived web pages at scale. This work was partially funded by the FCT project NOVA LINCS Ref. UIDP/04516/2020, by CMU|Portugal project iFetch, Ref. CMUP LISBOA-01-0247-FEDER-045920, and by the FCT project Ref. Nº CPCA-IAC/AV/594875/2023.

##  License
GlórIA's usage is restricted to research-only purposes, subject to the ClueWeb22 Dataset license, which can be freely obtained [here](https://www.lemurproject.org/clueweb22/obtain.php).
