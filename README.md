# GlórIA 

*Code no longer mantained, due to having finished my Master's Thesis and starting a new job.*

### This is the code used to perform pre-training, finetuning, and preliminary testing for the GlórIA models. 

**If you simply want to use the models, I recommend you to visit them on HuggingFace (to be made public)**
- [GlórIA 1.3B](https://huggingface.co/rvlopes/glorIA-1.3B)
- [GlórIA 2.7B](https://huggingface.co/rvlopes/glorIA-2.7B)


**License**: GlórIA's usage is restricted to research-only purposes, following the [CC BY-NC 4.0 Deed license](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

# Introduction
The code in this repository supports pre-training a *GPT-like* (GPT2 and GPTNeo) using either a multi-sourced dataset/corpora, or a single text dataset.
It also contains the code to finetune such a model for the following tasks/benchmarks: ASSIN-2, GLUE-PTPT and SquadPT. The code in this repository was ran and tested on a cluster using Slurm and several NVIDIA A100s (Pcie).

GlorIA 1.3B was pre-trained for a total of 3M steps, and the 2.7B version was pre-trained for 1M steps using this code repository.

**Code not tested and most likely will NOT work with TPUs due to not using the Transformers and Accelerate libraries support for TPUs.**

**Author's Notes:** I would like to point out that this was developed for my Master's Thesis, on the research and development of LLMs for the PT-PT language. Due to fast changing requirements, directions, needs, etc; the development of this Trainer and train scripts was subject to regular changes and "rewrites". Initially, this code was supposed to be a general Trainer that can train any kind of generative model - but it turned into something more specific and less extensible than planned. So, even if you can use these entire scripts, I recognize that there are better tools out there to pre-train and finetune your models, and I recommend resorting to this repository only to clear any doubts regarding protocols or for learning purposes. In hindsight, if I were to write my own Trainer again, I would do things radically different with what I have learned now - some design patterns in code could've CLEARLY been simplified - but it is all part of the learning process. 

**Author's Notes:** I will attempt to document anything I deem necessary and insightful to this repository. You can also find my contact at the end for any questions you might have - I will try to answer them even if no longer maintaining the code!

#### **IMPORTANT NOTICE 1)** if you wish to use this code and adapt it, beware of internal directories. Due to an oversight, some directories for folder creation, model saving and loading are somewhat hardcoded. I apologize for this inconvenience. You will most likely need to change or write your own code for saving and loading (altough the logic is in the code).
#### **IMPORTANT NOTICE 2)** Due to the nature of my thesis, the data did not change often and since I used a custom multidataset, I did not implement support to run new datasets in a user-friendly way - that is, you can't just specify a JSON file or a directory and have it load the data (FROM THE ARGUMENTS!). However, you can manually add code to load your own dataset as long as it can be used with a PyTorch Dataloader.
#### **IMPORTANT NOTICE 3)** You may find some references in the code to "BERT", "GPT2" or "BLOOM" models. These are remnants of preliminary tests on previous models before we chose GPTNeo as our base. You can safely ignore these references.
#### **IMPORTANT NOTICE 4)** Besides the previous notices, you may find the name "gptuga". This is the initial/preliminary name I chose and that was used during development - so internally the code doesnt refer to "glorIA" but to "gptuga". So if you see "gptuga", know that it is "glorIA" :)

## Running Scripts

To pre-train, you need to run *run-pretrain.py*. For any specific benchmark, you need to run their respective *run-finetune-task.py*. E.g: To finetune on ASSIN-2, you'd run *run-finetune-assin.py*. 

For finetuning, these *run* files have *premodels* variants. These were created when I was yet learning, and they serve the purpose of dealing with loading the tokenizers from existing pre-trained models on HuggingFace (not trained by me) and with other small nuances that appeared such as specifics directories for my models and actual out-of-the-box pretrained models from HF - this distinction also helped with organization. E.g: to finetune Albertina-PTPT on GLUE-PTPT, I'd run *run-finetune-glue-premodels.py* with the correct arguments, and it would load Albertina's tokenizer, which is done before initializing the Trainer. In hindsight this could've been simplified. 

The same logic is followed for the evaluation scripts. 

### Running Pre-Train
In your conda environment, you can run the following to launch the pre-train script. This example would pretrain a GPTNeo-1.3B, with a batch size of 128 p/ GPU, using 16 gradient accumulation steps - meaning the model will use an inner batch size of 128 / 16 = 8. Uses a learning rate of 1e-4 and a max sequence length of 512 (padded to max len). We specify both 1 epoch and 200k max steps (MS) - however, **since we're specifying the max steps, the number of epochs is ignored**. 
```PowerShell
python -m torch.distributed.launch --nproc_per_node 4 --master_port $master_port \
--use_env run-pretrain.py -b 128 -lr 1e-4 -ml 512 -ga 16 -e 1 -wd 0.01 -ws 10000 -ms 2000000 -ls 100 -ss 250000 \
-fp16 bf16 -hrr 4 -deepspeed -t gptuga-tk-512 -m GPTNEO-1.3B -wandb gptuganeo-1.3B-2M -scheduler cosine_hard
```
The **nproc_per_node** is a torch.distributed param, but it is important here, since it represents the number of DISTRIBUTED PROCESSES that will be launched - **this value corresponds usually to the number of GPUs you are using**. You can view the detailed arguments for pre-training further below.

The *-t* flag is to load our own produced tokenizer (gptuga-tk-512), and *-m* indicated the base model we wanted to use - which was GPTNeo-1.3B initially.

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

## Trainer Behavior Details
### Resuming Training
The resume mechanic works and has been tested for pre-training - it was not implemented for finetuning. 

Using it may seem...unorthodox. When resuming, you want to run the pre-train script with **almost exactly** the same arguments, adding the *-resume* flag and the *-checkpoint [checkpoint]* you wish to resume pre-training from. Due to the nature of our mainly used schedulers you NEED to know the total training steps you will perform in TOTAL (except for linear schedulers) - and during the development of my thesis, part of research was to figure out "how much pre-training"/how many steps would we want, and due to resources, ideally we would pre-train for X steps, and the resume for X+Y steps to see if the model's performance increased.

**In pratice**, to take this need into account, when launching a pre-train for the first time **with the intention of resuming it in the future, you need to specify the maximum number of steps.** This corresponds to the total number of steps. Then, using checkpoint saving, you can stop training and resume at any time with the previous flags. When, resuming training, the training arguments will stay EXACTLY the same with the added flags.

A "gimmick" you can perform to endlessly pre-train your model if you so require is to either:
1) Use a constant learning scheduler. Does not require the total training steps.
2) Or use a cosine annealing scheduler with hard restarts every few steps. E.g: You can pre-train for 1M steps, and perform a hard restart every 500k (-hrr 2). Then, you would **relaunch pre-training with the resume flags, but by changing the *total_train_steps* stored in the *resume_vars***.

For example, we pre-trained our model for 1M steps with 2 hard restarts. This means it performs a hard restart every 500k steps. But now we want to continue its pre-training, **mantaining the states of the scheduler, optimizer, etc**. If we're using **4 GPUs** and want to train for 1M steps, each GPU will perform 1M steps (since the batch is distributed) - so we have 4 x 1M = 4M total training steps. This is the value that will be stored in the *resume_vars*, so if we wanted to pre-train our model up to 2M steps, this value would have to be changed to 4 * 2M = 8M total training steps - and to ensure we perform hard restarts every 500k, we'd have to change the *-hrr* flag to 4 instead of 2, due to scheduler creation and state loading nuances (it will be created with the new values, but its state will be loading through Accelerate, and everything will go on as intended).
```python
if resume_vars is not None:
    # Total Train Steps: represents the TOTAL train steps to be made.
    # Total_train_steps = max_steps * num_GPUs
    total_train_steps = resume_vars['total_steps']  # -> CHANGE THIS VALUE HERE FOR RESUMING AS EXPLAINED ABOVE!
    print("Total train steps from resume:", str(total_train_steps))            
    global_step = resume_vars['global_step'] 
    print("Global step from resume:", str(global_step))
    start_epoch = resume_vars['epoch']  
    max_steps = total_train_steps // gpu_count
```
**Author's Notes:** I'm aware this won't quite work for 'epoch-based pre-training' unless you know the exact number of steps, and that it is completely non-optimal, but implementing this behavior was part a part of my TODO list that I wasn't able to put my hands on. **I will attempt to correct this even if I'm no longer mantaining the code**.

Resuming is only implemented for previous checkpoints that were pre-trained using this code - not TESTED/supported for finetuning cases as I didn't require stopping and resuming for benchmark training and evaluation.

#### Update December:
Uploaded (untested) fix for the previous resume issue. Now you should be able to resume a 1M pre-trained model by specifying, for example, 2M max steps when launching pre-training.
```python
if resume_vars is not None:
    # Total Train Steps: represents the TOTAL train steps to be made.
    # Total_train_steps = max_steps * num_GPUs
    total_train_steps = resume_vars['total_steps'] 
    if max_steps > 0 and max_steps * gpu_count > total_train_steps: # -> Alleged fix
        total_train_steps = max_steps * gpu_count
    print("Total train steps from resume:", str(total_train_steps))            
    global_step = resume_vars['global_step'] 
    print("Global step from resume:", str(global_step))
    start_epoch = resume_vars['epoch']  
    max_steps = total_train_steps // gpu_count
```
### Misc
- The Trainer was built as "steps oriented", meaning it was programmed with the concept of total steps in mind instead of epochs, due to the incredibly large amounts of text we had (35M documents or 35B tokens!), so measuring training in "epochs" was not very helpful. **When possible, launch pre-training with a given number of max steps for simplicity**;
- Specifying max steps WILL override *number of epochs* and run training until it reaches the specified number;
- Specifying save steps WILL override *save on epoch* behavior and will save a checkpoint every X steps;
- The Trainer uses a custom multidataset object to leverage multiple sources and arbitrary weights, but internally supports "regular" datasets as long as they can be used in a PyTorch Dataloader;


## Custom MultiDataset Object
Since we trained a model in a distributed setup, we must make sure that the data is correctly distributed across processes. Not only this, but it's imperative that we're able to build our batches of data according to the weights/sampling probabilities we attribute to our datasets.

To achieve this, two custom classes were implemented. A *MultiDataset* class and a *SingleDataset* class. 
The *SingleDataset* class is responsible for managing a single dataset - a SingleDataset is instantiated for each of our datasets that composed our corpora: 1 for OSCAR, 1 for ClueWeb, 1 for Arquivo.pt, etc. This class handles document counting and iterator instantiation. It also stores statistics, specifically how many samples were seen. According to the processes' rank, each *SingleDataset* will only load a split of the data, to make sure that **each process only reads non-redundant data**. This also \**stops different processes from seeing the same, duplicated data, as each other**. 

The **MultiDataset** class leverages multiple **SingleDataset** classes. It receives an array of these and their respective weights (or sampling probabilities) as input. This class works by being bundled inside a PyTorch Dataloader, together with a WeightedSampler. What happens with this combination is the following: the PyTorch WeightedSampler uses the given weights to generate an index. This index will correspond to one of the datasets (not their documents) that we passed to our *MultiDataset*. Inside it, the indexes are associated with each dataset's iterator, allowing this class to loop over their data - using this index, the *MultiDataset* class retrieves a document using the corresponding dataset's iterator. Accelerate wraps our dataloader and makes sure that the constructed batches are thus distributed across processes. 


## Datasets Pre-Processing
The *datasets_preprocess* folder contains code used for the pre-processing of our multiple sources of data. Some code was adapted from the [BLOOM](https://arxiv.org/abs/2211.05100) ([Data Prep Github](https://github.com/bigscience-workshop/data-preparation)) preprocessing pipeline and the idea behind the pre-processing steps was inspired by [Gopher's](https://arxiv.org/abs/2112.11446).

Some details:
- No deduplication was performed for ClueWeb, since it was already performed by the authors;
- We filtered for ".pt" domain documents using metadata when available (OSCAR and ClueWeb);
- A handwritten bad-word list was created in PT to filter out bad subtitles in the OpenSubtitle dataset to try to avoid polluting the data with toxic content.


## Dependencies
Please check the **environment.yml** file that contains the list of packages and their versions, which can then be used to create and import the Conda environment to a new machine.

## TODO List [Abandoned and Incomplete ATM]:
- Change names on some misleading arguments, such as "fp16" which also supports "bf16";
- Refactor code to multiple trainers, one for each goal: pre-training, ASSIN finetune, GLUE finetune, etc;
- Implement out-of-the-box tokenizer loading for pre-training instead of using my own (currently commented in the code!);
- Revamp the resume mechanic to avoid silly hardcodes :)

## Contacts
For any question or doubt I will try to answer as soon as I can. Feel free to use the following email: rv.lopes[at]campus.fct.unl.pt


## Acknowledgements
We would like to thank Arquivo.pt's team for their content preservation efforts, and for all the help and guidance in accessing the archived web pages at scale.
This work has been partially funded by the FCT project NOVA LINCS Ref. UIDP/04516/2020, by CMU|Portugal project iFetch (CMUP LISBOA-01-0247-FEDER-045920).