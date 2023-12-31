import math
import os
from pathlib import Path
from statistics import mean

import torch
from accelerate import DeepSpeedPlugin, Accelerator
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, IterableDataset, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer, GPTNeoConfig, GPTNeoForCausalLM, set_seed
from accelerate_pipe.checkpoint_eval_launch import launch_checkpoint_eval
from accelerate_pipe.multi_dataset import MultiDataset
from evaluation.assin.AssinDataset import AssinDataset
from evaluation.squad.squad_utils import compute_squad_metrics
from utils.model_selector import select_model_checkpoint, load_from_checkpoint
from utils.optimizer_and_scheduler_selector import get_optimizer
from utils.trainer_misc_builders import getWandbRun, build_wandb_proj


class TrainerAccelerate:

    def __init__(self, batchSize: int, learningRate: str,
                 weightDecay: float, warmupSteps: int, epochs: int, loggingSteps: int,
                 baseModel: str, wandbRun: str, wandb, tokenizer: PreTrainedTokenizer, maxSteps: int,
                 train_examples: int, eval_examples: int = 0,
                 gradAccum: int = 1, fp16: str = "no", finetune_task: str = None, checkpoint: str = None,
                 saveSteps: int = -1, save_best_checkpoint: bool = False,
                 deepspeed: bool = False, seed: int = 42,
                 **kwargs):

        self.deepspeed = deepspeed
        # Use DeepSpeed integration in Accelerate
        if self.deepspeed:
            # Deepspeed config: can create a more detailed config using the following documentation:
            # https://huggingface.co/docs/accelerate/usage_guides/deepspeed#how-it-works
            # HOWEVER, using that type of config was not thoroughly tested.
            ds_config = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=gradAccum,
                                        gradient_clipping=3)

            # Accelerate
            self.accelerator = Accelerator(log_with="wandb",
                                           gradient_accumulation_steps=gradAccum,
                                           mixed_precision=fp16, deepspeed_plugin=ds_config)
        else:
            # Setup Accelerate
            self.accelerator = Accelerator(log_with="wandb",
                                           gradient_accumulation_steps=gradAccum,
                                           mixed_precision=fp16)

        # Accelerate main objs
        self.train_dataloader, self.model, self.optimizer, self.scheduler = None, None, None, None
        self.config = None

        # Set Random seed: 42 by default
        set_seed(seed)

        # Hyperparams
        self.batchSize = batchSize
        self.lr = learningRate
        self.weightDecay = weightDecay
        self.warmupSteps = warmupSteps
        self.epochs = epochs
        self.maxSteps = maxSteps
        self.loggingSteps = loggingSteps
        self.saveSteps = saveSteps
        # If we specify save_steps, we will use them, otherwise we checkpoint and eval at every epoch
        self.save_on_steps = saveSteps > 0
        self.gradAccum = gradAccum
        self.mixedPrecision = fp16
        # Misc
        self.baseModel = baseModel
        self.tokenizer = tokenizer
        self.lastCheckpointSaved = None
        # Data counters
        self.train_examples = train_examples
        self.eval_examples = eval_examples
        # FT specific
        self.save_best_checkpoint = save_best_checkpoint

        # Prepare Wandb Stuff
        self.wandb = wandb
        # Prepare Run name (name that shows on wandb)
        self.wandbRun = getWandbRun(wandbRun, checkpoint, finetune_task, **kwargs)
        # Wandb group -> groups runs in wandb groups
        # Example
        # wandbRun = gptuga
        # Will group all pretraining runs under 'gptuga_pretraining' and finetunes under 'gptuga_finetune'
        self.wandb_group = wandbRun + "_pretraining" if finetune_task is None else wandbRun + "_finetune"

        # Finetuning vars
        self.finetune_task = finetune_task  # Helper var for specific task/benchmark
        self.checkpoint = checkpoint  # checkpoint to finetune
        self.best_metrics = None  # Dictionary that will register peak metrics depending on each task/benchmark
        self.last_eval_loss = 9999  # Var to register lowest eval loss

        # Model output directories - checkpoint dir for pretrained model checkpoints
        self.checkPointDir = "/data/rv.lopes/models/" + wandbRun + "/checkpoints"
        # Default pre-trained model save dir - may change at end of training if we're finetuning instead
        self.modelSaveDir = "/data/rv.lopes/models/" + wandbRun + "/pretrained"

        # Create checkpoints folder - sanity check!!!!
        if not os.path.exists(self.checkPointDir):
            Path(self.checkPointDir).mkdir(exist_ok=True)

        # If we're finetuning, we specify the folder for this specific task's checkpoints
        if self.finetune_task is not None:
            if self.checkpoint is not None:
                # 'Version' used in ASSIN due to versions 1 and 2
                if "version" in kwargs:
                    self.finetune_dir_build(checkpoint, kwargs['version'])
                else:
                    self.finetune_dir_build(checkpoint)
            else:
                # This branch corresponds to the directory of out-of-the-box pretrained models (albertina, gervasio,
                # bertimbau, etc) that do not have multiple checkpoints (like our own model)
                if "version" in kwargs:
                    self.finetune_dir_build_premodels(kwargs['version'])
                else:
                    self.finetune_dir_build_premodels()

    def prepareModel(self, resume=False, resCheckpoint=None):
        if self.finetune_task is None:
            # Resume pretraining if needed
            if resume:
                loadDir = self.checkPointDir + "/" + resCheckpoint
                if "NEO" in self.baseModel:
                    self.config = GPTNeoConfig.from_pretrained(loadDir + "/config.json")
                    self.model = GPTNeoForCausalLM(self.config)
                return torch.load(loadDir + "/vars.pkl")
            else:
                # PRE TRAINING SETUP
                self.model = select_model_checkpoint(self.baseModel)  # select_model(self.baseModel)

                # Save model config
                self.model.config.save_pretrained(self.modelSaveDir)
                #self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            # FINETUNING SETUP
            targetCheck = None
            if self.checkpoint is not None:
                targetCheck = self.checkPointDir + "/" + self.checkpoint
            self.model, self.config = load_from_checkpoint(baseModel=self.baseModel, ft_task=self.finetune_task,
                                                           targetCheck=targetCheck)

    # Forward for SQUAD FT
    def forward_squad(self, batch):
        # Inputs and labels already shifted inside the model
        outputs = self.model(input_ids=batch['input_ids'],
                             attention_mask=batch['attention_mask'],
                             start_positions=batch['start_positions'],
                             end_positions=batch['end_positions'])
        return outputs

    # Forward for every other type of training objective
    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        # sanity check
        if len(input_ids) > 1:
            input_ids = input_ids.squeeze()
            attention_mask = attention_mask.squeeze()
            labels = labels.squeeze()

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        return outputs

    def build_train_dataloader(self, ft_train_dataloader, weights, multidataset, resume, data_collator):
        # If we're pretraining, we'll be receiving parameters needed for our special
        # multidataset and dataloader
        if self.finetune_task is None and ft_train_dataloader is None:
            if weights is not None:
                # Prepare sampler that will use the arbitrary weights to sample which dataset we will get the
                # next sample from when building a batch
                sampler = WeightedRandomSampler(weights=weights.copy(), num_samples=self.train_examples)
                if resume:
                    self.shuffle_multidataset(multidataset, 0)
                # Train dataloader - we build the dataloader with our sampler, and our custom multidataset object
                return DataLoader(
                    multidataset, collate_fn=data_collator,
                    batch_size=self.batchSize // self.gradAccum,
                    sampler=sampler,
                )
            else:
                # If we dont use weights and just want to pull data randomly from our multidataset oject
                return DataLoader(
                    multidataset, collate_fn=data_collator,
                    batch_size=self.batchSize // self.gradAccum,
                    shuffle=True
                )
        else:
            # Otherwise, we're finetuning and we receive the prebuilt dataloader - normal pytorch behavior.
            return ft_train_dataloader

    def train_loop(self, train_dataloader: DataLoader = None, optim: str = "adamw",
                   scheduler: str = "cosine", resume=False, eval_dataloader=None,
                   dataset: MultiDataset | IterableDataset | Dataset | AssinDataset = None,
                   data_collator=None, weights=None,
                   resume_checkpoint: str = None, hard_rr: int = 0,
                   squad_dev=None, squad_dev_processed=None):  # These last 2 are used for squad ft eval

        # Setup run name and and start wandb tracking
        runName = self.wandbRun if resume is False else self.wandbRun + "-resume"
        wandb_proj, init_kwargs = build_wandb_proj(self.baseModel, self.wandb_group,
                                                   runName)
        self.accelerator.init_trackers(wandb_proj, init_kwargs=init_kwargs)
        if self.accelerator.is_main_process:
            self.accelerator.get_tracker("wandb").name = runName

        # Calculate total training steps
        max_steps = int(self.maxSteps)  # casts for sanity check
        warmup_steps = int(self.warmupSteps) if resume is False else 0
        ga_steps = int(self.gradAccum)
        batchSize_p_device = self.batchSize / ga_steps  # args: 128 = 64 + 64 (ga=2) -> batch size per GPU / GA = batch size per GPU per GA step
        gpu_count = os.environ.get("WORLD_SIZE")
        gpu_count = int(gpu_count) if gpu_count else 1 # always assume we're using 1 GPU
        self.accelerator.print("Gradient Accumulation Steps: " + str(ga_steps))
        self.accelerator.print("Batch Size p/ Device: " + str(batchSize_p_device))
        self.accelerator.print("GPU Count: " + str(gpu_count))
        epochs = int(self.epochs)
        # steps_p_epoch should be equivalent to num of batches in dataloader when distributed
        steps_p_epoch = math.ceil(self.train_examples / batchSize_p_device)
        if max_steps <= 0:
            total_train_steps = steps_p_epoch * self.epochs
        else:
            total_train_steps = max_steps * gpu_count
            epochs = math.ceil((max_steps * gpu_count) / steps_p_epoch)
            self.accelerator.print("Max Steps Specified: " + str(max_steps))
        self.accelerator.print("Train steps p/ epoch: " + str(steps_p_epoch))
        self.accelerator.print("Total Train steps: " + str(total_train_steps) + ". Divide this by the GPU count to"
                                                                                "get the total steps p/ GPU/Process.")

        # Prepare and load model.
        # When resuming, we only prepare the model architecture and the resume variables.
        # !!! We posteriorly resume training by loading the previous Accelerate state!!!
        resume_vars = self.prepareModel(resume=resume, resCheckpoint=resume_checkpoint)

        # IGNORE, DEPRECATED: Sanity check hotfix for gervasio use cases
        if "gervasio" in self.baseModel:
            self.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Starting global step logic
        global_step = 0
        start_epoch = 0
        # https://github.com/huggingface/accelerate/blob/main/examples/by_feature/checkpointing.py#L195
        # for new resume
        if resume_vars is not None:
            # Total Train Steps
            # Represents the TOTAL train steps to be made.
            # total_train_steps = total_steps_per_gpu * num_GPUs
            total_train_steps = resume_vars['total_steps']  # resume_vars['total_steps']  #14000000
            if max_steps > 0 and max_steps * gpu_count > total_train_steps:
                total_train_steps = max_steps * gpu_count
            print("Total train steps from resume:", str(total_train_steps))
            # Global Steps: step to resume training from
            global_step = resume_vars['global_step']  # resume_vars['global_step'] #1000001
            print("Global step from resume:", str(global_step))
            # Start Epoch: epoch to resume training from
            start_epoch = resume_vars['epoch']  # resume_vars['epoch'] #0
            max_steps = total_train_steps // gpu_count
            self.accelerator.print("Max Steps Updated due to resuming from checkpoint:", max_steps)

        # Get optimizer and scheduler
        # IMPORTANT: When accelerate distributes the scheduler, it divides the warmup steps and the
        # total training steps by the number of procs/gpus, and multiplies them by GA steps
        scheduler_warmup_steps = (warmup_steps * gpu_count) // ga_steps
        scheduler_max_steps = total_train_steps // ga_steps  # * gpu_count - not needed
        print("Scheduler Warmup Steps: " + str(scheduler_warmup_steps))
        print("Scheduler Max Steps: " + str(scheduler_max_steps))
        self.optimizer, self.scheduler = get_optimizer(model=self.model, lr=self.lr,
                                                       warmupSteps=scheduler_warmup_steps,
                                                       total_train_steps=scheduler_max_steps,
                                                       optimizer=optim,
                                                       scheduler=scheduler,
                                                       hard_rr=hard_rr)

        # Build dataloader
        aux_train_dataloader = self.build_train_dataloader(ft_train_dataloader=train_dataloader, weights=weights,
                                                           multidataset=dataset,
                                                           resume=resume, data_collator=data_collator)

        # PREPARE ACCELERATE
        eval_loader = eval_dataloader
        if eval_loader is None:
            # If pretraining, we dont receive an eval_dataloader bc we dont do eval - due to time.
            self.train_dataloader, self.model, self.optimizer, self.scheduler = (
                self.accelerator.prepare(aux_train_dataloader, self.model, self.optimizer, self.scheduler)
            )
        else:
            self.train_dataloader, eval_loader, self.model, self.optimizer, self.scheduler = (
                self.accelerator.prepare(aux_train_dataloader, eval_loader, self.model, self.optimizer, self.scheduler)
            )

        # Load all accelerator wrapped modules' states
        # WE EFFECTIVELY RESUME TRAINING HERE BY LOADING THE PREVIOUS ACCEL STATE!!!
        # Can only be used by saving state!!!
        if resume and resume_checkpoint is not None:
            loadDir = self.checkPointDir + "/" + resume_checkpoint
            self.accelerator.load_state(loadDir)

        # Gradient clipping - can be turned into an arg
        max_grad_norm = 3
        self.accelerator.print("Starting training!")
        self.accelerator.print("Epochs: " + str(epochs))
        self.accelerator.print("Total train examples: " + str(self.train_examples))
        if self.eval_examples > 0:
            self.accelerator.print("Total eval examples: " + str(self.eval_examples))

        stop_train = False
        with tqdm(total=total_train_steps, disable=not self.accelerator.is_main_process) as progress_bar:
            for epoch in range(start_epoch, epochs):  # goes from 0 to epoch-1
                if stop_train:
                    break
                self.accelerator.print("Train Epoch: " + str(epoch))
                self.model.train()
                for batch in self.train_dataloader:  # 128 -> 64 + 64 -> GA=2
                    # Accelerate wrapper for GA
                    with self.accelerator.accumulate(self.model):
                        if self.finetune_task is not None and "squad" in self.finetune_task:
                            outputs = self.forward_squad(batch)
                        else:
                            outputs = self.forward(batch)

                        metrics = self.calc_train_metrics(outputs, batch)
                        self.accelerator.backward(metrics['loss'])

                        # Perform clipping when grads are being sync
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                        # Step optimizer and scheduler
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                        # Log every X log steps
                        if global_step % self.loggingSteps == 0:
                            self.log_metrics(metrics=metrics,
                                             learningRate=self.scheduler.get_last_lr()[0],
                                             global_step=global_step)

                        global_step = global_step + 1
                        progress_bar.update(1)

                    # Launch evaluation and save checkpoint - save every X steps if save steps
                    # was specified
                    if self.save_on_steps and global_step % self.saveSteps == 0:
                        checkpoint = "checkpoint-" + str(global_step)
                        self.accelerator.print("Saving and launching evaluation for checkpoint: " + checkpoint)
                        chk_dir = self.get_checkpoint_dir(checkpoint)
                        self.lastCheckpointSaved = chk_dir
                        # Save checkpoint
                        self.save_model(saveDir=chk_dir,
                                        global_step=global_step,
                                        total_steps=total_train_steps,
                                        epoch=epoch)

                        # Launch checkpoint eval: launches eval loop for Finetunes. Used to launch async eval scrip
                        # in cluster for pre-training, which has been removed/turned off.
                        self.do_eval(global_step, checkpoint, eval_loader,
                                     squad_dev, squad_dev_processed)

                        # Save multidataset stats at time of checkpoint saving
                        self.save_multidataset_stats(checkpoint, dataset)

                    # If we specified max_steps, we stop training when we reach it
                    if 0 < max_steps <= global_step: # why did I choose to do this ridiculous condition instead of a simpler version?
                        self.accelerator.print("Reached max steps. Terminating training and saving model.")
                        stop_train = True
                        if self.finetune_task is None:
                            self.save_multidataset_stats("final", dataset)
                            break

                # If save_steps not specified, we checkpoint and eval every epoch
                # we also save the model at the very end of training
                if not self.save_on_steps or epoch == epochs - 1:
                    checkpoint = "checkpoint-" + str(global_step)
                    self.accelerator.print("Saving and launching evaluation for checkpoint: " + checkpoint)
                    chk_dir = self.get_checkpoint_dir(checkpoint)
                    self.lastCheckpointSaved = chk_dir
                    self.checkpoint_on_epoch(checkpoint=checkpoint,
                                             chk_dir=chk_dir,
                                             epoch=epoch,
                                             global_step=global_step,
                                             total_train_steps=total_train_steps,
                                             eval_loader=eval_loader,
                                             squad_dev=squad_dev,
                                             squad_dev_processed=squad_dev_processed)

                # We also reshuffle when using a Multidataset -> this only happens when
                # we're pretraining.
                if self.finetune_task is None and isinstance(dataset, MultiDataset) and epoch != epochs - 1:
                    self.shuffle_multidataset(dataset, epoch)
                    # We recreate the sampler - no other way to reset it
                    new_sampler = WeightedRandomSampler(weights=weights.copy(), num_samples=self.train_examples)
                    # Recreate and prepare dataloader with accelerate to prepare it for distributed
                    self.train_dataloader = self.accelerator.prepare_data_loader(
                        DataLoader(
                            dataset, collate_fn=data_collator,
                            batch_size=self.batchSize // ga_steps,
                            sampler=new_sampler,
                        )
                    )

            # Finish training
            self.accelerator.print("Reached end of training.")
            self.accelerator.end_training()

        # Save model final model - as an oversight, the first branch should never happen. It SHOULD only
        # perform the second branch, which is the correct way of saving.
        if self.lastCheckpointSaved is None:
            self.save_model(saveDir=self.modelSaveDir, final=True)
        else:
            checkpoint = "checkpoint-" + str(global_step)
            chk_dir = self.get_checkpoint_dir(checkpoint)
            self.save_model(saveDir=chk_dir, final=True)
            self.lastCheckpointSaved = chk_dir

    def shuffle_multidataset(self, dataset: MultiDataset, epoch: int):
        self.accelerator.print("Shuffling data!")
        for d in dataset.datasets:
            d.shuffle_data(epoch=epoch + 1)  # we pass epoch as a seed
        # We copy the weights to make sure we pass along the original weights, instead
        # of passing around the modified weights from each epoch iteration of our
        # dataset
        dataset.reset_dataset()

    def save_multidataset_stats(self, checkpoint: str, dataset: MultiDataset):
        if self.finetune_task is None:
            # Save dataset seen data stats
            rank = os.getenv("RANK")
            dataset_stats_dir = "/data/rv.lopes/models/" + self.wandbRun + "/stats"
            if not os.path.exists(dataset_stats_dir):
                os.makedirs(dataset_stats_dir, exist_ok=True)
            dataset.save_stats(checkpoint, rank, dataset_stats_dir)
            if rank == 0:
                dataset.print_stats()

    # Checkpoint only passed when pretraining and launching async eval (async eval deprecated and not in use)
    # Eval_Dataloader only passed when finetuning since we do sync eval
    def do_eval(self, global_step, checkpoint=None, eval_dataloader=None,
                squad_dev=None, squad_dev_processed=None):
        if self.finetune_task is None:
            # if pretraining we launch an async evaluation
            launch_checkpoint_eval(baseRun=self.wandbRun,
                                   checkpoint=checkpoint,
                                   global_step=global_step,
                                   batchSize=self.batchSize,
                                   baseModel=self.baseModel)
        else:
            # If finetune we run the eval loop
            return self.eval_loop(global_step, eval_dataloader,
                                  squad_dev, squad_dev_processed)

    # Calculate metrics for training
    def calc_train_metrics(self, outputs, batch):
        loss = outputs.loss
        metrics = {'loss': loss}
        # Specific models here due to preliminary tests on generative models.
        # Can remove the condition for clarity.
        if ("GPT" in self.baseModel or "BLOOM" in self.baseModel) and self.finetune_task is None:
            metrics['ppl'] = torch.exp(loss)

        return metrics

    # Calculate metrics for evaluation
    def calc_eval_metrics(self, outputs, batch):
        loss = outputs.loss
        metrics = {'loss': loss}
        # For perplexity, same logic as in calc_train_metrics
        if "GPT" in self.baseModel and self.finetune_task is None:
            metrics['ppl'] = torch.exp(loss)

        # METRICS FOR ASSIN BENCHMARK - following official evaluation
        if self.finetune_task == "assin":
            # Unpack label pairs: [RTE, STS] -> [RTE] [STS]
            aux_labels = [[], []]
            for label_pair in batch['labels']:
                aux_labels[0] += [label_pair[0]]
                aux_labels[1] += [label_pair[1]]

            # RTE METRICS
            pred = outputs.logits[0].cpu().argmax(dim=1)
            labels = torch.tensor(aux_labels[0])
            metrics['f1'] = f1_score(y_true=labels, y_pred=pred, average='macro', labels=labels)
            metrics['accuracy'] = accuracy_score(y_true=labels, y_pred=pred)
            # STS METRICS
            prs = pearsonr(torch.tensor(aux_labels[1]), outputs.logits[1].squeeze().cpu()).statistic  # [0]
            # https://stats.stackexchange.com/questions/267152/correlation-with-a-constant
            if math.isnan(prs):
                metrics['pearson'] = 0.0
            else:
                metrics['pearson'] = prs  # [0]

        # GLUE METRICS: rte + winograd -> acc | stsb -> pearson | mrpc -> f1 + acc
        if self.finetune_task == "glue_rte" or self.finetune_task == "glue_wnli":
            # calc acc
            pred = outputs.logits.detach().cpu().argmax(dim=1)
            labels = batch['labels'].cpu()
            metrics['accuracy'] = accuracy_score(y_true=labels, y_pred=pred)

        if self.finetune_task == "glue_stsb":
            # calc pearson
            prs = pearsonr(batch['labels'].detach().cpu(), outputs.logits.squeeze().cpu()).statistic  # [0]
            if math.isnan(prs):
                metrics['pearson'] = 0.0
            else:
                metrics['pearson'] = prs  # [0]

        if self.finetune_task == "glue_mrpc":
            # calc acc e f1
            pred = outputs.logits.detach().cpu().argmax(dim=1)
            labels = batch['labels'].cpu()
            metrics['accuracy'] = accuracy_score(y_true=labels, y_pred=pred)
            metrics['f1'] = f1_score(y_true=labels, y_pred=pred, average='macro', labels=labels)

        return metrics

    # Save model
    def save_model(self, saveDir: str,
                   global_step: int = -1, total_steps: int = -1, epoch: int = -1, final: bool = False):
        if not os.path.exists(saveDir):
            Path(saveDir).mkdir(exist_ok=True)

        # Save model config
        self.accelerator.unwrap_model(self.model).config.save_pretrained(saveDir)

        # Save training values (resume vars)
        if global_step > 0 and total_steps > 0 and epoch >= 0:
            self.accelerator.save({"global_step": global_step, "total_steps": total_steps,
                                   "epoch": epoch}, saveDir + "/vars.pkl")

        self.accelerator.wait_for_everyone()
        # Save model
        if final:
            if not os.path.exists(saveDir):
                Path(saveDir).mkdir(exist_ok=True)

            self.accelerator.unwrap_model(self.model).save_pretrained(saveDir)
            # state_dict = self.accelerator.unwrap_model(self.model).state_dict()
            # self.accelerator.save(state_dict, saveDir + "/model_state.pkl")
        else:
            # Checkpointing
            self.accelerator.save_state(saveDir)
            model = self.accelerator.unwrap_model(self.model)
            state_dict = model.state_dict()
            model.save_pretrained(saveDir)
            self.accelerator.save(state_dict, saveDir + "/model_state.pkl")

    # Log train metrics
    def log_metrics(self, metrics, learningRate, global_step):
        if ("GPT" in self.baseModel or "BLOOM" in self.baseModel) and self.finetune_task is None:
            self.accelerator.log({"ppl": metrics['ppl'], "loss": metrics['loss'],
                                  "learning_rate": float(learningRate)},
                                 step=global_step)
        if self.finetune_task is not None:
            self.accelerator.log({"loss": metrics['loss'],
                                  "learning_rate": float(learningRate)},
                                 step=global_step)

    # Log eval metrics - only done for finetunes
    def log_metrics_eval(self, metrics, learningRate, global_step):
        if self.finetune_task == "assin":
            self.accelerator.log({"f1": metrics['f1'], "eval_loss": metrics['loss'],
                                  "accuracy": metrics['accuracy'], "pearson": metrics['pearson'],
                                  "learning_rate": float(learningRate)},
                                 step=global_step)
        if (self.finetune_task == "assin_entailment") or ("glue_mrpc" in self.finetune_task):
            self.accelerator.log({"f1": metrics['f1'], "eval_loss": metrics['loss'],
                                  "accuracy": metrics['accuracy'],
                                  "learning_rate": float(learningRate)},
                                 step=global_step)
        elif self.finetune_task == "assin_similarity":
            self.accelerator.log({"pearson": metrics['pearson'], "eval_loss_mse": metrics['loss'],
                                  "learning_rate": float(learningRate)},
                                 step=global_step)
        elif "squad" in self.finetune_task:
            self.accelerator.log({"f1": metrics['f1'],
                                  "learning_rate": float(learningRate), "exact_match": metrics['exact_match']},
                                 step=global_step)
        elif "glue_rte" in self.finetune_task or "glue_wnli" in self.finetune_task:
            self.accelerator.log({"eval_loss": metrics['loss'], "accuracy": metrics['accuracy'],
                                  "learning_rate": float(learningRate)}, step=global_step)
        elif "glue_stsb" in self.finetune_task:
            self.accelerator.log({"pearson": metrics['pearson'], "eval_loss": metrics['loss'],
                                  "learning_rate": float(learningRate)},
                                 step=global_step)

    # Eval loop - only for ft - NOT FOR DISTRIBUTED TRAINING!
    def eval_loop(self, global_step, eval_dataloader,
                  squad_dev=None, squad_dev_processed=None):
        self.model.eval()
        # Not all metrics used, depends on benchmark.
        metrics = {}
        f1 = []
        pearson = []
        accuracy = []
        loss = []
        start_logits = []
        end_logits = []
        for batch in eval_dataloader:
            with torch.no_grad():
                if "squad" in self.finetune_task:
                    outputs = self.model(input_ids=batch['input_ids'],
                                         attention_mask=batch['attention_mask'])
                    # Accumulate start and end logits
                    start_logits += outputs.start_logits.cpu().tolist()
                    end_logits += outputs.end_logits.cpu().tolist()
                    loss.append(outputs.loss)
                else:
                    # Run default evaluation and simpler tasks evaluation
                    outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                         labels=batch['labels'])

                    metrics = self.calc_eval_metrics(outputs, batch)
                    loss.append(metrics['loss'].item())
                    if self.finetune_task == "assin":
                        f1.append(metrics['f1'])
                        accuracy.append(metrics['accuracy'])
                        pearson.append(metrics['pearson'])
                    if self.finetune_task == "assin_similarity" or self.finetune_task == "glue_stsb":
                        pearson.append(metrics['pearson'])
                    if self.finetune_task == "assin_entailment" or self.finetune_task == "glue_mrpc":
                        f1.append(metrics['f1'])
                        accuracy.append(metrics['accuracy'])
                    if self.finetune_task == "glue_rte" or self.finetune_task == "glue_wnli":
                        accuracy.append(metrics['accuracy'])

                    # PATIENCE MECHANICS CURRENTLY NOT WORKING AND NOT IN USE - IGNORE THE FOLLOWING CODE
                    # Patience check! - currently at batch level -> need to change
                    # to work at epoch/eval loop level!
                    if metrics['loss'] <= self.last_eval_loss:
                        # If loss improves, we regain patience
                        self.last_eval_loss = metrics['loss']
                        self.lost_patience = 0
                    else:
                        # If loss doesnt improve, we lost patience
                        self.last_eval_loss = metrics['loss']
                        self.lost_patience = self.lost_patience + 1

        # For Squad, we accumulate logits and evaluate it at the end of the eval loop
        if "squad" in self.finetune_task:
            computed_metrics = compute_squad_metrics(start_logits, end_logits, squad_dev_processed, squad_dev)
            metrics['f1'] = computed_metrics['f1']
            metrics['exact_match'] = computed_metrics['exact_match']
        else:
            metrics = {"loss": mean(loss)}
            if self.finetune_task == "assin":
                metrics['f1'] = mean(f1)
                metrics['accuracy'] = mean(accuracy)
                metrics['pearson'] = mean(pearson)
            if self.finetune_task == "assin_similarity" or self.finetune_task == "glue_stsb":
                metrics['pearson'] = mean(pearson)
            if self.finetune_task == "assin_entailment" or self.finetune_task == "glue_mrpc":
                metrics['f1'] = mean(f1)
                metrics['accuracy'] = mean(accuracy)
            if self.finetune_task == "glue_rte" or self.finetune_task == "glue_wnli":
                metrics['accuracy'] = mean(accuracy)

        self.log_metrics_eval(metrics=metrics,
                              learningRate=self.scheduler.get_last_lr()[0],
                              global_step=global_step)
        return metrics

    # Finetune directory builder - only called in __init__ for code org
    def finetune_dir_build(self, checkpoint, version=None):
        auxDirBuilder = self.checkPointDir + "/" + checkpoint
        if not os.path.exists(auxDirBuilder):
            Path(auxDirBuilder).mkdir(exist_ok=True)

        auxDirBuilder = auxDirBuilder + "/finetune/"
        if not os.path.exists(auxDirBuilder):
            Path(auxDirBuilder).mkdir(exist_ok=True)

        # Version for assin version tasks
        if version is None:
            auxDirBuilder = auxDirBuilder + self.finetune_task
        else:
            auxDirBuilder = auxDirBuilder + self.finetune_task + "-" + str(version)
        if not os.path.exists(auxDirBuilder):
            Path(auxDirBuilder).mkdir(exist_ok=True)

        self.ft_checkpoints = auxDirBuilder + "/ftcheckpoints"
        # Another sanity check
        if not os.path.exists(self.ft_checkpoints):
            Path(self.ft_checkpoints).mkdir(exist_ok=True)

    def finetune_dir_build_premodels(self, version=None):
        aux_ft_task = self.finetune_task
        if version is not None:
            aux_ft_task = aux_ft_task + "-" + str(version)
        self.ft_checkpoints = self.checkPointDir + "/" + aux_ft_task
        if not os.path.exists(self.ft_checkpoints):
            Path(self.ft_checkpoints).mkdir(exist_ok=True)

    # Builds checkpoint save directory for a single checkpoint
    def get_checkpoint_dir(self, checkpoint):
        if self.finetune_task is None:  # Default checkpoint dir if pretraining
            chk_dir = self.checkPointDir + "/" + checkpoint
        else:  # Save checkpoint in specific finetune task folder
            chk_dir = self.ft_checkpoints + "/" + checkpoint
        return chk_dir

    def checkpoint_on_epoch(self, checkpoint, chk_dir, global_step, total_train_steps, epoch, eval_loader,
                            squad_dev=None, squad_dev_processed=None):
        # if we're finetuning and we only want to save the best checkpoint for each metric
        if self.finetune_task is not None and self.save_best_checkpoint:
            # If Finetuning, we switch the order, because we want to eval the checkpoint,
            # and only save it if it improves
            metrics = self.do_eval(global_step, checkpoint, eval_loader,
                                   squad_dev, squad_dev_processed)

            if self.best_metrics is None:  # FIRST CHECKPOINT!!
                self.best_metrics = metrics

            if self.finetune_task == "assin":
                self.check_assin_metrics(metrics, global_step, total_train_steps, epoch)

            if self.finetune_task == "glue_rte" or self.finetune_task == "glue_wnli":
                self.check_glue_rte_wnli_metrics(metrics, global_step, total_train_steps, epoch)

            if self.finetune_task == "glue_stsb":
                self.check_glue_stsb_metrics(metrics, global_step, total_train_steps, epoch)

            if self.finetune_task == "glue_mrpc":
                self.check_glue_mrpc_metrics(metrics, global_step, total_train_steps, epoch)

            if "squad" in self.finetune_task:
                self.check_squad_metrics(metrics, global_step, total_train_steps, epoch)
        else:
            # Save checkpoint
            self.save_model(saveDir=chk_dir,
                            global_step=global_step,
                            total_steps=total_train_steps,
                            epoch=epoch + 1)  # we do +1 so if we load the checkpoint, we start at the next epoch

            # Do Eval
            self.do_eval(global_step, checkpoint, eval_loader,
                         squad_dev, squad_dev_processed)

    #
    # Functions to check if a task's specific metrics have improved.
    # If they have, they save the best checkpoint for each metric
    #
    def check_squad_metrics(self, metrics, global_step, total_train_steps, epoch):
        if self.best_metrics['f1'] <= metrics['f1']:
            self.best_metrics['f1'] = metrics['f1']
            chk_dir = self.get_checkpoint_dir("checkpoint-best-f1")
            self.save_model(saveDir=chk_dir,
                            global_step=global_step,
                            total_steps=total_train_steps,
                            epoch=epoch + 1)

        if self.best_metrics['exact_match'] <= metrics['exact_match']:
            self.best_metrics['exact_match'] = metrics['exact_match']
            chk_dir = self.get_checkpoint_dir("checkpoint-best-exact_match")
            self.save_model(saveDir=chk_dir,
                            global_step=global_step,
                            total_steps=total_train_steps,
                            epoch=epoch + 1)

    def check_assin_metrics(self, metrics, global_step, total_train_steps, epoch):
        if self.best_metrics['f1'] <= metrics['f1']:
            self.best_metrics['f1'] = metrics['f1']
            chk_dir = self.get_checkpoint_dir("checkpoint-best-f1")
            self.save_model(saveDir=chk_dir,
                            global_step=global_step,
                            total_steps=total_train_steps,
                            epoch=epoch + 1)

        if self.best_metrics['pearson'] <= metrics['pearson']:
            self.best_metrics['pearson'] = metrics['pearson']
            chk_dir = self.get_checkpoint_dir("checkpoint-best-pearson")
            self.save_model(saveDir=chk_dir,
                            global_step=global_step,
                            total_steps=total_train_steps,
                            epoch=epoch + 1)

    def check_glue_rte_wnli_metrics(self, metrics, global_step, total_train_steps, epoch):
        if self.best_metrics['accuracy'] <= metrics['accuracy']:
            self.best_metrics['accuracy'] = metrics['accuracy']
            chk_dir = self.get_checkpoint_dir("checkpoint-best-accuracy")
            self.save_model(saveDir=chk_dir,
                            global_step=global_step,
                            total_steps=total_train_steps,
                            epoch=epoch + 1)

    def check_glue_stsb_metrics(self, metrics, global_step, total_train_steps, epoch):
        if self.best_metrics['pearson'] <= metrics['pearson']:
            self.best_metrics['pearson'] = metrics['pearson']
            chk_dir = self.get_checkpoint_dir("checkpoint-best-pearson")
            self.save_model(saveDir=chk_dir,
                            global_step=global_step,
                            total_steps=total_train_steps,
                            epoch=epoch + 1)

    def check_glue_mrpc_metrics(self, metrics, global_step, total_train_steps, epoch):
        if self.best_metrics['accuracy'] <= metrics['accuracy']:
            self.best_metrics['accuracy'] = metrics['accuracy']
            chk_dir = self.get_checkpoint_dir("checkpoint-best-accuracy")
            self.save_model(saveDir=chk_dir,
                            global_step=global_step,
                            total_steps=total_train_steps,
                            epoch=epoch + 1)

        if self.best_metrics['f1'] <= metrics['f1']:
            self.best_metrics['f1'] = metrics['f1']
            chk_dir = self.get_checkpoint_dir("checkpoint-best-f1")
            self.save_model(saveDir=chk_dir,
                            global_step=global_step,
                            total_steps=total_train_steps,
                            epoch=epoch + 1)
