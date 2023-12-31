def build_wandb_proj(baseModel, wandb_group, run_name):
    # Wandb Preparation
    # https://github.com/huggingface/accelerate/issues/944
    init_kwargs = {"wandb": {"settings": {"console": "off"},
                             "group": wandb_group,
                             "name": run_name}}

    if "GPT2" in baseModel:
        wandbproj = "gptuga-accel"
    elif "NEO" in baseModel:
        wandbproj = "gptuga-neo-accel"
    elif "BLOOM" in baseModel:
        wandbproj = "bloompt"
    else:
        wandbproj = "default-project"
    return wandbproj, init_kwargs


def getWandbRun(model_name, checkpoint=None, task=None, **kwargs):
    # If no checkpoint is passed, we're pretraining
    if checkpoint is None and task is None:
        return model_name
    elif task is not None and checkpoint is not None:  # otherwise we're finetuning, and we must have passed a task
        if task == "assin":
            return model_name + "-" + checkpoint + "-ASSINMULTI-" + str(kwargs['version'])
        if task == "assin_similarity":
            return model_name + "-" + checkpoint + "-ASSINSIM-" + str(kwargs['version'])
        if task == "assin_entailment":
            return model_name + "-" + checkpoint + "-ASSINENTAIL-" + str(kwargs['version'])
        if "squad" in task:
            return model_name + "-" + checkpoint + "-" + task
        if "glue_rte" in task:
            return model_name + "-" + checkpoint + "-" + "GLUE-RTE"
        if "glue_wnli" in task:
            return model_name + "-" + checkpoint + "-" + "GLUE-WNLI"
        if "glue_stsb" in task:
            return model_name + "-" + checkpoint + "-" + "GLUE-STSB"
        if "glue_mrpc" in task:
            return model_name + "-" + checkpoint + "-" + "GLUE-MRPC"
    else: #finetuning preexisting model like bertimbau
        if task == "assin":
            return model_name + "-ASSINMULTI-" + str(kwargs['version'])
        if "glue_rte" in task:
            return model_name + "-GLUE-RTE"
        if "glue_wnli" in task:
            return model_name + "-GLUE-WNLI"
        if "glue_stsb" in task:
            return model_name + "-GLUE-STSB"
        if "glue_mrpc" in task:
            return model_name + "-GLUE-MRPC"
        if "squad" in task:
            return model_name + "-" + task
