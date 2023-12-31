import json
import os
import subprocess


def launch_checkpoint_eval(baseRun, checkpoint, global_step, batchSize, baseModel):
    world_rank = os.environ.get("RANK")
    is_distributed = world_rank is not None
    can_submit_job = ((is_distributed and int(world_rank) == 0) or not is_distributed)

    if can_submit_job:
        runName = baseRun
        # TEMPORARY REMOVED
        #subprocess.run(["sbatch", "model-accel-eval-checkpoint.sbatch", runName,
        #                checkpoint, str(batchSize), str(global_step),
        #                baseModel])

