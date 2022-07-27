import os
from re import search
import shutil
import yaml
import sys
sys.path.append(".")
from cfgnode import CfgNode
from subprocess import call
from deepdiff import DeepDiff

# JOB_NAME = "temp"
# CONFIG_FILE = "config/lego_SR.yml"
CONFIG_FILE = "config/lego_ds.yml"
# RESUME_TRAINING = 0
RESUME_TRAINING = None

CONDA_ENV = "/tigress/yb6751/envs/neural_sr"
RUN_TIME = 30 # 20 # 10 # Hours

with open(CONFIG_FILE, "r") as f:
    cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = CfgNode(cfg_dict)
    job_name = cfg.experiment.id

existing_ids = [int(search("(?<="+job_name+"_)(\d)+(?=\.sh)",f).group(0)) for f in os.listdir("slurm/scripts") if search("^"+job_name+"_(\d)+.sh$",f) is not None]
if RESUME_TRAINING is None:
    job_identifier = job_name+"_%d"%(0 if len(existing_ids)==0 else max(existing_ids)+1)
    os.mkdir(os.path.join("slurm/code",job_identifier))
    print("Starting to train a new job: %s"%(job_identifier))
else:
    job_identifier = job_name+"_%d"%(RESUME_TRAINING)
    saved_models_folder = "/tigress/yb6751/projects/NeuralMFSR/logs/%s"%(job_identifier)
    with open(os.path.join("slurm/code",job_identifier,"config.yml"), "r") as f:
        saved_config_dict = CfgNode(yaml.load(f, Loader=yaml.FullLoader))
    config_diffs = DeepDiff(saved_config_dict,cfg)
    diff_warnings = []
    for k,v in config_diffs['values_changed'].items():
        if k=="root['experiment']['id']":   continue
        diff_warnings.append("Configuration values changed comapred to old file:\n %s: %s"%(k,v))
    for ch_type in [c for c in ['dictionary_item_removed','dictionary_item_added'] if c in config_diffs]:
        for diff in config_diffs[ch_type]:
            diff_warnings.append("%s: %s"%(ch_type,diff))
    if len(diff_warnings)>0: 
        shutil.copyfile(os.path.join("slurm/code",job_identifier,"config.yml"),os.path.join("slurm/code",job_identifier,"config_old.yml"))
        print("\n\n!!! WARNING: Differences in configurations file: !!!")
        for warn in diff_warnings:  print(warn)
        print("\n")
    assert os.path.isdir(saved_models_folder),"Cannot resume training, since folder %s does not exist."%(saved_models_folder)
    print("Resuming training on job %s"%(job_identifier))

for f in [f for f in os.listdir() if f[-3:]==".py"]:
    shutil.copyfile(f,os.path.join("slurm/code",job_identifier,f))

id_string = "  id: %s "%(job_name)
with open(os.path.join("slurm/code",job_identifier,"config.yml"),"w") as f_write:
    with open(CONFIG_FILE,"r") as f_read:
        for line in f_read:
            if line[:len(id_string)]==id_string:
                f_write.write(line.replace(id_string,id_string.replace(job_name,job_identifier)))
            else:
                f_write.write(line)


python_command = "python train_nerf.py --config config.yml"
if RESUME_TRAINING is not None:
    python_command += " --load-checkpoint %s"%(saved_models_folder)

with open(os.path.join("slurm/scripts/%s.sh"%(job_identifier)),"w") as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --ntasks=1\n")
    f.write("#SBATCH --job-name=%s\n"%(job_name))
    f.write("#SBATCH --cpus-per-task=12\n")
    f.write("#SBATCH --mem=64G\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --time=%d:00:00\n"%(RUN_TIME))
    f.write("#SBATCH --mail-user=yb6751@princeton.edu\n")
    f.write("#SBATCH --output=slurm/out/%s.out\n\n"%(job_identifier+"_J%j"))
    f.write("module load anaconda3\n")
    f.write("conda activate %s\n\n"%(CONDA_ENV))
    f.write("cd %s\n"%(os.path.join("slurm/code",job_identifier)))
    f.write(python_command)

call("sbatch %s"%(os.path.join("slurm/scripts/%s.sh"%(job_identifier))),shell=True)






