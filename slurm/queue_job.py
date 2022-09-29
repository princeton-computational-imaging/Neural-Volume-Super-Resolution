from genericpath import isfile
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
# CONFIG_FILE = "config/lego_ds.yml"
# CONFIG_FILE = "config/planes.yml"
# CONFIG_FILE = "config/planes_SR.yml"
CONFIG_FILE = "config/planes_DTU.yml"
# CONFIG_FILE = "config/planes_SR_DTU.yml"
# CONFIG_FILE = "config/planes_multiScene.yml"
# CONFIG_FILE = "config/planes_internal_SR.yml"

# RESUME_TRAINING = 0
RESUME_TRAINING = None

LOGS_FOLDER = "/scratch/gpfs/yb6751/projects/VolumetricEnhance/logs"
CONDA_ENV = "volumetric_enhance"
RUN_TIME = 40 # 20 # 10 # Hours

OVERWRITE_RESUMED_CONFIG = False
# OVERWRITE_RESUMED_CONFIG = True

with open(CONFIG_FILE, "r") as f:
    cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = CfgNode(cfg_dict)
    job_name = cfg.experiment.id

# existing_ids = [int(search("(?<="+job_name+"_)(\d)+(?=\.sh)",f).group(0)) for f in os.listdir("slurm/scripts") if search("^"+job_name+"_(\d)+.sh$",f) is not None]
existing_ids = [int(search("(?<="+job_name+"_)(\d)+(?=$)",f).group(0)) for f in os.listdir(LOGS_FOLDER) if search("^"+job_name+"_(\d)+$",f) is not None]
config_file = ''+CONFIG_FILE
if RESUME_TRAINING is None:
    job_identifier = job_name+"_%d"%(0 if len(existing_ids)==0 else max(existing_ids)+1)
    if not os.path.exists(os.path.join("slurm/code",job_identifier)):
        os.mkdir(os.path.join("slurm/code",job_identifier))
    os.mkdir(os.path.join(LOGS_FOLDER,job_identifier))
    print("Starting to train a new job: %s"%(job_identifier))
else:
    job_identifier = job_name+"_%d"%(RESUME_TRAINING)
    saved_models_folder = os.path.join(LOGS_FOLDER,job_identifier)
    assert os.path.isdir(saved_models_folder),"Cannot resume training, since folder %s does not exist."%(saved_models_folder)
    # if os.path.isfile(os.path.join("slurm/code",job_identifier,"config.yml")):
    # If a configuration file already exists, checking whether there are any differences with respect to the current confg used. It can happen that an old file does not exist if this is 
    with open(os.path.join(saved_models_folder,"config.yml"), "r") as f:
    # with open(os.path.join("slurm/code",job_identifier,"config.yml"), "r") as f:
        saved_config_dict = CfgNode(yaml.load(f, Loader=yaml.FullLoader))
    config_diffs = DeepDiff(saved_config_dict,cfg)
    diff_warnings = []
    for ch_type in [c for c in ['values_changed','type_changes'] if c in config_diffs]:
        for k,v in config_diffs[ch_type].items():
            if k=="root['experiment']['id']":   continue
            diff_warnings.append("(%s): Configuration values changed comapred to old file:\n %s: %s"%(ch_type,k,v))
    for ch_type in [c for c in ['dictionary_item_removed','dictionary_item_added'] if c in config_diffs]:
        for diff in config_diffs[ch_type]:
            diff_warnings.append("%s: %s"%(ch_type,diff))
    if len(diff_warnings)>0:
        print("\n\n!!! WARNING: Differences in configurations file: !!!")
        for warn in diff_warnings:  print(warn)
        print("\n")
        if OVERWRITE_RESUMED_CONFIG:
            shutil.copyfile(os.path.join(saved_models_folder,"config.yml"),os.path.join("slurm/code",job_identifier,"config_old.yml"))
        else:
            config_file = os.path.join(saved_models_folder,"config.yml")
        # shutil.copyfile(os.path.join("slurm/code",job_identifier,"config.yml"),os.path.join("slurm/code",job_identifier,"config_old.yml"))
    print("Resuming training on job %s"%(job_identifier))

for f in [f for f in os.listdir() if f[-3:]==".py"]:
    shutil.copyfile(f,os.path.join("slurm/code",job_identifier,f))

id_string = "  id: %s "%(job_name)
with open(os.path.join("slurm/code",job_identifier,"config.yml"),"w") as f_write:
    with open(config_file,"r") as f_read:
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






