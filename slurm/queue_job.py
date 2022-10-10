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
import socket
from nerf_helpers import get_config
import functools

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

# PARAM2SWEEP = (['optimizer','lr'],[1e-5*(2**i) for i in range(9)])
# PARAM2SWEEP = (['dataset','max_scenes'],[10*i for i in range(1,11)])
PARAM2SWEEP = None

LOGS_FOLDER = "/scratch/gpfs/yb6751/projects/VolumetricEnhance/logs"
CONDA_ENV = "torch-env" if 'della-' in socket.gethostname() else "volumetric_enhance"
RUN_TIME = 33 # 20 # 10 # Hours

OVERWRITE_RESUMED_CONFIG = False
# OVERWRITE_RESUMED_CONFIG = True

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

with open(CONFIG_FILE, "r") as f:
    cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = CfgNode(cfg_dict)
    job_name = cfg.experiment.id

if PARAM2SWEEP is None:
    PARAM2SWEEP = ([],[None])

sweep_jobs = len(PARAM2SWEEP[0])>0
sweep_suffix = '_'+''.join([w[0].upper()+w[1:].replace('_','') for w in PARAM2SWEEP[0]]) if sweep_jobs else ''
job_name += sweep_suffix
existing_ids = [int(search("(?<="+job_name+"_)(\d)+(?=$)",f).group(0)) for f in os.listdir(LOGS_FOLDER) if search("^"+job_name+"_(\d)+$",f) is not None]
config_file = ''+CONFIG_FILE
cfg = get_config(config_file)

job_name += '%s'
for run_num in range(len(PARAM2SWEEP[1])):
    run_suffix = str(PARAM2SWEEP[1][run_num]).replace('.','_') if sweep_jobs else ''
    if RESUME_TRAINING is None:
        job_identifier = job_name%(run_suffix)+"_%d"%(0 if len(existing_ids)==0 else max(existing_ids)+1)
        if run_num==0:
            code_folder = os.path.join("slurm/code",job_identifier.replace(sweep_suffix+run_suffix,'_SWP') if sweep_jobs else job_identifier)
            if not os.path.exists(code_folder):
                os.mkdir(code_folder)
        os.mkdir(os.path.join(LOGS_FOLDER,job_identifier))
        print("Starting to train a new job: %s"%(job_identifier))
    else:
        job_identifier = job_name%(run_suffix)+"_%d"%(RESUME_TRAINING)
        if run_num==0:
            code_folder = os.path.join("slurm/code",job_identifier.replace(sweep_suffix+run_suffix,'_SWP') if sweep_jobs else job_identifier)
        saved_models_folder = os.path.join(LOGS_FOLDER,job_identifier)
        assert os.path.isdir(saved_models_folder),"Cannot resume training, since folder %s does not exist."%(saved_models_folder)
        # If a configuration file already exists, checking whether there are any differences with respect to the current confg used. It can happen that an old file does not exist if this is 
        with open(os.path.join(saved_models_folder,"config.yml"), "r") as f:
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
                cfg = saved_config_dict
                # config_file = os.path.join(saved_models_folder,"config.yml")
        print("Resuming training on job %s"%(job_identifier))

    config_filename = "config%s.yml"%(str(run_num) if sweep_jobs else '')
    if RESUME_TRAINING is None or OVERWRITE_RESUMED_CONFIG:
        if run_num==0:
            for f in [f for f in os.listdir() if f[-3:]==".py"]:
                shutil.copyfile(f,os.path.join(code_folder,f))

        if sweep_jobs:
            rsetattr(cfg, '.'.join(PARAM2SWEEP[0]), PARAM2SWEEP[1][run_num])
        setattr(cfg.experiment, 'id',job_identifier)
        with open(os.path.join(code_folder,config_filename), "w") as f:
            f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    python_command = "python train_nerf.py --config %s"%(config_filename)
    if RESUME_TRAINING is not None:
        python_command += " --load-checkpoint %s"%(saved_models_folder)

    with open(os.path.join("slurm/scripts/%s.sh"%(job_identifier)),"w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks=1\n")
        f.write("#SBATCH --job-name=%s\n"%(job_name%(run_suffix)))
        f.write("#SBATCH --cpus-per-task=10\n")
        # f.write("#SBATCH --mem=64G\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBATCH --time=%d:00:00\n"%(RUN_TIME))
        f.write("#SBATCH --mail-user=yb6751@princeton.edu\n")
        f.write("#SBATCH --output=slurm/out/%s.out\n\n"%(job_identifier+"_J%j"))
        f.write("module load anaconda3/2022.5\n")
        f.write("source /home/yb6751/miniconda3/etc/profile.d/conda.sh\n")
        f.write("conda activate %s\n\n"%(CONDA_ENV))
        f.write("cd %s\n"%(os.path.join(code_folder)))
        f.write(python_command)

    call("sbatch %s"%(os.path.join("slurm/scripts/%s.sh"%(job_identifier))),shell=True)