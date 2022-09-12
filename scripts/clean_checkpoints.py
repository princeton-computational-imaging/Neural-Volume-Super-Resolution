import os
from tensorboard.backend.event_processing import event_accumulator
from re import search

PARENT_FOLDER = '/scratch/gpfs/yb6751/projects/VolumetricEnhance/logs'
KEYS = ['validation/eval_SR_psnr_gain','validation/SR_psnr_gain','validation/psnr']

for folder in os.listdir(PARENT_FOLDER):
    tb_files = sorted([f for f in os.listdir(os.path.join(PARENT_FOLDER,folder)) if f[:len('events.out.tfevents.')]=='events.out.tfevents.'])
    ckpt_files = [f for f in os.listdir(os.path.join(PARENT_FOLDER,folder)) if f[-5:]=='.ckpt']
    ckpt_files = dict([(int(search('(?<=checkpoint)(\d)+(?=.ckpt)',f).group(0)),f) for f in ckpt_files])
    events = []
    for f in tb_files:
        ea = event_accumulator.EventAccumulator(os.path.join(PARENT_FOLDER,folder,f))
        ea.Reload()
        for key in KEYS:
            if key in ea.Tags()['scalars']:
                events += ea.Scalars(key)
                break
    events = dict([(e.step,e.value) for e in events])