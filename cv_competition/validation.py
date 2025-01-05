import os
import yaml
import math
import torch

from loguru import logger

import numpy as np

from PIL import Image
from tqdm import tqdm

from time import time
import pandas as pd

from torch.profiler import profile, record_function, ProfilerActivity, schedule


from MODNet.inference.image_matting.inference_class import InfernceIMG

logger.add("log_file.log")

with open("config.yaml") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def uint8(x):
    return x.astype(np.uint8)

def one_clip(x):
    return np.clip(x,0,1) 

def HardJaccard(matte_true, matte_pred):
    return uint8(one_clip(np.array(matte_true)) != one_clip(np.array(matte_pred))).sum()

def MSE(matte_true, matte_pred):
    return ((np.array(matte_true) -  np.array(matte_pred))**2).sum()

def MSE(matte_true, matte_pred):
    return ((np.array(matte_true) -  np.array(matte_pred))**2).sum()


def get_images_paths(config):
    original = config['original_path']
    matte = config['matte_path']

    original_paths = [os.path.join(original,f) for f in  os.listdir(original)]
    matte_paths = [os.path.join(matte,f) for f in  os.listdir(matte)]
    return original_paths, matte_paths

def quality_validation(model):
    logger.info('Running performance validation MSE and JACCARD')
    inferer = InfernceIMG(device='cpu')
    
    mse = []
    jaccard = []
    
    for (original_path, matte_path) in tqdm(zip(*get_images_paths(config))):
        original_image = Image.open(original_path)
        matte_true = Image.open(matte_path)
        matte_pred =  inferer.transform(model, original_image)
        
        mse.append(MSE(matte_true, matte_pred))
        jaccard.append(HardJaccard(matte_true, matte_pred))
    
    metircs = {'MSE':sum(mse)/len(mse),'JACCARD':sum(jaccard)/len(jaccard)}
    logger.success(f'Metrics were computed:{metircs}')
    return metircs


def get_human_readable_size(file_path):
    logger.info('Getting model file size')
    size_bytes = os.path.getsize(file_path)
    
    """
    Convert a file size to a human-readable format.
    """
    if size_bytes == 0:
        return "0B"
    
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    model_size = {'model size': f"{s} {size_name[i]}"}
    logger.success(f'Model size was computed: {model_size}')
    return model_size




def time_lopp(model, n=100, batch_size=1, prof=False):

    time_total = np.zeros(shape=n)
    # run inference
    batch = torch.rand(batch_size,3,512,512)
    for i in range(n):
        start = time()
        with torch.no_grad():
            output = model(batch, True)
        end = time()
        time_total[i] = end - start
        if not prof is False:
            prof.step()
        
    logger.info(f'Execution time for model  (using naive approach): {time_total.mean():.4f} +/- {time_total.std():.4f}')
    
    
def time_examine(model):
    logger.info('Time validation started')
    
    

    my_schedule = schedule(
        skip_first=10,
        wait=1,
        warmup=1,
        active=2,
        repeat=3)

    with profile(activities=[ProfilerActivity.CPU],schedule=my_schedule, profile_memory=True) as prof:
        with record_function("model_inference"):
            time_lopp(model, n=100, batch_size=1,prof=prof)

    prof_results = prof.key_averages()
    df = pd.DataFrame(map(vars, prof_results))
    df = df.set_index('key')
    
    result = {'time ms': df['cpu_time_total'].iloc[0]/3e3}
    logger.success(f'Time estimation was finished: {result} ')
    return result 