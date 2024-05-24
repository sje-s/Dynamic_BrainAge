import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import  List, Union
from dynamic_brainage.dataloaders.get_dataset import get_dataset
from dynamic_brainage.dfnc.GaussianWindow import SlidingGaussianWindow1d, FixedGaussianWindow1d
WSIZES = [30, 60, 120, 240]

def do_multiwindow(tc):
    windowers = [SlidingGaussianWindow1d(tc.shape[1], wsize, 3, False).to("cuda") for wsize in WSIZES]
    windows = [windower(tc) for windower in windowers]
    window = torch.stack(windows, -1)
    return window.float()

if __name__=="__main__":
    import tqdm
    import sys
    import os
    import pandas as pd
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    index = int(os.getenv("SLURM_ARRAY_TASK_ID", default=0))
    offset = int(os.getenv("DATA_CHUNK_OFFSET", default=0))
    print("Index ", index, " Offset ", offset)
    index += offset
    index -= 1
    print("Index ", index)
    tc_data = get_dataset("tctest",N_subs=9999999999999)
    tc = tc_data[index][0].to(device).squeeze()
    tc = tc.view(1,tc.shape[0],tc.shape[1])

    windowers = [FixedGaussianWindow1d(tc.shape[1], wsize, 3, False).to(device) for wsize in WSIZES]
    pbar = tqdm.tqdm(enumerate(windowers))
    rows = []
    for i, windower in pbar:
        window = windower(tc)
        subject = tc_data.subjects[i]
        session = tc_data.sessions[i]
        original_filepath = tc_data.file_paths[i]
        mask = torch.ones(window.shape[-1],window.shape[-1]).to(device)
        window = window[:,:,torch.triu(mask,diagonal=1)==1]
        wsize = WSIZES[i]
        new_fname_ = 'sample_%d_ws-%d.pth' % (index, wsize)
        new_fname = os.path.join("/data/users3/bbaker/projects/Dynamic_BrainAge/data/tcwindows_v2", new_fname_)
        rows.append(dict(old_filename=original_filepath, subject=subject, session=session, new_filename=new_fname))
        torch.save(window.squeeze(), new_fname)
    pd.DataFrame(rows).to_csv(os.path.join("/data/users3/bbaker/projects/Dynamic_BrainAge/data/tcwindows_v2/csvs", new_fname_ + ".csv"), index=False)
    print("Done")