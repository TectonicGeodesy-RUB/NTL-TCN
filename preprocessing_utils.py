import scipy
import numpy as np
import torch
from torch.utils.data import DataLoader


def inferece_loop(dataloader: DataLoader, model: torch.nn.Module):
    """
    args:
        - dataloader

    returns:
        - target, prediction
    """

    model.eval()
    pred_arr = []
    tar_arr = []
    tar_time_arr = []
    # iterate over mini-batches
    with torch.no_grad():
        # with tqdm(dataloader, unit='batch', desc=f'Test Data') as loader_epoch:
        for i_batch, (inp, aux, tar, tar_time) in enumerate(dataloader):
            inp, aux, tar = (inp.cuda(non_blocking=True),
                             aux.cuda(non_blocking=True),
                             tar.cuda(non_blocking=True))
            pred = model(inp, aux)

            pred_arr.append(pred.detach().cpu().numpy())
            tar_arr.append(tar.detach().cpu().numpy())
            tar_time_arr.append(tar_time)

    return tar_arr[0], pred_arr[0], tar_time_arr


def amp_spec(y: np.ndarray, sampling_rate: float = 365.25) -> tuple[np.ndarray, np.ndarray]:
    """
    sampling_rate in 1/year
    """
    T = 1. / sampling_rate  # sample spacing
    n = int(2 * sampling_rate)

    y_fft = 2 / n * abs(scipy.fft.rfft(y, n=n))
    x_fft = scipy.fft.rfftfreq(n, T)

    return x_fft, y_fft


def butterworth_filter(data: np.ndarray,
                       ftype: str = 'lowpass',
                       cutoff: float = 0.2,
                       fs: float = 1,
                       order: int = 4,
                       zerophase: bool = True
                       ) -> np.ndarray:
    """
    zero-phase Butterworth filter. Standard values:
    fs = 1 # per day
    cutoff = 0.2 # 2/365.25 # per day
    order = 4
    """
    if data.ndim > 1:
        data = np.ravel(data)

    if data.size <= 18:
        print(data.shape)

    b, a = scipy.signal.butter(order, cutoff, fs=fs, btype=ftype, analog=False, output='ba')

    if zerophase:
        y = scipy.signal.filtfilt(b, a, data, axis=0)
    else:
        y = scipy.signal.lfilter(b, a, data, axis=0)

    return y