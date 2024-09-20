# -------------------------------------------------------
# Copyright Kaan Cökerim (Ruhr-Universität Bochum)
# Mail: kaan.coekerim@rub.de
# Created on 13.01.2024
#
# tools to fit and remove steps, trends from GNSS time series
#
# -------------------------------------------------------
import os
import json
import multiprocessing
from multiprocessing_logging import install_mp_handler
import logging
import pathlib
import datetime
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from hampel import hampel
import time
from typing import Union
import sys

sys.path.append('/home/kaan/git/Gratsid')
from gratsid import gratsid_fit, fit_decompose

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# num_thread = "1"
# os.environ["OMP_NUM_THREADS"] = num_thread
# os.environ["OPENBLAS_NUM_THREADS"] = num_thread
# os.environ["MKL_NUM_THREADS"] = num_thread
# os.environ["VECLIB_MAXIMUM_THREADS"] = num_thread
# os.environ["NUMEXPR_NUM_THREADS"] = num_thread
# os.environ["NUMEXPR_MAX_THREADS"] = num_thread
# os.environ['KMP_WARNINGS'] = '0'


def read_gnss(fpath: Union[str, pathlib.Path], starttime=None, endtime=None,
              component: Union[str, list] = 'u', format='gratsid') -> (np.ndarray, np.ndarray):
    """function to read in the GNSS csv-files"""

    gnss = pd.read_csv(fpath)
    gnss = gnss.rename(columns={gnss.columns[0]: 'times'})
    gnss['times'] = pd.to_datetime(gnss['times'])

    if starttime:
        gnss = gnss.loc[gnss['times'] >= starttime]
    if endtime:
        gnss = gnss.loc[gnss['times'] <= endtime]

    if format == 'gratsid':
        component = [component] if not isinstance(component, list) else component
        return (gnss['times'].copy().apply(lambda x: x.toordinal()).to_numpy(),
                gnss[component].copy().to_numpy())
    else:
        return (gnss['times'].copy().apply(lambda x: x.toordinal()).to_numpy(),
                gnss[component].copy().to_numpy())


def decomposition(
        gnss_file: Union[str, pathlib.Path],
        output_folder: Union[str, pathlib.Path],
        starttime: Union[str, datetime.datetime, None], endtime: Union[str, datetime.datetime, None],
        param: dict, known_steps: pd.DataFrame,
) -> None:
    """
    Decompose a GNSS time series
    :param gnss_file:
    :param output_folder:
    :param starttime:
    :param endtime:
    :param param:
    :param known_steps:
    :return:
    """

    start_timer = time.time()
    curr_proc = multiprocessing.current_process().name
    logging.info(f'+\t[{curr_proc} - {datetime.datetime.now()}] Working on {gnss_file.stem}')

    if starttime:
        if isinstance(starttime, str):
            starttime = datetime.datetime.strptime(starttime, "%Y-%m-%d")
        starttime = starttime.toordinal()

    if endtime:
        if isinstance(endtime, str):
            endtime = datetime.datetime.strptime(endtime, "%Y-%m-%d")
        endtime = endtime.toordinal()

    # read gnss
    t, data = read_gnss(gnss_file, starttime, endtime, format=None)
    # print(gnss_file.stem, t.shape, data.shape)

    # data = data[:, -1]
    data -= np.nanmedian(data)
    station = gnss_file.stem.split('_')[0]

    # check gap fraction TODO: not needed since all gap and partinioning task are done on download. Just fill gaps
    # deltas = np.diff(t)
    # total_gaps = deltas[deltas > 1].sum()
    # gap_fraction = total_gaps / abs(t.max() - t.min())

    # if len(t) < min_data_length or gap_fraction > (1 - degree_of_completeness):
    #     logging.info(f'Station has not enough data or too many gaps: '
    #                  f'{len(t)=} {min_data_length=} | {gap_fraction=:.3f}  {degree_of_completeness=:.3f} | '
    #                  f'{max_gap_length}')
    #     return
    # elif any(deltas > max_gap_length):
    #     logging.info(f'Station has gaps larger than tolerance threshold: {max_gap_length}')
    #     return

    # Hampel filter to remove outliers
    data = hampel(data=data, window_size=10, n_sigma=3.).filtered_data
    # data = data[:, None]

    # get steps
    known_steps_sta = known_steps.loc[(known_steps.STA == station) &
                                      (known_steps.ORD >= t[0]) &
                                      (known_steps.ORD <= t[-1])]
    if known_steps_sta.empty:
        known_steps_ = []
    else:
        known_steps_sta = known_steps_sta.reset_index(drop=True)  # reset dataframe index to avoid issues in loops

        known_steps_ = known_steps_sta.loc[:, ['ORD', 'DFRAC']].values

    # fit decompose x,y,err,known_steps,options
    # t = t[: None]
    logging.info(f'+\t[{curr_proc} - {datetime.datetime.now()}] Starting fitting for {gnss_file.stem}')
    try:
        perm, sols, options_out = gratsid_fit(x=t, y=data[:, None], err=None, known_steps=known_steps_, options=param)
    except Exception as err:
        logging.info(f'+\t[{curr_proc} - {datetime.datetime.now()}] - {err}')
        raise ValueError('SOMETHING WEIRD HAPPENED')

    np.savez(file=output_folder.joinpath(f'{gnss_file.stem}.npz'),
             time=t,
             raw=data,
             perm=perm,
             sols=np.array(sols, dtype=object),
             options=options_out,
             data_cols=['u'],
             tbounds=[t[0], t[-1]],
             allow_pickle=True)

    logging.info(
        f'+\t[{curr_proc} - {datetime.datetime.now()}] Successfully fitted {gnss_file.stem} - {(t[-1] - t[0]) / 365.25:>5.2f} y of Data'
        f' - Ellapsed: {datetime.timedelta(seconds=time.time() - start_timer)}'
    )
    return


def decomposition_parallel(
        gnss_folder: Union[str, pathlib.Path],
        output_folder: Union[str, pathlib.Path],
        starttime: Union[str, datetime.datetime, None], endtime: Union[str, datetime.datetime, None],
        param_file: Union[str, pathlib.Path] = pathlib.Path('./data/solutions/gratsid_options.json'),
        known_steps: Union[str, pathlib.Path] = pathlib.Path('./data/raws/steps.csv'),
        n_processes: int = 10,
) -> None:
    """
    Decompose GNSS signals in parallel
    :param gnss_folder:
    :param output_folder:
    :param starttime:
    :param endtime:
    :param param_file:
    :param known_steps:
    :param n_processes:
    :return:
    """

    gnss_folder = pathlib.Path(gnss_folder) if isinstance(gnss_folder, str) else gnss_folder
    output_folder = pathlib.Path(output_folder) if isinstance(output_folder, str) else output_folder
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True, exist_ok=True)

    # collect files
    gnss_files = list(sorted(list(gnss_folder.glob('*.csv'))))

    # load parameter file
    with open(param_file, 'r') as fopen:
        param = json.load(fopen)

    # load steps
    steps_df = pd.read_csv(known_steps)

    # assemble proc_args
    proc_args = [
        (gnss_files[i], output_folder,
         starttime, endtime, param.copy(), steps_df.copy())
        for i in range(len(gnss_files[:100]))
    ]

    logging.basicConfig(filename='decomp.log', level=logging.INFO, filemode='w')
    install_mp_handler()
    logging.info(f'+ [{datetime.datetime.now()}] STARTING POOL with {n_processes} for {len(gnss_files)} files')

    # import dask
    # client = dask.distributed.Client(threads_per_worker=4, n_workers=n_processes)
    # res = client.map(decomposition, proc_args)

    # for i in range(len(gnss_files):

    with multiprocessing.Pool(processes=n_processes) as pool:
        # res = [pool.appy_async(decomposition, proc_args[i]) for i in range(len(proc_args))]
        pool.starmap(decomposition, proc_args)

    logging.info(f'+ [{datetime.datetime.now()}] POOL CLOSED')


def decomposition_array(
        gnss_folder: Union[str, pathlib.Path],
        output_folder: Union[str, pathlib.Path],
        job_id: int,
        starttime: Union[str, datetime.datetime, None],
        endtime: Union[str, datetime.datetime, None],
        param_file: Union[str, pathlib.Path] = pathlib.Path('./data/solutions/gratsid_options.json'),
        known_steps: Union[str, pathlib.Path] = pathlib.Path('./data/raws/steps.csv'),
        overwrite: bool = False
):
    gnss_folder = pathlib.Path(gnss_folder) if isinstance(gnss_folder, str) else gnss_folder
    output_folder = pathlib.Path(output_folder) if isinstance(output_folder, str) else output_folder
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True, exist_ok=True)

    # collect files
    gnss_files = list(sorted(list(gnss_folder.glob('*.csv'))))

    logging.basicConfig(filename=f'decomp_test_logs/decomp_{job_id}.log', level=logging.INFO, filemode='w')

    selected_file = gnss_files[job_id]
    station_sol_file = output_folder.joinpath(selected_file.stem + '.npz')
    if station_sol_file.is_file() and not overwrite:
        logging.info(f'+\t[{datetime.datetime.now()}] - {station_sol_file} already exists. Skipping File....')
        return

    # load parameter file
    with open(param_file, 'r') as fopen:
        param = json.load(fopen)

    # load steps
    steps_df = pd.read_csv(known_steps)

    decomposition(gnss_file=selected_file, output_folder=output_folder,
                  starttime=starttime, endtime=endtime, param=param, known_steps=steps_df.copy())


def get_signal(signal: np.ndarray, signal_ids: list) -> np.ndarray:
    sols = np.array(signal)
    sols = sols.reshape(sols.shape[:-1])
    sols = sols[signal_ids]

    sols = sols.sum(axis=0)
    sols = np.nanmedian(sols, axis=0)
    return sols


def get_mean_tect_vel(signal: np.ndarray) -> np.ndarray:
    sols = np.array(signal)
    sols = sols.reshape(sols.shape[:-1])
    sols = sols[[1, 3, 4]]

    sols = sols.sum(axis=0)
    sols_diff = np.diff(sols, axis=-1)
    diff_mean = np.nanmedian(sols_diff, axis=0)
    mean_vel = np.nanmedian(sols)

    return mean_vel


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def impute_gaps(t, data, kind='linear') -> (np.ndarray, np.ndarray):
    """
    Impute gaps in the time series and add gaussian noise scaled by the standard deviation of the data
    :param t:
    :param data:
    :param kind:
    :return:
    """

    # flatten data arrays
    if len(data.shape) > 1:
        data = data.reshape(-1)

    # fill time array and extend data array to contain nans at missing time stamps
    t_arr = np.arange(t[0], t[-1] + 1)
    data_arr = np.array([data[t == val][0] if val in t else np.nan for idx, val in enumerate(t_arr)])

    # get nans
    nans, x = nan_helper(data_arr)
    # get characteristic noise
    noise = np.random.randn(data_arr[nans].size) * (np.nanstd(data_arr[~nans]) / 3)

    f = scipy.interpolate.interp1d(x(~nans), data_arr[~nans], kind=kind)  # create interpolator
    data_arr[nans] = f(x(nans))  # interpolate over nans
    data_arr[nans] += noise  # add noise
    # apply Hampel filter to remove outliers from interpolation
    data_arr = hampel(data=data_arr, window_size=15, n_sigma=2.).filtered_data

    return data_arr, t_arr


def get_target_series(
        solution_file: Union[str, pathlib.Path],
        output_folder: Union[str, pathlib.Path]
) -> None:
    # perm, sols, options = read_gratsid(solution_file)

    solution_file = pathlib.Path(solution_file) if isinstance(solution_file, str) else solution_file
    output_folder = pathlib.Path(output_folder) if isinstance(output_folder, str) else output_folder
    station = solution_file.stem[:6]  # gnss_file.stem
    if output_folder.joinpath(f'{station}_decomposition.npz').exists():
        return

    with np.load(solution_file, allow_pickle=True) as gratsid_in:
        perm = gratsid_in['perm']
        sols = gratsid_in['sols']

        options = gratsid_in['options'].item()

        t = gratsid_in['time']
        data = gratsid_in['raw']

    try:
        decomp_sig = fit_decompose(x=t, y=data[:, None], err=None, sols=sols, perm_table=perm, options=options)
    except AttributeError as att_err:
        print(f'[{datetime.datetime.now()}] Error in {station} - {att_err}')
        return

    seasonal = get_signal(decomp_sig, [2])
    residuals = get_signal(decomp_sig, [5])
    tectonics = get_signal(decomp_sig, [1, 3, 4])
    steps = get_signal(decomp_sig, [0])
    y = get_signal(decomp_sig, [2, 5])  # data - tectonics

    # get mean tect velocity
    mean_tect_vel = get_mean_tect_vel(decomp_sig)

    # impute signals# impute signals
    tectonics, _ = impute_gaps(t=t, data=tectonics, kind='slinear')
    seasonal, _ = impute_gaps(t=t, data=seasonal, kind='quadratic')
    residuals, _ = impute_gaps(t=t, data=residuals, kind='slinear')
    steps, _ = impute_gaps(t=t, data=steps, kind='nearest')
    y, t = impute_gaps(t=t, data=y, kind='quadratic')

    # save
    np.savez(file=output_folder.joinpath(f'{station}_decomposition.npz'),
             seasonal=seasonal,
             residuals=residuals,
             tectonics=tectonics - mean_tect_vel,
             steps=steps,
             mean_tect_vel=np.array([mean_tect_vel]),
             data=y,
             time=t)


if __name__ == '__main__':
    import argparse

    cl = argparse.ArgumentParser()
    cl.add_argument("-n", "--n_processes", type=int, default=10,
                    help="Number of processes for parallelization")
    cl.add_argument("-d", "--dataset", type=str, default='test')
    cl.add_argument("-p", "--parallel_job", default=False, action=argparse.BooleanOptionalAction)
    cl.add_argument("-a", "--array_job", default=False, action=argparse.BooleanOptionalAction)
    cl.add_argument("-g", "--get_signals", default=False, action=argparse.BooleanOptionalAction)
    cl.add_argument("-j", "--jobID", type=int, default=None)
    cl.add_argument("-o", "--overwrite", default=False, action=argparse.BooleanOptionalAction)

    args = cl.parse_args()

    assert args.parallel_job or args.array_job or args.get_signals

    if isinstance(args.n_processes, str):
        args.n_processes = int(args.n_processes)

    if args.parallel_job:
        decomposition_parallel(gnss_folder=pathlib.Path(f'data/raws/{args.dataset}'),
                               output_folder=pathlib.Path(f'data/solutions/{args.dataset}'),
                               starttime=None, endtime=None,
                               param_file=pathlib.Path('./data/solutions/gratsid_options.json'),
                               known_steps=pathlib.Path('./data/raws/steps.csv'),
                               n_processes=args.n_processes)
    elif args.array_job:
        assert args.jobID
        decomposition_array(gnss_folder=pathlib.Path(f'data/raws/{args.dataset}'),
                            output_folder=pathlib.Path(f'data/solutions/{args.dataset}'),
                            starttime=None, endtime=None,
                            param_file=pathlib.Path('./data/solutions/gratsid_options.json'),
                            known_steps=pathlib.Path('./data/raws/steps.csv'),
                            job_id=int(args.jobID) - 1,
                            overwrite=args.overwrite)

    elif args.get_signals:
        sol_folder = pathlib.Path(f'data/solutions/{args.dataset}')
        sol_files = list(sorted(list(sol_folder.glob('*.npz'))))

        for sol_file in tqdm(sol_files):
            get_target_series(solution_file=sol_file,
                              output_folder=pathlib.Path(f'data/targets/{args.dataset}'))  # '_new'
