# -----------------------------------------------
# Created by Kaan Cökerim on 09.02.24
# Email: kaan.coekerim@rub.de
#
# Description: script to handle everything related to the generation of the surface loading time series
# -----------------------------------------------
import argparse
import multiprocessing
import pathlib
import numpy as np
import pandas as pd


def agglomerate_stations(combined_stations: str | pathlib.Path | pd.DataFrame,
                         batch_size: int, out_folder: str | pathlib.Path) -> None:
    """
    Reads and builds an aggregated station list for ESMGFZ loading series bash script and saves it in specified batches

    Output station list has format: (no header, UNIX format) STATION_ID LONGITUDE LATITUDE
    Args:
        out_folder: output folder where files will be stored
        batch_size: batch size to divide the station file to allow for parallel processing later on
        combined_stations: path to combined station list of Dataframe object

    Returns: None

    """

    if isinstance(combined_stations, str) or isinstance(combined_stations, pathlib.Path):
        combined_stations = pd.read_csv(combined_stations)

    if isinstance(out_folder, str):
        out_folder = pathlib.Path(out_folder)

    # Drop ending of station names, i.e. go back to 4-char code
    combined_stations['Sta'] = combined_stations.loc[:, 'Sta'].str.slice(stop=4)

    # Drop duplicates
    combined_stations = combined_stations.drop_duplicates(subset=["Sta"], keep='first')

    # reduce to necessary columns
    combined_stations = combined_stations.loc[:, ['Sta', 'Lat(deg)', 'Long(deg)']]
    combined_stations.to_csv(out_folder.joinpath('station_list.csv'), index=False)

    # Split the dataframe into batches and save
    for i_batch, batch_start in enumerate(range(0, len(combined_stations), batch_size)):
        if batch_start + batch_size < len(combined_stations):
            batch = combined_stations.iloc[batch_start:batch_start + batch_size, :]
        else:
            batch = combined_stations.iloc[batch_start:, :]
        batch.to_csv(out_folder.joinpath(f'station_list_{i_batch}.csv'), index=False, header=False, sep=' ')


def load_bash_loading(station_file: str | pathlib.Path, resample=None) -> pd.DataFrame:
    """

    Args:
        station_file:
        resample:

    Returns:

    """
    date = np.loadtxt(station_file, dtype=str, usecols=0, skiprows=4)
    time = np.loadtxt(station_file, dtype=str, usecols=1, skiprows=4)
    values = np.loadtxt(station_file, dtype=float, usecols=2, skiprows=4)

    date_time = np.array([f'{d} {t}' for d, t in zip(date, time)])
    # date_time = pd.to_datetime(date_time, format='%Y-%m-%d %H:%M')

    TimeSeries = pd.DataFrame(index=date_time, data=dict(vals=values))
    TimeSeries.index = pd.to_datetime(date_time, format='%Y-%m-%d %H:%M')

    if resample:
        TimeSeries = TimeSeries.resample(resample).mean()

    return TimeSeries


def merge_loading_series(station_id: str, loading_bash_path: str | pathlib.Path, out_path: str | pathlib.Path) -> None:
    """
    reads in the individual loading series of station with name ´station_id´ and merges them into one npz file

    Args:
        loading_bash_path:
        out_path:
        station_id: 4-character code of station

    Returns: None

    """

    loading_bash_path = pathlib.Path(loading_bash_path) if isinstance(loading_bash_path, str) else loading_bash_path
    out_path = pathlib.Path(out_path) if isinstance(out_path, str) else out_path

    station_files = [loading_bash_path.joinpath(f'{prod}/{prod}.{station_id}.txt') for prod in ['HYDL', 'NTOL', 'NTAL']]

    # load all three loading time series for station
    hydl_ts = load_bash_loading(station_files[0])  # idx: 0 = pd.datetime_index; 1 = values
    ntol_ts = load_bash_loading(station_files[1], resample='D')
    ntal_ts = load_bash_loading(station_files[2], resample='D')

    np.savez(out_path.joinpath(f'{station_id}.npz'),
             hydl=hydl_ts.vals.to_numpy(),
             ntol=ntol_ts.vals.to_numpy(),
             ntal=ntal_ts.vals.to_numpy(),
             time=np.array([d.toordinal() for d in hydl_ts.index]))


def parallel_merge_loading(station_csv: str | pathlib.Path,
                           out_path: str | pathlib.Path,
                           loading_bash_path: str | pathlib.Path,
                           n_proc: int = 10) -> None:
    station_df = pd.read_csv(station_csv)

    func_args = [(station, loading_bash_path, out_path) for station in station_df.Sta.values]

    with multiprocessing.Pool(n_proc) as pool:
        result = pool.starmap(merge_loading_series, func_args)


if __name__ == '__main__':

    cl = argparse.ArgumentParser()
    cl.add_argument('--station_csv', type=str,
                    help='csv containing name, lat, lon ofall stations over all datasets')
    cl.add_argument('--out_path', type=str, help='output folder for merged NTL time series')
    cl.add_argument('--lbp', type=str, help='folder containing the temporary NTL time series')

    cl.add_argument('--n_proc', type=int, default=10, help='number of cores for parallel processing')
    args = cl.parse_args()

    parallel_merge_loading(
        station_csv=args.station_csv,
        out_path=args.out_path,
        loading_bash_path=args.lbp,
        n_proc=args.n_proc
    )

    # parallel_merge_loading(
    #     station_csv='/home/kaan/git/GlobalGNNMod/data/ESMGFZ/batch_stations/station_list.csv',
    #     out_path='/home/kaan/git/GlobalGNNMod/data/ESMGFZ/loading_series',
    #     loading_bash_path='/home/kaan/git/GlobalGNNMod/data/ESMGFZ/temp_loading_series',
    #     n_proc=args.n_proc
    # )
