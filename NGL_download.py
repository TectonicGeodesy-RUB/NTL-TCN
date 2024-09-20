# -------------------------------------------------------
# Copyright Kaan Cökerim (Ruhr-Universität Bochum)
# Mail: kaan.coekerim@rub.de
# Created on 13.01.2024
#
# function for scraping
#
# -------------------------------------------------------
import logging
import os
import pathlib
import requests
import io
import datetime
import numpy as np
import pandas as pd
import tqdm


def download_steps(savepath: str | pathlib.Path = './data/raws/steps.csv') -> None:
    """
    Download the step file from NGL
    :param savepath: path to save the file as csv
    :return: None
    """

    df = pd.read_csv('http://geodesy.unr.edu/NGLStationPages/steps.txt',
                     header=None, names=['STA', 'DATE', 'TYPE'],
                     usecols=[0, 2, 4], sep=' ')

    df['DATE'] = pd.to_datetime(df['DATE'], format='%y%b%d')
    df = df.drop_duplicates(subset=['STA', 'DATE'], keep='first')
    df.loc[:, ['ORD']] = df['DATE'].apply(lambda x: x.toordinal())
    df.loc[:, ['DFRAC']] = 0

    df.to_csv(savepath, index=False)


def get_consecutive_segments(time_arr, max_gap_length) -> list:
    return np.split(time_arr, np.where(np.diff(time_arr) > max_gap_length)[0] + 1)


def download_waveforms(
        savepath: str | pathlib.Path, csv_extention: str,
        minlon: float = None, maxlon: float = None, minlat: float = None, maxlat: float = None,
        starttime: str | datetime.datetime = '2002-01-01', endtime: str | datetime.datetime = '2024-02-01',
        chunk_length: int = None, min_chunk_length: int = 730,
        min_data_length: int = 6 * 365, degree_of_completeness: float = 0.85, max_gap_length: int = 60,
) -> None:
    """
    Download and save waveforms form NGL web server
    :param savepath: path to save waveforms file
    :param csv_extention: extention to save the station csv file
    :param minlon: minimum longitude as decimal degrees in East
    :param maxlon: maximum longitude as decimal degrees in East
    :param minlat: minimum latitude as decimal degrees in North
    :param maxlat: maximum latitude as decimal degrees in North
    :param min_data_length: minimum number of solutions in the time series
    :param max_gap_length:
    :param degree_of_completeness:
    :param starttime: start date as a `datetime.datetime` object or string in YYYY-MM-DD format
    :param endtime: end date as a `datetime.datetime` object or string in YYYY-MM format
    :param chunk_length: Time series will be divided into `length/chunk_size` amount of chunks with length
                         `chunk_length`. Keep in mind that this might produce time series with small lengths as the
                         remainder of `length/chunk_size`. If `None`, the time series will be not be divided.
    :param min_chunk_length: Minimum length of a chunk. Filters out small remainder time series with
                             fewer samples than `min_chunk_length`
    :return: None
    """
    # set logging
    logging.basicConfig(filename=f'download_{csv_extention}.log', level=logging.INFO, filemode='w')

    # download station list
    df_ngl = np.loadtxt('http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt',
                        dtype=str, usecols=np.arange(11))

    df_ngl = pd.DataFrame(df_ngl[1:], columns=df_ngl[0])
    df_ngl = df_ngl.astype({'Sta': str,
                            'Lat(deg)': float, 'Long(deg)': float, 'Hgt(m)': float,
                            'X(m)': float, 'Y(m)': float, 'Z(m)': float,
                            'NumSol': int})
    df_ngl['Dtbeg'] = pd.to_datetime(df_ngl['Dtbeg'], format='%Y-%m-%d')
    df_ngl['Dtend'] = pd.to_datetime(df_ngl['Dtend'], format='%Y-%m-%d')
    df_ngl['Dtmod'] = pd.to_datetime(df_ngl['Dtmod'], format='%Y-%m-%d')

    df_ngl.loc[:, ['TheoreticNumSol']] = (df_ngl['Dtend'] - df_ngl['Dtbeg']).dt.days

    df_ngl['Long(deg)'] = (df_ngl['Long(deg)'] - 360).where(cond=df_ngl['Long(deg)'] > 180, other=df_ngl['Long(deg)'])

    # set geographic bounds if all are given
    if all([minlon, maxlon, minlat, maxlat]):
        df_ngl = df_ngl.loc[(df_ngl['Long(deg)'] >= minlon) & (df_ngl['Long(deg)'] <= maxlon) &
                            (df_ngl['Lat(deg)'] >= minlat) & (df_ngl['Lat(deg)'] <= maxlat)]
    elif any([minlon, maxlon, minlat, maxlat]) and not all([minlon, maxlon, minlat]):
        raise Warning(f'At least one geographic boundary is not set. Will continue to download global data instead.\n'
                      f'\tReceived boundaries: {maxlon=}, {minlon=}, {maxlat=}, {minlat=}')

    # set start and end times
    if starttime:
        if isinstance(starttime, datetime.datetime):
            starttime = starttime.strftime('%Y-%m-%d')
        df_ngl = df_ngl.loc[(df_ngl['Dtend'] >= starttime)]
    if endtime:
        if isinstance(endtime, datetime.datetime):
            endtime = endtime.strftime('%Y-%m-%d')
        df_ngl = df_ngl.loc[(df_ngl['Dtbeg'] <= endtime)]
    if min_data_length:
        df_ngl = df_ngl.loc[(df_ngl['NumSol'] >= min_data_length)]

    # prepare download folder and base url
    # base_url_ngl = 'http://geodesy.unr.edu/gps_timeseries/tenv3_loadpredictions'
    base_url_ngl = 'http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14'

    os.makedirs(savepath, exist_ok=True)

    for sta in tqdm.tqdm(df_ngl.Sta.values):

        base_file_suffix = 0
        if os.path.exists(os.path.join(savepath, f'{sta}.csv')):
            continue

        # get file from url
        try:
            file_url = os.path.join(base_url_ngl, sta + '.tenv3')
            response = requests.get(file_url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            logging.info(f'{sta} has fetching issues - {err}')
            df_ngl = df_ngl.drop(index=df_ngl.index[df_ngl.Sta == sta])
            df_ngl = df_ngl.reset_index(drop=True)
            continue

        data = np.loadtxt(io.BytesIO(response.content), dtype=str)

        if not len(data[1:, 1]):
            df_ngl = df_ngl.drop(index=df_ngl.index[df_ngl.Sta == sta])
            df_ngl = df_ngl.reset_index(drop=True)
            logging.info(f'{sta} returned empty sequence. Removing station.')
            continue

        # get data TODO: should really be parallelized
        time_series = pd.DataFrame(
            index=np.array([datetime.datetime.strptime(d, '%y%b%d') for d in data[1:, 1]]),
            data=dict(
                # t=np.array([datetime.datetime.strptime(d, '%y%b%d') for d in data[1:, 1]]),
                e=np.array(data[1:, 7]).astype(float) + np.array(data[1:, 8]).astype(float),
                n=np.array(data[1:, 9]).astype(float) + np.array(data[1:, 10]).astype(float),
                u=np.array(data[1:, 11]).astype(float) + np.array(data[1:, 12]).astype(float),

            )
        )

        time_series.index = pd.to_datetime(time_series.index)

        time_series = time_series.loc[(time_series.index >= starttime) & (time_series.index < endtime)]

        # check gap fraction
        # check gap length
        t = np.array([datetime.datetime.strptime(d, '%y%b%d').toordinal() for d in data[1:, 1]])
        t = t[(t >= datetime.datetime.strptime(starttime, '%Y-%m-%d').toordinal()) &
              (t < datetime.datetime.strptime(endtime, '%Y-%m-%d').toordinal())]

        if not len(t):
            df_ngl = df_ngl.drop(index=df_ngl.index[df_ngl.Sta == sta])
            df_ngl = df_ngl.reset_index(drop=True)
            logging.info(f'{sta} returned empty sequence. Removing station.')
            continue
        try:
            deltas = np.diff(t)
            total_gaps = deltas[deltas > 1].sum()
            gap_fraction = total_gaps / abs(t.max() - t.min())
        except ValueError as val_err:
            df_ngl = df_ngl.drop(index=df_ngl.index[df_ngl.Sta == sta])
            df_ngl = df_ngl.reset_index(drop=True)
            logging.info(f'{sta} returned empty sequence. Removing station. - {val_err}')
            continue

        if len(t) < min_data_length or gap_fraction > (1 - degree_of_completeness):
            logging.info(f'{sta} has not enough data or too many gaps: '
                         f'{len(t)=:>4} {min_data_length=:>4} | {gap_fraction=:.3f}  {degree_of_completeness=:.3f} | '
                         f'{max_gap_length}')
            df_ngl = df_ngl.drop(index=df_ngl.index[df_ngl.Sta == sta])
            df_ngl = df_ngl.reset_index(drop=True)
            continue

        elif any(deltas > max_gap_length):
            logging.info(f'{sta} has gaps larger than tolerance threshold: {max_gap_length}:\n\t'
                         f'Trying to find segments that satisfy `min_data_length`...')
            num_usable_segments = 0
            # find non-nan segments that are >= min_data_length and have gaps < max_gap_length
            segments = get_consecutive_segments(time_arr=t, max_gap_length=max_gap_length)
            for segment in segments:
                if len(segment) >= min_data_length:
                    # save usable segment
                    select_idx_dt = list(map(datetime.datetime.fromordinal, segment))
                    select_idx = [dt.strftime(format='%Y-%m-%d') for dt in select_idx_dt]

                    segment_series = time_series.loc[select_idx, :]

                    seg_row = {
                        'Sta': f'{sta}_{base_file_suffix}',
                        'Lat(deg)': df_ngl.loc[df_ngl.Sta == sta, 'Lat(deg)'].values[0],
                        'Long(deg)': df_ngl.loc[df_ngl.Sta == sta, 'Long(deg)'].values[0],
                        'Hgt(m)': df_ngl.loc[df_ngl.Sta == sta, 'Hgt(m)'].values[0],
                        'X(m)': df_ngl.loc[df_ngl.Sta == sta, 'X(m)'].values[0],
                        'Y(m)': df_ngl.loc[df_ngl.Sta == sta, 'Y(m)'].values[0],
                        'Z(m)': df_ngl.loc[df_ngl.Sta == sta, 'Z(m)'].values[0],
                        'Dtbeg': pd.to_datetime(select_idx_dt[0]),
                        'Dtend': pd.to_datetime(select_idx_dt[-1]),
                        'Dtmod': pd.to_datetime(datetime.datetime.now()),
                        'NumSol': segment_series['e'].shape[0],
                        'TheoreticNumSol': (segment_series.index[-1] - segment_series.index[0]).days
                               }
                    df_ngl.loc[df_ngl.shape[0]] = seg_row

                    # save as csv (high save time BUT low enough file size and low memory growth during loading)
                    st = select_idx_dt[0].strftime(format='%Y%m%d')
                    et = select_idx_dt[-1].strftime(format='%Y%m%d')
                    segment_series.to_csv(os.path.join(savepath, f'{sta}_{base_file_suffix}__{st}__{et}.csv'))
                    base_file_suffix += 1
                    num_usable_segments += 1

            if num_usable_segments == 0:
                df_ngl = df_ngl.drop(index=df_ngl.index[df_ngl.Sta == sta])
                df_ngl = df_ngl.reset_index(drop=True)
                logging.info(f'\t{sta} has no usable segments. Skipping and removing station')
                continue
            else:
                df_ngl = df_ngl.drop(index=df_ngl.index[df_ngl.Sta == sta])
                df_ngl = df_ngl.reset_index(drop=True)
                logging.info(f'\t{sta} has {num_usable_segments} usable segments.')

        else:  # i.e. everything is good and all thresholds satisfied (NOTE: TIME SERIES CAN HAVE GAPS < max_gap_length)
            if time_series['e'].count() < min_data_length:
                raise AttributeError(f'{sta} - MADE ERROR WITH CONDITIONS')
            # update sample count
            df_ngl.loc[df_ngl.Sta == sta, 'Sta'] = f'{sta}_{base_file_suffix}'
            df_ngl.loc[df_ngl.Sta == sta, 'NumSol'] = time_series['e'].shape[0]
            df_ngl.loc[df_ngl.Sta == sta, 'TheoreticNumSol'] = (time_series.index[-1] - time_series.index[0]).days
            df_ngl.loc[df_ngl.Sta == sta, 'Dtbeg'] = time_series.index[0]
            df_ngl.loc[df_ngl.Sta == sta, 'Dtend'] = time_series.index[-1]

            # save as csv (high save time BUT low enough file size and low memory growth during loading)
            st = time_series.index[0].strftime(format='%Y%m%d')
            et = time_series.index[-1].strftime(format='%Y%m%d')
            time_series.to_csv(os.path.join(savepath, f'{sta}_{base_file_suffix}__{st}__{et}.csv'))

    # remove download artifacts
    # save data table
    if isinstance(savepath, str):
        savepath = pathlib.Path(savepath)
    df_ngl.to_csv(savepath.parent.joinpath(f'stations_{csv_extention}.csv'), index=False)
    logging.info(f'Number of Stations: {df_ngl.loc[df_ngl.NumSol > min_data_length].shape[0]}\n-----\n')


if __name__ == '__main__':
    import argparse

    cl = argparse.ArgumentParser()
    cl.add_argument("-n", "--n_processes", type=int, default=10,
                    help="Number of processes for parallelization")
    cl.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction)
    cl.add_argument("--get_steps", default=False, action=argparse.BooleanOptionalAction)
    cl.add_argument("--gnss", default=False, action=argparse.BooleanOptionalAction)
    cl.add_argument("--train", default=False, action=argparse.BooleanOptionalAction)
    cl.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    cl.add_argument("--val", default=False, action=argparse.BooleanOptionalAction)

    args = cl.parse_args()
    print(f'{args.gnss=} | {args.train=} | {args.val=} | {args.test=} | {args.get_steps}')

    # get step file
    if args.get_steps:
        print('Getting steps')
        download_steps()

    if args.gnss:
        # print('Getting all gnss')
        # download_waveforms(savepath=r'./data/raws/fullrange_enhanced', csv_extention='train',
        #                    starttime='2002-01-01', endtime='2024-02-01', min_data_length=6 * 365)

        # get train waveforms
        if args.train:
            print('Getting train gnss')
            download_waveforms(savepath=r'./data/raws/train', csv_extention='train',
                               starttime='2002-01-01', endtime='2017-01-01', min_data_length=6 * 365)

        # get validation waveforms
        if args.val:
            print('Getting val gnss')
            download_waveforms(savepath=r'./data/raws/validation', csv_extention='val',
                               starttime='2017-01-01', endtime='2021-01-01', min_data_length=2 * 365)

        # get test waveforms
        if args.test:
            print('Getting test gnss')
            download_waveforms(savepath=r'./data/raws/test_long', csv_extention='test_long',
                               starttime='2021-01-01', endtime='2024-06-01', min_data_length=3 * 365)
