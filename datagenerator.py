# -------------------------------------------------------
# Copyright Kaan Cökerim (Ruhr-Universität Bochum)
# Mail: kaan.coekerim@rub.de
# Created on 13.01.2024
#
# generator to create feature and target time series sets
#
# -------------------------------------------------------

import pathlib
import preprocessing_utils
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import datetime


class EncapsulatedTimeSeriesDataset(Dataset):
    def __init__(self,
                 station_df: str | pathlib.Path | pd.DataFrame,
                 features: list[str] | tuple[str],
                 target: str,
                 feature_dir: str | pathlib.Path,
                 target_dir: str | pathlib.Path,
                 gnss_feature_dir: str | pathlib.Path = None,
                 feature_length: int = 365, feature_lookahead: int = 182, feature_stride: int = 1,
                 target_length: int = 1, target_stride: int = 1,
                 offset: int = 1, lowpassfilter: float = False, ntl_lowpassfilter: float = False,
                 mm_to_m: bool = True, diff: bool = False, norm: bool = True,
                 norm_stats: str | pathlib.Path | pd.DataFrame = '/home/kaan/git/GlobalGNNMod/data/features/norm_stats.csv',
                 eval_mode: bool = False
                 ) -> None:

        self.station_database = station_df if isinstance(station_df, pd.DataFrame) else pd.read_csv(station_df)
        self.target = target.lower()
        self.features = [f.lower() for f in features]
        self.feature_dir = feature_dir if isinstance(feature_dir, pathlib.Path) else pathlib.Path(feature_dir)
        self.target_dir = target_dir if isinstance(target_dir, pathlib.Path) else pathlib.Path(target_dir)
        self.gnss_feature_dir = gnss_feature_dir
        self.feature_length = feature_length
        self.feature_lookahead = feature_lookahead
        self.feature_lookback = feature_length - feature_lookahead
        self.feature_stride = feature_stride
        self.target_length = target_length
        self.target_stride = target_stride
        self.offset = offset
        self.eval_mode = eval_mode

        self.diff = diff
        self.lowpassfilter = lowpassfilter
        self.ntl_lowpassfilter = ntl_lowpassfilter
        self.mm_to_m = mm_to_m
        self.norm = norm
        if self.norm:
            self.norm_stats = norm_stats if isinstance(norm_stats, pd.DataFrame) else pd.read_csv(norm_stats,
                                                                                                  index_col=0)
            self.harmonic_std = 2 / (2 * np.sqrt(2))
        else:
            self.norm_stats = None

        # ------------------------
        # TODO: add Exception check before assertions
        possib_load_feats = ['hydl', 'ntol', 'ntal']
        possib_gnss_feats = possib_tar = ['data', 'residuals', 'seasonal', 'tectonics', 'full']
        possible_feat = possib_load_feats + possib_gnss_feats

        assert self.station_database.shape[0] > 0 and len(self.features) > 0
        assert all([f in possible_feat for f in self.features])
        assert self.target in possib_tar

        if not self.eval_mode:
            assert self.target_dir.exists() and self.feature_dir.exists()
        else:
            assert self.feature_dir.exists()

        self.loading_feats = [f for f in self.features if f in possib_load_feats]
        self.gnss_feats = [f for f in self.features if f in possib_gnss_feats]

        # ------------------------
        self.num_stations = self.station_database.shape[0]
        self.num_target_samples = self.station_database.TheoreticNumSol.sum()
        self.target_start_idx = (self.feature_stride * (self.feature_lookback - 1))
        self.target_stop_idx = (self.feature_stride * (self.feature_lookahead - 1)) + 1
        # get number of usable samples per station
        self.station_database['UseNumSol'] = (self.station_database['TheoreticNumSol'] -
                                              (self.feature_length * self.feature_stride) + self.feature_stride)
        # get cumulative running sum over usable station samples given the order of the database table
        self.station_database['CumSumSol'] = self.station_database['UseNumSol'].cumsum()

    def __len__(self) -> int:
        # return self.num_target_samples - (self.num_stations * self.target_start_idx)
        return self.station_database['UseNumSol'].sum()

    def __getitem__(self, idx: int | torch.Tensor) -> (np.ndarray, np.ndarray, np.ndarray, int):
        # make idx numpy
        if isinstance(idx, torch.Tensor):
            idx = idx.numpy()

        # select sample
        sample = self.get_samples(idx)  # get sample at idx where sample_idx = [station, target_idx, feature_idx]

        # load data and get label
        X = self.load_features(station=sample[0], idx=sample[2])
        y, gnss_feats, y_times = self.load_targets(station=sample[0], idx=sample[1], idx_feat=sample[4])


        aux = sample[3]

        # add gnss feats to X
        if self.gnss_feats and X.size:
            X = np.vstack((X, gnss_feats))  # equivalent to np.concatenate((X, gnss_feat), dim=0)
        elif self.gnss_feats and not X.size:
            X = gnss_feats

        X, y, aux = (torch.from_numpy(X).to(torch.float32),
                     torch.from_numpy(y).to(torch.float32),
                     torch.from_numpy(aux).to(torch.float32))

        return X, aux, y, y_times

    def get_samples(self, idx: int | torch.Tensor) -> list:
        row_idx = np.min(np.where((self.station_database['CumSumSol'].to_numpy() - idx) >= 0)[0])
        col_idx = self.station_database.loc[row_idx, 'UseNumSol'] - (
                self.station_database.loc[row_idx, 'CumSumSol'] - idx)

        station_name, lat, lon = self.station_database.loc[row_idx, ['Sta', 'Lat(deg)', 'Long(deg)']]
        assert isinstance(lat, float) and isinstance(lon, float)
        starttime = datetime.datetime.strptime(self.station_database.loc[row_idx, 'Dtbeg'], '%Y-%m-%d')
        starttime = starttime.toordinal()

        target_start_idx = col_idx + self.target_start_idx - 1
        target_stop_idx = target_start_idx + self.target_length * self.target_stride

        feature_first_idx = target_start_idx + self.feature_stride - self.feature_lookback * self.feature_stride
        feature_last_idx = target_start_idx + self.feature_stride + self.feature_lookahead * self.feature_stride

        # get feature indices
        feature_samples_ = np.arange(start=feature_first_idx, stop=feature_last_idx, step=self.feature_stride)
        # convert to ordinal by summing with the ordinal of the start date of GNNS signal
        feature_samples = feature_samples_ + starttime
        # convert to loading indices by subtraction ordinal of start date of loading time series
        feature_samples -= datetime.date(year=2000, month=1, day=1).toordinal()

        # get target indices
        target_samples = np.arange(start=target_start_idx, stop=target_stop_idx, step=self.target_stride)

        # get aux features
        aux_features = np.array([*self.temporal_encoding(t_in=starttime + col_idx),
                                 *self.spatial_encoding(lat=lat, lon=lon)])

        return [station_name, target_samples, feature_samples, aux_features[:, None], feature_samples_]

    def load_features(self, station: str, idx: list[int] | tuple[int] | np.ndarray[int]) -> np.ndarray:
        """
        Load features from disk
        Args:
            station: station code WITHOUT segment specifier
            idx: global sample index

        Returns: numpy array of sequential feature with shape [batch_size, num_features, feature_length]

        """
        # TODO: first diff then norm

        file_name = self.feature_dir.joinpath(f'{station[:4]}.npz')
        with np.load(file=file_name, mmap_mode='r') as feat_io:
            if not self.ntl_lowpassfilter:
                feats = np.array([feat_io[feat][idx] for feat in self.loading_feats])
                feats -= np.nanmedian(feats, axis=1, keepdims=True)

            else:
                feats = np.array([
                    preprocessing_utils.butterworth_filter(data=feat_io[feat], fs=365.25, cutoff=self.ntl_lowpassfilter)
                    for feat in self.loading_feats
                ])
                feats = feats[:, idx]

        if self.mm_to_m:
            feats *= 1e3

        if self.diff:
            feats = np.diff(feats, axis=-1)

        if self.norm:
            for ifeat, feat in enumerate(self.loading_feats):
                feats[ifeat, :] = (feats[ifeat, :] - self.norm_stats.loc[feat, 'mu']) / self.norm_stats.loc[feat, 'sig']

        return feats

    def load_targets(self, station: str,
                     idx: list[int] | tuple[int] | np.ndarray[int],
                     idx_feat: list[int] | tuple[int] | np.ndarray[int] = None) -> (np.ndarray, np.ndarray | None, int):
        """
        Load targets from disk
        Args:
            station: station code WITH segment specifier
            idx: global sample index for target sample series
            idx_feat: global sample index to use for extracting gnss series to be used as feature

        Returns: numpy array of sequential target with shape [batch_size, num_targets, target_length]
        """

        file_name = self.target_dir.joinpath(f'{station}_decomposition.npz')

        if self.diff and self.target_length == 1:
            # elongate the retrieved index list to enable 1st order differentiation
            idx = [idx[0], idx[0] + 1]

        with np.load(file=file_name, mmap_mode='r') as tar_io:

            if not self.lowpassfilter:
                tars = np.array([tar_io[self.target][idx]])
                tar_time = tar_io['time'][idx]
            else:
                tars = np.array([tar_io[self.target]])
                tars = preprocessing_utils.butterworth_filter(data=tars, fs=365.25, cutoff=self.lowpassfilter)
                tars = tars[idx]
                tars = tars[:, None]
                tar_time = tar_io['time'][idx]

            if len(self.gnss_feats):

                gnss_feat = np.array([tar_io[gfeat][idx_feat] if gfeat != 'full'
                                      else tar_io['data'][idx_feat] + tar_io['tectonics'][idx_feat]
                                      for gfeat in self.gnss_feats])

            else:
                gnss_feat = None

        if self.mm_to_m:
            tars *= 1e3
            if self.gnss_feats:
                gnss_feat *= 1e3

        if self.diff:
            tars = np.diff(tars, axis=-1)
            if len(self.gnss_feats):
                gnss_feat = np.diff(gnss_feat, axis=-1)

        if self.norm:
            tars = (tars - self.norm_stats.loc[self.target, 'mu']) / self.norm_stats.loc[self.target, 'sig']

            if len(self.gnss_feats):
                for ifeat, feat in enumerate(self.gnss_feats):
                    gnss_feat[ifeat, :] = (
                            (gnss_feat[ifeat, :] - self.norm_stats.loc[feat, 'mu']) / self.norm_stats.loc[feat, 'sig']
                    )

        return tars, gnss_feat, tar_time

    def temporal_encoding(self, t_in: int, period: float = 365.25) -> (float, float):
        t_in = datetime.datetime.fromordinal(t_in).timetuple().tm_yday - 1
        omega = 2 * np.pi / period
        cos_t = np.cos(t_in * omega)
        sin_t = np.sin(t_in * omega)

        if self.norm:
            cos_t /= self.harmonic_std
            sin_t /= self.harmonic_std

        return cos_t, sin_t

    def spatial_encoding(self, lat: float, lon: float) -> (float, float, float):
        lon_1 = np.sin(np.deg2rad(lon) * (2 * np.pi / np.deg2rad(360)))
        lon_2 = np.cos(np.deg2rad(lon) * (2 * np.pi / np.deg2rad(360)))
        lat_1 = np.sin(np.deg2rad(lat) * (2 * np.pi / np.deg2rad(180)))

        if self.norm:
            lon_1 /= self.harmonic_std
            lon_2 /= self.harmonic_std
            lat_1 /= self.harmonic_std

        return lon_1, lon_2, lat_1
