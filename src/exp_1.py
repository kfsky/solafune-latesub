import numpy as np
import pandas as pd
from dataclasses import dataclass
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import re
from tabulate import tabulate
import typing as tp
from typing import List
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib

import string
import lightgbm as lgb
import xgboost as xgb
import random
import os
from tqdm import tqdm

from vivid import create_runner
from vivid.backends import LocalExperimentBackend
from vivid.cacheable import cacheable
from vivid.env import Settings
from vivid.estimators.base import MetaBlock
from vivid.estimators.boosting import LGBMRegressorBlock
from vivid.estimators.boosting import XGBRegressorBlock
from vivid.estimators.boosting.block import create_boosting_seed_blocks
from vivid.estimators.ensumble import RFRegressorBlock
from vivid.estimators.linear import TunedRidgeBlock
from vivid.features.base import CountEncodingBlock, OneHotEncodingBlock, FilterBlock
from vivid.core import BaseBlock as VividBaseBlock
from vivid.setup import setup_project
from vivid.utils import Timer


from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from sklearn import metrics
import gensim
from contextlib import contextmanager
from time import time

from collections import Counter
import category_encoders as ce
from xfeat import (SelectCategorical, LabelEncoder, Pipeline, ConcatCombination, SelectNumerical,
                   ArithmeticCombinations, TargetEncoder, aggregation, GBDTFeatureSelector, GBDTFeatureExplorer)

import warnings
warnings.filterwarnings('ignore')

input_dirs = '../input/'
output_dirs = '../output/'


# seed set
def seed_everything(seed=71):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True


seed_everything()


class FillnaBlock(VividBaseBlock):
    _save_attributes = [
        "fill_values_"
    ]

    def fit (self,
             source_df: pd.DataFrame,
             y: Union[None, np.ndarray],
             experiment) -> pd.DataFrame:
        self.fill_values_ = source_df.median()
        return self.transform(source_df)

    def transform(self, source_df:pd.DataFrame) -> pd.DataFrame:
        output_df = source_df.replace([np.inf, -np.inf], np.nan).fillna(self.fill_values_).fillna(0)
        return output_df


class BaggingSVRegressorBlock(MetaBlock):
    def model_class(selfself, *args, **kwargs):
        return BaggingRegressor(
            base_estimator=make_pipeline(
                StandardScaler(),
                SVR(*args, **kwargs)
            ),
            n_estimators=10,
            max_samples=2,
            n_jobs=-1
        )


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


@cacheable
def read_csv(name: str) -> pd.DataFrame:
    if '.csv' not in name:
        name = name + '.csv'

    project = setup_project()
    return pd.read_csv(os.path.join(project.output_root, name))


# aggregationのagg_methodsで使用
def max_min(x):
    return x.max() - x.min()


def q75_q25(x):
    return x.quantile(0.75) - x.quantile(0.25)


def q25(x):
    return x.quantile(0.25)


def q50(x):
    return x.quantile(0.5)


def q75(x):
    return x.quantile(0.75)


def aggregation_cumfeat(input_df, group_key, group_values):
    output_df = pd.DataFrame()
    for col in group_values:
        if input_df.AverageLandPrice.min() == 1:
            new_col = f"cum_feat_{col}_grpby_{group_key}"
        else:
            new_col = f"all_cum_feat_{col}_grpby_{group_key}"
        input_df["lag"] = input_df.groupby(group_key)[[col]].shift(1)
        cum = input_df[[group_key] + ["lag"]].groupby(group_key).lag.agg(["cumsum", "cumcount"])
        new_df = pd.DataFrame(cum["cumsum"] / cum["cumcount"])
        new_df.columns = [new_col]
        output_df = pd.concat([output_df, new_df], axis=1)

    return output_df


def get_agg_cumfeat_features(input_df):
    _input_df = pd.concat([input_df, get_area_feature(input_df)], axis=1)

    group_key = "PlaceID"
    group_values = ["MeanLight", "SumLight"]
    output_df = aggregation_cumfeat(input_df,
                                    group_key=group_key,
                                    group_values=group_values)
    return output_df


# group 内で diffをとる関数
def diff_aggregation(input_df, group_key, group_values, num_diffs):
    dfs = []
    for nd in num_diffs:
        _df = input_df.groupby(group_key)[group_values].diff(nd)
        _df.columns = [f'diff={nd}_{col}_grpby_{group_key}' for col in group_values]
        dfs.append(_df)
    output_df = pd.concat(dfs, axis=1)
    return output_df


# group 内で shiftをとる関数
def shift_aggregation(input_df, group_key, group_values, num_shifts):
    dfs = []
    for ns in num_shifts:
        _df = input_df.groupby(group_key)[group_values].shift(ns)
        _df.columns = [f'shift={ns}_{col}_grpby_{group_key}' for col in group_values]
        dfs.append(_df)
    output_df = pd.concat(dfs, axis=1)
    return output_df


# そのままの値の特徴量
def get_raw_features(input_df):
    cols = [
        "MeanLight",
        "SumLight",
        "Year"
    ]
    return input_df[cols].copy()


# 面積
def get_area_feature(input_df):
    output_df = pd.DataFrame()
    output_df["Area"] = input_df["SumLight"] / (input_df["MeanLight"] + 1e-3)
    return output_df


# aggregation PlaceID
def get_agg_place_id_features(input_df):
    _input_df = pd.concat([input_df, get_area_feature(input_df)], axis=1)

    cols = 'PlaceID'

    output_df = pd.DataFrame()
    output_df, agg_cols = aggregation(_input_df,
                                      group_key=cols,
                                      group_values=["MeanLight", "SumLight", "Area"],
                                      agg_methods=["min", "max", "median", "mean", "std", "var", max_min, q75_q25, q25,
                                                   q50, q75],
                                      )

    return output_df[agg_cols]


# aggregation Year
def get_agg_year_features(input_df):
    _input_df = pd.concat([input_df, get_area_feature(input_df)], axis=1)

    cols = 'Year'

    output_df = pd.DataFrame()
    output_df, agg_cols = aggregation(_input_df,
                                      group_key=cols,
                                      group_values=["MeanLight", "SumLight", "Area"],
                                      agg_methods=["min", "max", "median", "mean", "std", "var", max_min, q75_q25, q25,
                                                   q50, q75],
                                      )

    return output_df[agg_cols]


# PlaceID をキーにしたグループ内差分
def get_diff_agg_place_id_features(input_df):
    group_key = "PlaceID"
    group_values = ["MeanLight", "SumLight"]
    num_diffs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    output_df = diff_aggregation(input_df,
                                 group_key=group_key,
                                 group_values=group_values,
                                 num_diffs=num_diffs)
    return output_df


# PlaceID をキーにしたグループ内シフト
def get_shift_agg_place_id_features(input_df):
    group_key = "PlaceID"
    group_values = ["MeanLight", "SumLight"]
    num_shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    output_df = shift_aggregation(input_df,
                                  group_key=group_key,
                                  group_values=group_values,
                                  num_shifts=num_shifts)
    return output_df


# pivot tabel を用いた特徴量
def get_place_id_vecs_features(input_df):
    _input_df = pd.concat([input_df, get_area_feature(input_df)], axis=1)
    # pivot table
    area_df = pd.pivot_table(_input_df, index="PlaceID", columns="Year", values="Area").add_prefix("Area=")
    mean_light_df = pd.pivot_table(_input_df, index="PlaceID", columns="Year", values="MeanLight").add_prefix(
        "MeanLight=")
    sum_light_df = pd.pivot_table(_input_df, index="PlaceID", columns="Year", values="SumLight").add_prefix("SumLight=")
    all_df = pd.concat([area_df, mean_light_df, sum_light_df], axis=1)

    # PCA all
    sc_all_df = StandardScaler().fit_transform(all_df.fillna(0))
    pca = PCA(n_components=64, random_state=2021)
    pca_all_df = pd.DataFrame(pca.fit_transform(sc_all_df), index=all_df.index).rename(
        columns=lambda x: f"PlaceID_all_PCA={x:03}")
    # PCA Area
    sc_area_df = StandardScaler().fit_transform(area_df.fillna(0))
    pca = PCA(n_components=16, random_state=2021)
    pca_area_df = pd.DataFrame(pca.fit_transform(sc_area_df), index=all_df.index).rename(
        columns=lambda x: f"PlaceID_Area_PCA={x:03}")
    # PCA MeanLight
    sc_mean_light_df = StandardScaler().fit_transform(mean_light_df.fillna(0))
    pca = PCA(n_components=16, random_state=2021)
    pca_mean_light_df = pd.DataFrame(pca.fit_transform(sc_mean_light_df), index=all_df.index).rename(
        columns=lambda x: f"PlaceID_MeanLight_PCA={x:03}")
    # PCA SumLight
    sc_sum_light_df = StandardScaler().fit_transform(sum_light_df.fillna(0))
    pca = PCA(n_components=16, random_state=2021)
    pca_sum_light_df = pd.DataFrame(pca.fit_transform(sc_sum_light_df), index=all_df.index).rename(
        columns=lambda x: f"PlaceID_SumLight_PCA={x:03}")

    df = pd.concat([all_df, pca_all_df, pca_area_df, pca_mean_light_df, pca_sum_light_df], axis=1)
    output_df = pd.merge(_input_df[["PlaceID"]], df, left_on="PlaceID", right_index=True, how="left")
    return output_df.drop("PlaceID", axis=1)


# PlaceIDをキーにしたグループ内相関係数
def get_corr_features(input_df):
    _input_df = pd.concat([input_df, get_area_feature(input_df)], axis=1)
    group_key = "PlaceID"
    group_vlaues = [
        ["Year", "MeanLight"],
        ["Year", "SumLight"],
        ["Year", "Area"],
    ]
    dfs = []
    for gv in group_vlaues:
        _df = _input_df.groupby(group_key)[gv].corr().unstack().iloc[:, 1].rename(f"Corr={gv[0]}-{gv[1]}")
        dfs.append(pd.DataFrame(_df))
    dfs = pd.concat(dfs, axis=1)
    output_df = pd.merge(_input_df[[group_key]], dfs, left_on=group_key, right_index=True, how="left").drop(group_key,
                                                                                                            axis=1)
    return output_df


# count 63
def get_count63_feature(input_df):
    # 各地域でMeanLightが63をとった回数を特徴量にする
    _mapping = input_df[input_df['MeanLight'] == 63].groupby('PlaceID').size()

    output_df = pd.DataFrame()
    output_df['count63'] = input_df['PlaceID'].map(_mapping).fillna(0)
    return output_df


def to_features(train, test):
    input_df = pd.concat([train, test]).reset_index(drop=True)

    processes = [
        get_raw_features,
        get_area_feature,
        get_agg_place_id_features,
        get_agg_year_features,
        get_diff_agg_place_id_features,
        get_shift_agg_place_id_features,
        get_place_id_vecs_features,
        get_corr_features,
        get_count63_feature,
        get_agg_cumfeat_features
    ]

    output_df = pd.DataFrame()
    for func in tqdm(processes):
        _df = func(input_df)
        assert len(_df) == len(input_df), func.__name__
        output_df = pd.concat([output_df, _df], axis=1)

    train_x = output_df.iloc[:len(train)]
    test_x = output_df.iloc[len(train):].reset_index(drop=True)
    return train_x, test_x


class GroupKFold:
    """
    GroupKFold with random shuffle with a sklearn-like structure
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, group=None):
        return self.n_splits

    def split(self, X=None, y=None, group=None):
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        unique_ids = group.unique()
        for tr_group_idx, va_group_idx in kf.split(unique_ids):
            # split group
            tr_group, va_group = unique_ids[tr_group_idx], unique_ids[va_group_idx]
            train_idx = np.where(group.isin(tr_group))[0]
            val_idx = np.where(group.isin(va_group))[0]
            yield train_idx, val_idx


# PlaceID をキーにした Group K fold
def make_gkf(X, y, base_df, n_splits=5, random_state=2020):
    gkf = GroupKFold(n_splits=n_splits, random_state=random_state)
    return list(gkf.split(X, y, base_df["PlaceID"]))


def get_filename_without_extension(filename: str):
    return os.path.basename(filename).split('.')[0]


@dataclass
class RuntimeEnv:
    input_dir: str
    output_root: str
    force: bool = False
    simple: bool = False

    @property
    def output_dirpath(self) -> str:
        """実験結果を出力するディレクトリ"""
        file_name = get_filename_without_extension(__file__)
        return os.path.join(self.output_root, 'experiments', file_name)

    def initialize(self):
        Settings.PROJECT_ROOT = self.input_dir
        os.makedirs(self.output_dirpath, exist_ok=True)


def create_runtime_environment() -> RuntimeEnv:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', help='input directory', default='./data/inputs')
    parser.add_argument('--output', help='output directory', default='./data/outputs')
    parser.add_argument('--force', action='store_true', help='If add me, re-create all models.')
    parser.add_argument('--simple', action='store_true', help='if add me, create lightgbm model only. (skip other models)')
    args = parser.parse_args()
    runtime_env = RuntimeEnv(
        input_dir=args.input,
        output_root=args.output,
        force=args.force,
        simple=args.simple
    )
    runtime_env.initialize()
    return runtime_env


def create_model_blocks(cv, simple=False) -> List[VividBaseBlock]:
    fillna_block = FillnaBlock(name='FNA')

    init_params = {
        'cv': cv
    }
    single_models = [
        create_boosting_seed_blocks(
            feature_class=LGBMRegressorBlock, prefix='lgbm_',
            add_init_params={
                "n_estimators": 20000,
                "learning_rate": 0.01,
                "num_leaves": 36,
                "n_jobs": -1,
                "importance_type": "gain",
                'colsample_bytree': .5,
                "reg_lambda": 5,
                "max_depth": 7,
            },
            init_params=init_params
        ),
    ]
    if simple:
        return single_models

    single_models += [
        # bagging SVR
        BaggingSVRegressorBlock(name='bagging_svr', parent=fillna_block, **init_params),

        # seed average(xgboost)
        create_boosting_seed_blocks(feature_class=XGBRegressorBlock, prefix='xgb_', add_init_params={
            'n_estimators': 10000, 'colsample_bytree': .5, 'learning_rate': .01
        }, init_params=init_params, n_seeds=3),

        # seed average (lightgbm)
        # コピペなんで、適宜変更すること
        create_boosting_seed_blocks(feature_class=LGBMRegressorBlock, prefix='lgbm_poisson_',
                                    add_init_params={
                                        'n_estimators': 10000, 'colsample_bytree': .2, 'learning_rate': .01,
                                        'objective': 'poisson',
                                        'eval_metric': 'rmse'
                                    }, init_params=init_params),

        # random forest
        RFRegressorBlock('rf', parent=fillna_block, add_init_param={'n_jobs':-1}, **init_params),

        # ridge(linear model)
        TunedRidgeBlock('ridge', parent=fillna_block, n_trials=50, **init_params),
    ]

    stacked_models = [
        # stacking する ligthgbm
        LGBMRegressorBlock('stacked_lgbm', parent=single_models, add_init_param={
            'n_estimators': 10000, 'colsample_bytree': 1, 'learning_rate': .05
        }, **init_params),

        # stacking + もとの特徴量を使う lightgbm
        LGBMRegressorBlock('stacked_and_raw_lgbm', parent=[*single_models, fillna_block], add_init_param={
            'n_estimators': 10000, 'colsample_bytree': 1, 'learning_rate': .05
        }, **init_params),

        # stacking ridge
        TunedRidgeBlock('stacked_ridge', parent=single_models, n_trials=50, **init_params)
    ]

    return stacked_models


def decoration(s, deco=None):
    if deco is None:
        deco = '=' * 30
    s = deco + s + deco
    return s


def run_pseudo_round(train_df: pd.DataFrame,
              test_df: pd.DataFrame,
              y,
              cv,
              output_dir,
              n_train,
              force: bool,
              simple: bool) -> np.ndarray:

    # round が進んでいるときには学習用データに pseudo label の data を付与
    if len(train_df) != n_train:
        extend_index = range(n_train, len(train_df))
        cv = [[np.hstack([idx_tr, extend_index]), idx_val] for idx_tr, idx_val in cv]

    blocks = create_model_blocks(cv=cv, simple=simple)
    runner = create_runner(blocks, experiment=LocalExperimentBackend(output_dir))
    oof_results = runner.fit(train_df=train_df, y=y, ignore_past_log=force)

    y_origin = y[:n_train]

    scores = []
    for result in oof_results:
        score_i = mean_squared_error(y_origin, result.out_df.values[:n_train, 0]) ** .5
        scores.append([result.block.name, score_i])

    score_df = pd.DataFrame(scores, columns=['name', 'score']).sort_values('score').reset_index()
    print(decoration(' OOF Scores '))
    print(tabulate(score_df, headers='keys', tablefmt='psql'))
    score_df.to_csv(os.path.join(output_dir, "score.csv"), index=False)

    test_results = runner.predict(test_df)

    best_test_pred = None
    base_oof_model_name = score_df["name"].values[0]

    for result in test_results:
        x = result.out_df.values[:, 0]
        x = np.expm1(x)
        x = np.where(x < 0, 0, x)

        if result.block.name == base_oof_model_name:
            print(f'use {result.block.name} for next predict')
            best_test_pred = np.log1p(np.copy(x))

        pd.DataFrame({
            "AverageLandPrice": x
        }).to_csv(os.path.join(output_dir, result.block.name + '.csv'), index=False)

    return best_test_pred


def main():
    runtime = create_runtime_environment()
    train_df = pd.read_csv(input_dirs+"TrainDataSet.csv")
    test_df = pd.read_csv(input_dirs + "EvaluationData.csv")

    target_data = "AverageLandPrice"

    train_x, test_x = to_features(train_df, test_df)
    train_ys = train_df[target_data]
    train_ys = np.log1p(train_ys)

    cv = make_gkf(train_x, train_ys, train_df)

    joblib.dump(train_x, os.path.join(runtime.output_dirpath, "train_feat.joblib"))
    joblib.dump(test_x, os.path.join(runtime.output_dirpath, 'test_feat.joblib'))

    feat_df = train_x.copy()
    run_pseudo_round(feat_df,
                     test_x,
                     train_ys,
                     cv,
                     output_dir=runtime.output_dirpath,
                     n_train=len(train_df),
                     force=runtime.force,
                     simple=runtime.simple)


if __name__ == '__main__':
    main()
