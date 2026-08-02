from typing import Literal
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed


# Константы

TRAIN_WINDOW = 76
OFFSET = 13
TEST_WINDOW = 32


# Приведение типов
def prepare_types(train:pd.DataFrame, test:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.sort_values(['date', 'fullVisitorId']).reset_index(drop=True)
    test = test.sort_values(['date', 'fullVisitorId']).reset_index(drop=True)

    train.totals_transactionRevenue = train.totals_transactionRevenue.fillna(0)

    train.date = pd.to_datetime(train.visitStartTime, unit='s').dt.to_period("D")
    test.date = pd.to_datetime(test.visitStartTime, unit='s').dt.to_period("D")

    train['totals_hits'] = train['totals_hits'].fillna(0).astype(int)
    test['totals_hits'] = test['totals_hits'].fillna(0).astype(int)

    train['totals_pageviews'] = train['totals_pageviews'].fillna(1).astype(int)
    test['totals_pageviews'] = test['totals_pageviews'].fillna(1).astype(int)

    train.totals_bounces = train.totals_bounces.fillna(0).astype(int)
    test.totals_bounces = test.totals_bounces.fillna(0).astype(int)

    train.totals_newVisits = train.totals_newVisits.fillna(0).astype(int)
    test.totals_newVisits = test.totals_newVisits.fillna(0).astype(int)

    train = train[train.columns.drop(list(train.filter(regex='trafficSource')))]
    test = test[test.columns.drop(list(test.filter(regex='trafficSource')))]

    return train, test

# Обработка категорий


def get_cat_label_encoding(train:pd.DataFrame, test:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    le = LabelEncoder()

    data = pd.concat([train.drop(['totals_transactionRevenue', 'fullVisitorId', 'sessionId'], axis=1),
                      test.drop(['fullVisitorId', 'sessionId'], axis=1)])
    for feat in data.select_dtypes(include = "object").columns:
        le_trained = le.fit(data[feat].fillna(data[feat].mode()[0]).astype(str))
        train[feat] = le_trained.transform(train[feat].fillna(data[feat].mode()[0]))
        test[feat] = le_trained.transform(test[feat].fillna(data[feat].mode()[0]))

    return train, test


def get_cat_target_encoding(train:pd.DataFrame, test:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    category_columns = train.drop(['fullVisitorId', 'sessionId'], axis=1).select_dtypes(include=["object"]).columns
    train_coded = train.copy()
    test_coded = test.copy()
    y = train["totals_transactionRevenue"]
    for i in category_columns:
        train_coded[i], test_coded[i] = target_encode(train_coded[i],
                                                      test_coded[i],
                                                      target=y,
                                                      min_samples_leaf=100,
                                                      smoothing=10,
                                                      noise_level=0.01)

    return train_coded, test_coded


def add_noise(data, noise_level):
    return data * (1 + noise_level * np.random.randn(len(data)))


def target_encode(train_series=None,
                  test_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    assert len(train_series) == len(target)
    assert train_series.name == test_series.name
    temp = pd.concat([train_series, target], axis=1)

    averages = temp.groupby(by=train_series.name)[target.name].agg(["mean", "count"])

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    prior = target.mean()
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    ft_train_series = pd.merge(
        train_series.to_frame(train_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=train_series.name,
        how='left')['average'].rename(train_series.name + '_mean').fillna(prior)

    ft_train_series.index = train_series.index
    ft_test_series = pd.merge(
        test_series.to_frame(test_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=test_series.name,
        how='left')['average'].rename(train_series.name + '_mean').fillna(prior)

    ft_test_series.index = test_series.index
    return add_noise(ft_train_series, noise_level), add_noise(ft_test_series, noise_level)

# Подготовка данных для валидации


def data_to_chunks(train:pd.DataFrame) -> list[pd.DataFrame]:
    first_date = train.date[0]
    chunks = []
    while first_date <= train.iloc[-1]['date']:
        last_date = first_date+timedelta(days=TRAIN_WINDOW+OFFSET+TEST_WINDOW)
        chunk = train[(train.date>= first_date) & (train.date<= last_date)]
        chunks.append(chunk)
        first_date = last_date+timedelta(days=1)

    for i, chunk in enumerate(chunks):
        train_part = chunk[chunk.date <= min(chunk.date) + timedelta(days=TRAIN_WINDOW)]
        test_part = chunk[chunk.date >= max(chunk.date) - timedelta(days=TEST_WINDOW)]
        returned_users = set(train_part.fullVisitorId).intersection(set(test_part.fullVisitorId))
        chunk['is_returned'] = np.where(chunk.fullVisitorId.isin(returned_users), 1, 0)
        chunks[i] = chunk

    return chunks


# Агрегация данных

def calculate_visitor_agg(chunk:pd.DataFrame) -> pd.DataFrame:
        return chunk.groupby('fullVisitorId', sort=False).agg(
            mode_channelGrouping = ('channelGrouping', lambda x: x.mode()[0]),
            last_channelGrouping = ('channelGrouping', lambda x: x.iloc[-1]),
            time_from_chunk_start=('date', lambda x: pd.to_timedelta(x.iloc[0] - chunk.iloc[0].date).days),
            time_to_chunk_finish=('date', lambda x: pd.to_timedelta(chunk.iloc[-1].date - x.iloc[-1]).days),
            time_interval=('date', lambda x: pd.to_timedelta(chunk.iloc[-1].date - chunk.iloc[0].date).days),
            unique_dates=('date', lambda x: x.nunique()),
            max_visitNumber=('visitNumber', lambda x: x.max()),
            mode_device_browser=('device_browser', lambda x: x.mode()[0]),
            mode_device_operatingSystem=('device_operatingSystem', lambda x: x.mode()[0]),
            mode_device_deviceCategory=('device_deviceCategory', lambda x: x.mode()[0]),
            geoNetwork_continent=('geoNetwork_continent', lambda x: x.mode()[0]),
            geoNetwork_subContinent=('geoNetwork_subContinent', lambda x: x.mode()[0]),
            geoNetwork_country=('geoNetwork_country', lambda x: x.mode()[0]),
            geoNetwork_region=('geoNetwork_region', lambda x: x.mode()[0]),
            geoNetwork_city=('geoNetwork_city', lambda x: x.mode()[0]),
            geoNetwork_networkDomain=('geoNetwork_networkDomain', lambda x: x.mode()[0]),
            max_hits=('totals_hits', lambda x: x.max()),
            min_hits=('totals_hits', lambda x: x.min()),
            sum_hits=('totals_hits', lambda x: x.sum()),
            mean_hits=('totals_hits', lambda x: x.mean()),
            max_pageviews=('totals_pageviews', lambda x: x.max()),
            min_pageviews=('totals_pageviews', lambda x: x.min()),
            sum_pageviews=('totals_pageviews', lambda x: x.sum()),
            mean_pageviews=('totals_pageviews', lambda x: x.mean()),
            sum_transactionRevenue=('totals_transactionRevenue', lambda x: x.sum()),
            sum_bounces=('totals_bounces', lambda x: np.nansum(x)),
            sum_newVisits=('totals_newVisits', lambda x: np.nansum(x)),
            sessions=('visitStartTime', lambda x: x.nunique()),
            is_returned=('is_returned', lambda x: x.max()),
        )


def calculate_visitor_agg_test(chunk: pd.DataFrame) -> pd.DataFrame:
    return chunk.groupby('fullVisitorId', sort=False).agg(
        mode_channelGrouping=('channelGrouping', lambda x: x.mode()[0]),
        last_channelGrouping=('channelGrouping', lambda x: x.iloc[-1]),
        time_from_chunk_start=('date', lambda x: pd.to_timedelta(x.iloc[0] - chunk.iloc[0].date).days),
        time_to_chunk_finish=('date', lambda x: pd.to_timedelta(chunk.iloc[-1].date - x.iloc[-1]).days),
        time_interval=('date', lambda x: pd.to_timedelta(chunk.iloc[-1].date - chunk.iloc[0].date).days),
        unique_dates=('date', lambda x: x.nunique()),
        max_visitNumber=('visitNumber', lambda x: x.max()),
        mode_device_browser=('device_browser', lambda x: x.mode()[0]),
        mode_device_operatingSystem=('device_operatingSystem', lambda x: x.mode()[0]),
        mode_device_deviceCategory=('device_deviceCategory', lambda x: x.mode()[0]),
        geoNetwork_continent=('geoNetwork_continent', lambda x: x.max()),
        geoNetwork_subContinent=('geoNetwork_subContinent', lambda x: x.max()),
        geoNetwork_country=('geoNetwork_country', lambda x: x.max()),
        geoNetwork_region=('geoNetwork_region', lambda x: x.max()),
        geoNetwork_city=('geoNetwork_city', lambda x: x.max()),
        geoNetwork_networkDomain=('geoNetwork_networkDomain', lambda x: x.max()),
        max_hits=('totals_hits', lambda x: x.max()),
        min_hits=('totals_hits', lambda x: x.min()),
        sum_hits=('totals_hits', lambda x: x.sum()),
        mean_hits=('totals_hits', lambda x: x.mean()),
        max_pageviews=('totals_pageviews', lambda x: x.max()),
        min_pageviews=('totals_pageviews', lambda x: x.min()),
        sum_pageviews=('totals_pageviews', lambda x: x.sum()),
        mean_pageviews=('totals_pageviews', lambda x: x.mean()),
        sum_bounces=('totals_bounces', lambda x: np.nansum(x)),
        sum_newVisits=('totals_newVisits', lambda x: np.nansum(x)),
        sessions=('visitStartTime', lambda x: x.nunique()),
    )


def get_visitor_aggs(path_to_data='H:/D/PycharmProjects/ML2/google',
                     from_cache=False, cat_encoding:Literal['target', 'label'] = 'label') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if from_cache:
        visitor_aggs_train = pd.read_parquet(path_to_data+'/visitor_aggs_train.parquet')
        visitor_aggs_valid = pd.read_parquet(path_to_data+'/visitor_aggs_valid.parquet')
        visitor_aggs_test = pd.read_parquet(path_to_data+'/visitor_aggs_test.parquet')

    else:
        train = pd.read_parquet(path_to_data+'/train.parquet')
        test = pd.read_parquet(path_to_data+'/test.parquet')
        if cat_encoding == 'label':
            train, test = get_cat_label_encoding(*prepare_types(train, test))
        elif cat_encoding == 'target':
            train, test = get_cat_target_encoding(*prepare_types(train, test))
        else:
            msg = "Invalid cat_encoding. It must be label or target"
            raise ValueError(msg)
        chunks = data_to_chunks(train)

        visitor_aggs_train = Parallel(n_jobs=-1)(
            delayed(calculate_visitor_agg)(chunk[chunk.date <= chunk.date.min()+timedelta(days=TRAIN_WINDOW)]) for chunk in chunks)
        visitor_aggs_valid = Parallel(n_jobs=-1)(
            delayed(calculate_visitor_agg)(chunk[chunk.date >= chunk.date.max() - timedelta(days=TEST_WINDOW)]) for chunk in
            chunks)
        visitor_aggs_test = calculate_visitor_agg_test(test)
        for i in range(len(visitor_aggs_train)):
            visitor_aggs_train[i]['order'] = i
            visitor_aggs_valid[i]['order'] = i
        visitor_aggs_train_all = pd.concat(visitor_aggs_train)
        visitor_aggs_valid_all = pd.concat(visitor_aggs_valid)
        visitor_aggs_train_all.to_parquet(path_to_data+'visitor_aggs_train.parquet')
        visitor_aggs_valid_all.to_parquet(path_to_data+'visitor_aggs_valid.parquet')
        visitor_aggs_test.to_parquet(path_to_data+'visitor_aggs_test.parquet')

    return visitor_aggs_train, visitor_aggs_valid, visitor_aggs_test


def prepare_data_for_model(visitor_aggs_train, visitor_aggs_valid, only_returned):
    if only_returned:
        is_returned = 0
    else:
        is_returned = -1
    X = visitor_aggs_train.drop(columns=["sum_transactionRevenue", "is_returned"])
    y = visitor_aggs_train["is_returned"]
    X_ret = visitor_aggs_train[visitor_aggs_train['is_returned'] > is_returned].drop(
        columns=["sum_transactionRevenue", "is_returned"])
    y_ret = visitor_aggs_train[visitor_aggs_train['is_returned'] > is_returned]["sum_transactionRevenue"]
    X_valid = visitor_aggs_valid.drop(columns=["sum_transactionRevenue", "is_returned"])
    y_valid = visitor_aggs_valid["sum_transactionRevenue"]
    X_valid_ret = visitor_aggs_valid[visitor_aggs_valid['is_returned'] > is_returned].drop(
        columns=["sum_transactionRevenue", "is_returned"])

    return X, y, X_ret, y_ret, X_valid, y_valid, X_valid_ret
