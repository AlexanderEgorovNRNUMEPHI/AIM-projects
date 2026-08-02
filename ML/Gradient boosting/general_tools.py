import pandas as pd
import numpy as np


def get_outlier_bounds(df:pd.DataFrame) -> tuple[float, float]:
    m, std = np.mean(df), np.std(df)
    threshold = std * 3
    lower, upper = m - threshold, m + threshold

    outliers_lower = [x for x in df if x < lower]
    outliers_higher = [x for x in df if x > upper]
    outliers_total = [x for x in df if x < lower or x > upper]

    print('Identified lowest outliers: %d' % len(outliers_lower))
    print('Identified upper outliers: %d' % len(outliers_higher))
    print('Identified outliers: %d' % len(outliers_total))
    return lower, upper


def get_outliers(train_all:pd.DataFrame) -> list[str]:
    left_thresh, right_thresh = get_outlier_bounds(
        np.log(train_all[train_all['sum_transactionRevenue'] > 0]["sum_transactionRevenue"]))

    log_sum_transactionRevenue = np.log(train_all[train_all['sum_transactionRevenue'] > 0]["sum_transactionRevenue"])
    outliers = (log_sum_transactionRevenue[
        (log_sum_transactionRevenue <= left_thresh) | (log_sum_transactionRevenue >= right_thresh)]).index
    return outliers


def prepare_final_df_one_model(X_valid:pd.DataFrame,
                               X_valid_ret:pd.DataFrame,
                               y_valid:pd.Series,
                               predict_regression:pd.DataFrame,
                               predict_prob:pd.DataFrame,
                               shift_proba: float= 0) -> pd.DataFrame:
    reg_pred = pd.DataFrame(index=X_valid_ret.index)
    reg_pred['money'] = np.clip(predict_regression, 0, max(predict_regression))

    class_pred = pd.DataFrame(index=X_valid.index)
    class_pred['proba'] = predict_prob
    class_pred['proba'] += shift_proba
    df = class_pred.join(reg_pred).fillna(0)
    df['result'] = df.proba * df.money

    df = df.reset_index().groupby('fullVisitorId').agg({'result': 'sum'})
    df['real'] = y_valid.reset_index().groupby('fullVisitorId').agg({'sum_transactionRevenue': 'sum'})
    return df


def prepare_final_df_several_models(models:list[pd.DataFrame], model_order:int) -> pd.DataFrame:
    reg_pred = pd.DataFrame(index=models[model_order][3].index)
    reg_pred['money'] = np.clip(models[model_order][-1], 0, max(models[model_order][-1]))

    class_pred = pd.DataFrame(index=models[model_order][2].index)
    class_pred['proba'] = models[model_order][-2]
    df = class_pred.join(reg_pred).fillna(0)

    df['result'] = df.proba * df.money
    df['real'] = models[model_order][4]
    return df
