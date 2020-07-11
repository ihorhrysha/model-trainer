import numpy as np
import pandas as pd


def order_feature(
        df: pd.DataFrame,
        name: str,
        product_price_column: str,
        order_price_column: str
) -> pd.DataFrame:
    """ Create new order price based feature from `price_column`
    """
    df = df.copy()
    order_id_col = 'OrderId'
    order_qty_col = 'OrderQty'
    df[product_price_column] = df[order_price_column] * df[order_qty_col]

    right_df = df[[order_id_col, product_price_column]].\
        groupby(order_id_col).\
        sum().\
        reset_index().\
        rename(columns={product_price_column: name})

    if name in df.columns:
        df = df.drop(columns=name)
    return df.merge(
        right_df,
        how='left',
        on='OrderId'
    )


def order_price_feature(df):
    """ Create TotalOrderPrice feature (total BasePrice of items in order)
    """
    return order_feature(
        df,
        name='TotalOrderPrice',
        product_price_column='TotalOrderProductPrice',
        order_price_column='BasePrice'
    )


def order_revenue_feature(df):
    """ Create TotalOrderRevenue feature (total OrderPrice of items in order)
    """
    return order_feature(
        df,
        name='TotalOrderRevenue',
        product_price_column='TotalOrderProductRevenue',
        order_price_column='OrderPrice'
    )


def order_cost_feature(df):
    """ Create TotalOrderCost feature (total BasePrice of items in order)
    """
    return order_feature(
        df,
        name='TotalOrderCost',
        product_price_column='TotalOrderProductCost',
        order_price_column='CostPerItem'
    )


def discount_metric(y, y_pred, ds_test):
    """ Calculates metric to estimate the goodness of the discount policy.

    Needs input in original scale !!!
    """

    predicted_df = ds_test.copy()
    predicted_df["BaseDiscount"] = y_pred
    predicted_df["OrderPrice"] = \
        predicted_df["BasePrice"] \
        - predicted_df["BaseDiscount"] * predicted_df["BasePrice"]

    cost_df = order_cost_feature(ds_test)
    predicted_revenue_df = order_revenue_feature(predicted_df)
    revenue_df = order_revenue_feature(ds_test)

    predicted_revenue = predicted_revenue_df.TotalOrderProductRevenue.sum()
    real_revenue = revenue_df.TotalOrderProductRevenue.sum()

    # special metric
    predicted_income_product = (
            predicted_revenue_df.TotalOrderProductRevenue
            - cost_df.TotalOrderProductCost
    ).reset_index(drop=True)

    loss_orders_ind = np.where(predicted_income_product <= 0)[0]
    profit_orders_ind = np.where(predicted_income_product > 0)[0]

    mses = ((y - y_pred) ** 2).reset_index(drop=True)
    mses = (
        (mses[loss_orders_ind] * 10).to_list()
        + mses[profit_orders_ind].to_list()
    )
    return np.mean(mses), predicted_revenue, real_revenue
