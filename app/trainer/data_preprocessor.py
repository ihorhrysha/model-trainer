from functools import partial

import numpy as np
import pandas as pd
import pycountry
from countryinfo import CountryInfo
from joblib import delayed, Parallel


class DataPreprocessor:
    def preprocess(self, df):
        # Feature construction

        print('Shape before Feature construction:', df.shape)

        df = self._finance_features(df)
        df = self._construct_RFMD(df)
        df = self._location_features(df, if_print=False)
        df = self._time_features(df)
        df = self._order_price_feature(df)

        # Feature preparation

        print('Shape before Feature preparation:', df.shape)

        df = self._status_preparation(df)
        df = self._segment_preparation(df)
        df = self._platform_preparation(df)

        # filters & droppers ))
        print('Before filters:', df.shape)

        df = self._filter_orders_w_products_wo_price(df)
        df = self._filter_orders_wo_price(df)
        df = self._filter_orders_range(df, 'BasePrice', (0.3, 1.5))
        df = self._filter_orders_range(df, 'UserPrice', (0.3, 1.5))
        df = self._drop_orders_w_negative_values(df, 'BaseDiscount')
        df = self._drop_orders_w_negative_values(df, 'UserDiscount')

        # cleaning nan's
        df = self._drop_orders_nan_values(df, 'BaseDiscount')
        df = self._drop_orders_nan_values(df, 'UserDiscount')
        df = self._nan_cleaner(df, threshold=0.7)

        return df

    @staticmethod
    def _get_products_wo_price(df):
        all_nan_products = df[
            ['Product', 'UserPrice', 'BasePrice']].\
            groupby('Product').\
            apply(lambda x: x.isna().all()).\
            any(axis=1)
        return all_nan_products[all_nan_products].index.values

    def _filter_orders_w_products_wo_price(self, df):
        """ Drops orders which contain products with only NaN `BasePrice` or `UserPrice`
        """
        all_nan_products = self._get_products_wo_price(df)
        orders_to_drop = df[df['Product'].isin(all_nan_products)]
        orders_to_drop = orders_to_drop['OrderId'].unique()
        return df[~df['OrderId'].isin(orders_to_drop)]

    @staticmethod
    def _filter_orders_wo_price(df):
        """ Drop orders with some missing prices
        """
        orders_to_drop = df.set_index(
            'OrderId')[['BasePrice', 'UserPrice']].isna().any(axis=1)
        orders_to_drop = orders_to_drop[orders_to_drop].index.unique()
        return df[~df['OrderId'].isin(orders_to_drop)]

    @staticmethod
    def _filter_orders_range(df, column, range=(0.3, 1.5)):
        """ Drop orders with suspiciously high or low value of `column` (relatively to the mean value for the Product)
        """
        gb = df[['Product', column]].groupby('Product')
        bp_df = gb.mean().squeeze().rename(column+'Mean')

        df_temp = df.merge(bp_df.reset_index(), how='left', on='Product')
        rows_to_drop = (df_temp[column] / df_temp[column+'Mean'])
        rows_to_drop = (rows_to_drop > range[1]) | (rows_to_drop < range[0])
        rows_to_drop = rows_to_drop[rows_to_drop].index.unique().values
        orders_to_drop = df[df.index.isin(rows_to_drop)]['OrderId'].unique()
        return df[~df['OrderId'].isin(orders_to_drop)]

    @staticmethod
    def _drop_orders_w_negative_values(df, column):
        """ Drops orders with any negative value in `column` (e.g. to drop negative discounts)
        """
        orders_to_drop = df[df[column] < 0]['OrderId'].unique()
        return df[~df['OrderId'].isin(orders_to_drop)]

    @staticmethod
    def _drop_orders_nan_values(df, column):
        """ Drops orders with nan value in `column` (e.g. to drop nan discounts)
        """
        orders_to_drop = df[df[column].isna()]['OrderId'].unique()
        return df[~df['OrderId'].isin(orders_to_drop)]

    @staticmethod
    def _finance_features(df: pd.DataFrame):
        df = df.copy()

        df.eval('OrderPrice = OrderAmount / OrderQty', inplace=True)
        df.eval('BaseDiscount = (BasePrice - OrderPrice) / BasePrice',
                inplace=True)
        df.eval('UserDiscount = (UserPrice - OrderPrice) / UserPrice',
                inplace=True)

        # from transaction data
        df["CostPerItem"] = df["SoldCost"]/df["SoldQty"]

        # sort data
        df.sort_values(by=['OrderDate'], inplace=True)

        # suppose in case with unsold OrderQty=SoldQty
        df['SoldQty'].fillna(df['OrderQty'], inplace=True)

        # filling cost from prev/next line
        df['CostPerItem'] = df.groupby(['ProductId'], sort=False)[
            'CostPerItem'].apply(lambda x: x.ffill().bfill())

        # calculate dependant fields
        df.eval('ProfitPerItem = OrderPrice - CostPerItem', inplace=True)
        df.eval('Profit = ProfitPerItem * SoldQty', inplace=True)

        df['SoldCost'].fillna(df.eval('SoldQty * CostPerItem'),
                              inplace=True)
        return df

    @staticmethod
    def _construct_RFMD(df):
        """ 

        Add leading zeros and creates feature RFMD_zeros
        Splits RFMD in 4 separate classes: Recency, Frequency, Monetary, Duration 
        more about RFMD
        https://en.wikipedia.org/wiki/RFM_(market_research)

        Parameters: 
        df (DataFrame): Pandas dataframe

        Returns: 
        df (DataFrame): Pandas dataframe 

        """

        df = df.copy()
        df["RFMD"].fillna(0, inplace=True)
        df["RFMD"] = df["RFMD"].astype(int)
        df["RFMD_zeros"] = df["RFMD"].apply(lambda x: str(x).zfill(4))

        #Recency, Frequency, Monetary, Duration
        df["Recency"] = df["RFMD_zeros"].apply(lambda x: x[0:1])
        df["Frequency"] = df["RFMD_zeros"].apply(lambda x: x[1:2])
        df["Monetary"] = df["RFMD_zeros"].apply(lambda x: x[2:3])
        df["Duration"] = df["RFMD_zeros"].apply(lambda x: x[3:4])

        return df

    @staticmethod
    def _location_features(price_data, if_print=True):
        '''
        Function to construct two location features: region and subregion

        if_print == True => print all logs
        '''

        def set_info(country, info):
            try:
                new_country = str(country).strip().split(",")[0]
                return (country, CountryInfo(new_country).info()[info])
            except:
                try:
                    new_country = pycountry.countries.search_fuzzy(new_country)[
                        0].official_name
                    return (country, CountryInfo(new_country).info()[info])
                except:
                    if if_print:
                        print(f"not found {info} for {country}")
                    unknown.append(country)
                    return (country, "")

        unknown = []
        regions = Parallel(n_jobs=-1)(
            delayed(partial(set_info, info="region"))(x)
            for x in price_data["OrderCountry"].unique()
        )
        print(f"Total {len(unknown)} country regions were not found")
        print("Total rows of unknown region in the data: ",
              price_data[price_data.OrderCountry.isin(unknown)].shape[0])
        unknown = []
        subregions = Parallel(n_jobs=-1)(
            delayed(partial(set_info, info="subregion"))(x)
            for x in price_data["OrderCountry"].unique()
        )
        print(f"Total {len(unknown)} country subregions were not found")
        print("Total rows of unknown subregions in the data: ",
              price_data[price_data.OrderCountry.isin(unknown)].shape[0])

        df = price_data.merge(
            pd.DataFrame(regions,
                         columns=["OrderCountry", "CountryRegion"]),
            on="OrderCountry",
            how="left"
        )
        df = df.merge(
            pd.DataFrame(subregions,
                         columns=["OrderCountry", "CountrySubregion"]),
            on="OrderCountry",
            how="left"
        )
        return df

    @staticmethod
    def _time_features(df):
        '''
        Create time features from date
        '''
        df = df.copy()
        df["OrderDay"] = df.OrderDate.dt.day
        df["OrderYearDay"] = df.OrderDate.dt.dayofyear
        df["OrderWeekDay"] = df.OrderDate.dt.weekday
        df["OrderWeek"] = df.OrderDate.dt.week
        df["OrderHour"] = df.OrderDate.dt.hour
        df["OrderMonth"] = df.OrderDate.dt.month
        return df

    @staticmethod
    def _order_price_feature(df):
        """ Create TotalOrderPrice feature (total BasePrice of items in order)
        """
        df = df.copy()
        df.eval('TotalOrderProductPrice = BasePrice * OrderQty', inplace=True)

        right_df = df[['OrderId', 'TotalOrderProductPrice']].\
            groupby('OrderId').\
            sum().\
            reset_index().\
            rename(columns={'TotalOrderProductPrice': 'TotalOrderPrice'})

        if 'TotalOrderPrice' in df.columns:
            df.drop(columns='TotalOrderPrice', inplace=True)
        return df.merge(
            right_df,
            how='left',
            on='OrderId'
        )

    @staticmethod
    def _order_revenue_feature(df):
        """ Create TotalOrderRevenue feature (total OrderPrice of items in order)
        """
        df = df.copy()
        df.eval('TotalOrderProductRevenue = OrderPrice * OrderQty',
                 inplace=True)

        right_df = df[['OrderId', 'TotalOrderProductRevenue']].\
            groupby('OrderId').\
            sum().\
            reset_index().\
            rename(columns={'TotalOrderProductRevenue': 'TotalOrderRevenue'})

        if 'TotalOrderRevenue' in df.columns:
            df.drop(columns='TotalOrderRevenue', inplace=True)
        return df.merge(
            right_df,
            how='left',
            on='OrderId'
        )

    @staticmethod
    def _platform_preparation(data):
        data = data.copy()
        data.loc[data['IsExternal'], 'ExternalInternal'] = 'external'
        data.loc[~data['IsExternal'], 'ExternalInternal'] = 'internal'
        data['PlatformType'] = data['Region'] + '_' + data['ExternalInternal']
        data.drop('ExternalInternal', axis=1, inplace=True)
        return data

    @staticmethod
    def _order_cost_feature(df):
        """ Create TotalOrderCost feature (total BasePrice of items in order)
        """
        df = df.copy()
        df.eval('TotalOrderProductCost = CostPerItem * OrderQty', inplace=True)
        right_df = df[['OrderId', 'TotalOrderProductCost']].\
            groupby('OrderId').\
            sum().\
            reset_index().\
            rename(columns={'TotalOrderProductCost': 'TotalOrderCost'})
        if 'TotalOrderCost' in df.columns:
            df.drop(columns='TotalOrderCost', inplace=True)
        return df.merge(
            right_df,
            how='left',
            on='OrderId'
        )

    def _status_preparation(self, data):
        # Duplicate/trial from business perspective are not valid values(test, errors, and some other garbage)
        df_filtered = data.query('Status!= "Duplicate/trial"')
        df_filtered = self._delete_rare_cat(
            df_filtered, 'Status', 'occurrence',
            min_occur=0.005,
            fill_value='Cancelled'
        )
        df_filtered.loc[df_filtered.Status.isna(), 'Status'] = 'Cancelled'
        return df_filtered

    def _segment_preparation(self, data):
        # change rare categories to 'Сегмент не определен' within non nans
        df_filtered = data[~data['Segment'].isnull()]
        df_filtered = self._delete_rare_cat(
            df_filtered, 'Segment', 'occurrence',
            min_occur=0.005,
            fill_value='Сегмент не определен'
        )
        # Сегмент не определен == NaN
        df_nan = data[data['Segment'].isnull()]
        df_nan.eval('Segment = "Сегмент не определен"', inplace=True)
        return pd.concat([df_nan, df_filtered])

    @staticmethod
    def _nan_cleaner(df: pd.DataFrame, threshold=0.7):
        # Not sure if needed but can make TransactionDate equal OrderData
        df['TransactionDate'].fillna(df['OrderDate'], inplace=True)

        # Dropping columns with missing value rate higher than threshold
        df = df[df.columns[df.isnull().mean() < threshold]]

        # Max fill function for categorical columns
        cats_with_nan = ['BrandId', 'Brand', 'PriceTypeId',
                         'UnifiedPriceType', 'ManagerId', 'CategoryId']

        for cat_with_nan in cats_with_nan:
            df[cat_with_nan].fillna(
                df[cat_with_nan].value_counts().idxmax(), inplace=True)

        # Filling all missing values with 0
        return df.fillna(0)

    @staticmethod
    def _delete_rare_cat(price_data, cat_col, by="number", min_occur=0.01, n_leave=0.2, fill_value=None):
        '''
        Function to delete orders with rare values of some feature(or replace rare features with some value)

        by: {"number", "occurrence"} the strategy of deleting(replacing) rare features:
            "number": n_leave% of feature will be left, other - deleted(replaced)
            нп. Ми хочемо лишити тільки 20% категорій
            "occurrence": delete(replace) values that occur less than  min_occur% in the data
            нп. Ми хочемо видалити(замінити) всі категорії, які зустрічаються менше 1% у даних

        n_leave: % of feature values you want to leave (work only with by == "number")
            P.S. If n_leave is number, not percent, it will be converted to percents automatically
        min_occur: minimin % occurance in the data (work only with by == "occurrence")
            P.S. If min_occur is number, not percent, it will be converted to percents automatically

        fill_value: {None, strig} value to repleace rare values. If None, rare values will be deleted

        '''
        if by == "number":

            if n_leave >= 1:
                n_leave = n_leave/price_data[cat_col].nunique()

            temp = price_data[cat_col].value_counts()
            # 1-n_leave = % of the data we want to delete
            th = temp.quantile(1-n_leave)
            temp = temp[temp <= th]

            if fill_value == None:
                delete_orders = price_data[price_data[cat_col].isin(
                    temp.index)].OrderId.unique()
                price_data_less = price_data[~price_data["OrderId"].isin(
                    delete_orders)]
                print(
                    f"{len(delete_orders)} orders were deleted({(len(delete_orders)/price_data.OrderId.nunique())*100}%)")
            else:
                price_data_less = price_data.copy()
                price_data_less[cat_col] = np.where(price_data_less[cat_col].isin(
                    temp.index), fill_value, price_data_less[cat_col])

        elif by == "occurrence":
            if min_occur < 1:
                min_occur = min_occur*price_data.shape[0]

            temp = price_data[cat_col].value_counts()
            temp = temp[temp <= min_occur]
            if fill_value == None:
                delete_orders = price_data[price_data[cat_col].isin(
                    temp.index)].OrderId.unique()
                price_data_less = price_data[~price_data["OrderId"].isin(
                    delete_orders)]
                print(
                    f"{len(delete_orders)} orders were deleted({(len(delete_orders)/price_data.OrderId.nunique())*100}%)")
            else:
                price_data_less = price_data.copy()
                price_data_less[cat_col] = np.where(price_data_less[cat_col].isin(
                    temp.index), fill_value, price_data_less[cat_col])

        else:
            print("by paramter is not valid")
            return price_data
        # TODO somothing went wrong
        # print(
        #     f"{price_data_less[cat_col].nunique()} {cat_col} left({(price_data_less[cat_col].nunique()/price_data[cat_col].nunique())*100}%)")
        # print(
        #     f"{price_data.shape[0] - price_data_less.shape[0]} rows were deleted({((price_data.shape[0] - price_data_less.shape[0])/price_data.shape[0])*100}%)")
        return price_data_less
