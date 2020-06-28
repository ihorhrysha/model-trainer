import pandas
from datetime import date, timedelta
import calendar
import random
from countryinfo import CountryInfo
import pycountry


class DataPreprocessor:
    def __init__(self, df):
        self.df = df

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

    def _get_products_wo_price(self, df):
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

    def _filter_orders_wo_price(self, df):
        """ Drop orders with some missing prices
        """
        orders_to_drop = df.set_index(
            'OrderId')[['BasePrice', 'UserPrice']].isna().any(axis=1)
        orders_to_drop = orders_to_drop[orders_to_drop].index.unique()
        return df[~df['OrderId'].isin(orders_to_drop)]

    def _filter_orders_range(self, df, column, range=(0.3, 1.5)):
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

    def _drop_orders_w_negative_values(self, df, column):
        """ Drops orders with any negative value in `column` (e.g. to drop negative discounts)
        """
        orders_to_drop = df[df[column] < 0]['OrderId'].unique()
        return df[~df['OrderId'].isin(orders_to_drop)]

    def _drop_orders_nan_values(self, df, column):
        """ Drops orders with nan value in `column` (e.g. to drop nan discounts)
        """
        orders_to_drop = df[df[column].isna()]['OrderId'].unique()
        return df[~df['OrderId'].isin(orders_to_drop)]

    def _finance_features(self, df):
        df["OrderPrice"] = df["OrderAmount"] / df["OrderQty"]
        df["BaseDiscount"] = (
            df["BasePrice"] - df["OrderPrice"])/df["BasePrice"]
        df["UserDiscount"] = (
            df["UserPrice"] - df["OrderPrice"])/df["UserPrice"]

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
        df["ProfitPerItem"] = df["OrderPrice"] - df["CostPerItem"]
        df["Profit"] = df["ProfitPerItem"]*df["SoldQty"]

        df['SoldCost'].fillna(df['SoldQty']*df["CostPerItem"], inplace=True)
        return df

    def _construct_RFMD(self, df):
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

        df["RFMD"].fillna(0, inplace=True)

        df["RFMD"] = df["RFMD"].astype(int)

        df["RFMD_zeros"] = df["RFMD"].apply(lambda x: str(x).zfill(4))

        #Recency, Frequency, Monetary, Duration
        df["Recency"] = df["RFMD_zeros"].apply(lambda x: x[0:1])
        df["Frequency"] = df["RFMD_zeros"].apply(lambda x: x[1:2])
        df["Monetary"] = df["RFMD_zeros"].apply(lambda x: x[2:3])
        df["Duration"] = df["RFMD_zeros"].apply(lambda x: x[3:4])

        return df

    def _location_features(self, price_data, if_print=True):
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
        regions = [set_info(x, "region")
                   for x in price_data["OrderCountry"].unique()]
        print(f"Total {len(unknown)} country regions were not found")
        print("Total rows of unknown region in the data=",
              price_data[price_data.OrderCountry.isin(unknown)].shape[0])
        unknown = []
        subregions = [set_info(x, "subregion")
                      for x in price_data["OrderCountry"].unique()]
        print(f"Total {len(unknown)} country subregion were not found")
        print("Total rows of unknown subregion in the data=",
              price_data[price_data.OrderCountry.isin(unknown)].shape[0])

        price_data = price_data.merge(pd.DataFrame(regions, columns=[
            "OrderCountry", "CountryRegion"]), on="OrderCountry", how="left")
        price_data = price_data.merge(pd.DataFrame(subregions, columns=[
            "OrderCountry", "CountrySubregion"]), on="OrderCountry", how="left")
        return price_data

    def _time_features(self, df):
        '''
        Create time features from date
        '''
        df["OrderDay"] = df.OrderDate.dt.day
        df["OrderYearDay"] = df.OrderDate.dt.dayofyear
        df["OrderWeekDay"] = df.OrderDate.dt.weekday
        df["OrderWeek"] = df.OrderDate.dt.week
        df["OrderHour"] = df.OrderDate.dt.hour
        df["OrderMonth"] = df.OrderDate.dt.month
        return df

        def _order_price_feature(self, df1):
            """ Create TotalOrderPrice feature (total BasePrice of items in order)
            """
            df = df1.copy()
            # df = df[['OrderId', 'BasePrice', 'OrderQty']].copy()
            df['TotalOrderProductPrice'] = df['BasePrice'] * df['OrderQty']

            right_df = df[['OrderId', 'TotalOrderProductPrice']].\
                groupby('OrderId').\
                sum().\
                reset_index().\
                rename(columns={'TotalOrderProductPrice': 'TotalOrderPrice'})

            if 'TotalOrderPrice' in df.columns:
                df = df.drop(columns='TotalOrderPrice')
            return df.merge(
                right_df,
                how='left',
                on='OrderId'
            )

    def _order_revenue_feature(self, df1):
        """ Create TotalOrderRevenue feature (total OrderPrice of items in order)
        """
        df = df1.copy()
        # df = df[['OrderId', 'BasePrice', 'OrderQty']].copy()
        df['TotalOrderProductRevenue'] = df['OrderPrice'] * df['OrderQty']

        right_df = df[['OrderId', 'TotalOrderProductRevenue']].\
            groupby('OrderId').\
            sum().\
            reset_index().\
            rename(columns={'TotalOrderProductRevenue': 'TotalOrderRevenue'})

        if 'TotalOrderRevenue' in df.columns:
            df = df.drop(columns='TotalOrderRevenue')
        return df.merge(
            right_df,
            how='left',
            on='OrderId'
        )

    def _platform_preparation(self, data):

        data.loc[data['IsExternal'] == True, 'ExternalInternal'] = 'external'
        data.loc[data['IsExternal'] == False, 'ExternalInternal'] = 'internal'

        data['PlatformType'] = data['Region'] + '_' + data['ExternalInternal']

        data.drop('ExternalInternal', axis=1)

        return data

    def _order_cost_feature(self, df1):
        """ Create TotalOrderCost feature (total BasePrice of items in order)
        """
        df = df1.copy()
        # df = df[['OrderId', 'BasePrice', 'OrderQty']].copy()
        df['TotalOrderProductCost'] = df['CostPerItem'] * df['OrderQty']

        right_df = df[['OrderId', 'TotalOrderProductCost']].\
            groupby('OrderId').\
            sum().\
            reset_index().\
            rename(columns={'TotalOrderProductCost': 'TotalOrderCost'})

        if 'TotalOrderCost' in df.columns:
            df = df.drop(columns='TotalOrderCost')
        return df.merge(
            right_df,
            how='left',
            on='OrderId'
        )

    def _status_preparation(self, data):

        # Duplicate/trial from business perspective are not valid values(test, errors, and some other garbage)
        df_filtered = data[data['Status'] != 'Duplicate/trial']

        df_filtered = delete_rare_cat(
            df_filtered, 'Status', 'occurrence', min_occur=0.005, fill_value='Cancelled')

        df_filtered.loc[df_filtered.Status.isna(), 'Status'] = 'Cancelled'

        return df_filtered

    def _segment_preparation(self, data):

        # change rare categories to 'Сегмент не определен' within non nans
        df_filtered = data[~data.Segment.isnull()]
        df_filtered = delete_rare_cat(
            df_filtered, 'Segment', 'occurrence', min_occur=0.005, fill_value='Сегмент не определен')

        # Сегмент не определен == NaN
        df_nan = data[data.Segment.isnull()]
        df_nan['Segment'] = 'Сегмент не определен'

        return pd.concat([df_nan, df_filtered])

    def _nan_cleaner(self, df, threshold=0.7):

        # Dropping columns with missing value rate higher than threshold
        df = df[df.columns[df.isnull().mean() < threshold]]

        # Not sure if needed but can make TransactionDate equal OrderData
        df['TransactionDate'].fillna(df['OrderDate'], inplace=True)

        # Max fill function for categorical columns
        cats_with_nan = ['BrandId', 'Brand', 'PriceTypeId',
                         'UnifiedPriceType', 'ManagerId', 'CategoryId']

        for cat_with_nan in cats_with_nan:
            df[cat_with_nan].fillna(
                df[cat_with_nan].value_counts().idxmax(), inplace=True)

        # Filling all missing values with 0
        df = df.fillna(0)

        return df
