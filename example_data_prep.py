"""Prepare data REDACTED AND ANONYMIZED VERSION

This script reads and preps local raw data, aggregates them, merges
them, and saves them in the local file structure. It is currently
run monthly to model 2 target datasets, as well as was run on a
different set of historical data for comparison purposes. 

This file contains the following functions:

    * create_sales_region - adds column sales_region to the dataframe
    * prep_sales - returns list of prepped units dataframes
    * prep activities - returns list of prepped activities dfs
    * prep initiations - returns list of prepped init dfs
    * agg_sales - returns list of aggregated units dfs
    * agg_initiations - returns list of aggregated init dfs
    * agg_activities - returns list of aggregated activities dfs
    * merge_sales_and_activities - returns list of finalized units dfs
    * merge_initiations_and_activities - returns list of finalized\
        initiation dfs
    * add_product2_dfs - returns list with appended product2 only dfs
    * save_units_aggregations - saves files to 'units' folder
    * save_init_aggregations - saves files to the 'initiations' folder
"""

import pandas as pd

GOOD_SPECIALISMS = ['specialism1', 'specialism2', 'specialism3']
CHANNELS = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5']
PRODUCTS = ['product1', 'product2', 'product3', 'product4']
GRANULARITIES = [[],  # blank means only by product
                 ['team'],
                 ['team', 'sales_region'],  
                 ['specialties_top'],
                 ['team', 'specialties_top', 'sales_region'],
                 ['province'],
                 ['team', 'province', 'sales_region'],
                 ['team', 'specialties_top', 'province', 'sales_region'],
                 ]

INIT_GRANULARITIES = [[],  # blank means only by product
                      ['team'],
                      ['team', 'sales_region'],
                      ['province'],
                      ['province', 'team'],
                      ['province', 'team', 'sales_region']]


def create_sales_region(df):
    """Create new column from provinces & teams for regional teams"""
    df.loc[(df['team'] == '1st team') & (
            (df['province'] < XX) |
            ((df['province'] > XX) & (df['province'] < XX))
    ),
           'sales_region'
    ] = '1st team REDACTED'
    df.loc[(df['team'] == '1st team') & (
            ((df['province'] > XX) & (df['province'] < XX)) |
            ((df['province'] > XX) & (df['province'] < XX))
    ),
           'sales_region'
    ] = '1st team REDACTED'
    # 2nd team is only 1 sales regions
    df.loc[df['team'] == '2nd team', 'sales_region'] = '2nd team national'


def prep_sales(file_name, sheet_name=None, end_date=None):
    """Read and prepare units data.

    Parameters
    ----------
    file_name : str
        The filename within the ./data/0_raw_data directory
    sheet_name : str, optional
        The sheet name if the file is an Excel workbook
    end_date : str
        The final year-month in the dataset, like 202204

    Returns
    -------
    list
        A list of dataframes
    """

    try:
        sales = pd.read_csv(f'data/0_raw_data/{file_name}')
    except UnicodeDecodeError:
        sales = pd.read_excel(f'data/0_raw_data/{file_name}',
                              sheet_name=sheet_name)

    # no missing values
    # sales.isnull().sum().sum() # 0

    # drop single value column
    sales = sales.drop(columns='REDACTED')

    # replace Month with Date
    sales['date'] = pd.to_datetime(sales['Month'].apply(str), format='%Y%m')
    sales = sales.drop(columns=['Month'])

    # reorder columns
    cols = sales.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    sales = sales[cols]

    # make most column names lowercase (unless it contains 'XXX')
    sales.columns = [col.lower() if 'XXX' not in col else col for col in cols]

    # make string columns lowercase
    sales = sales.applymap(lambda s: s.lower() if type(s) == str else s)

    sales = sales.drop(columns='product name REDACTED', errors='ignore')
    # create product_simp that aggregates all product1s and product2s
    # sales['product_simp'] = sales['product'] \
    #     .apply(lambda x: 'product1' if 'product1' in x else ('product2' if 'product2' in x else 'other'))

    create_sales_region(sales)

    # still missing XXX 1st team in province XX
    # sales.loc[sales['team'] == '1st team', 'sales_region'].isnull().sum()
    sales['province'] = 'P' + sales['province'].astype(str)

    ### TOP SPECIALISMS ###
    sales['specialties_top'] = sales['specialism']
    sales['specialties_top'] = sales['specialties_top'].replace({'REDACTED': 'specialism1',
                                                                 'REDACTED': 'specialism2',
                                                                 'REDACTED': 'specialism3',
                                                                 'REDACTED': 'specialism1',
                                                                 'REDACTED': 'specialism1'
                                                                 })

    sales.loc[~sales['specialties_top'].isin(GOOD_SPECIALISMS), 'specialties_top'] = 'other'

    # sales[['product','team','specialties_top']].groupby(['product','team','specialties_top']).size()

    # REDACTED

    sales.to_csv(f'data/1_prepped_data/sales_2020-{end_date}.csv', index=False)
    return sales


def prep_activities(file_name, sheet_name=None, end_date=None):
    """Read and prepare REDACTED activity data.

    Parameters
    ----------
    file_name : str
        The filename within the ./data/0_raw_data directory
    sheet_name : str, optional
        The sheet name if the file is an Excel workbook
    end_date : str
        The final year-month in the dataset, like 202204

    Returns
    -------
    A list of dataframes
    """

    activities = pd.read_excel(f'data/0_raw_data/{file_name}',
                               sheet_name=sheet_name, parse_dates=['date'])

    # drop unneeded columns
    activities = activities.drop(columns=['status', 'account_id', 'firstname', 'lastname', 'created_by'],
                                 errors='ignore')

    # channel3s have nan values for 'created_date', 'REDACTED_month', 'REDACTED_start_date', 'REDACTED'
    # Julien already got dates into the 'month' and 'date' column
    activities = activities.drop(columns=['created_date', 'REDACTED_month', 'REDACTED_start_date', 'REDACTED'],
                                 errors='ignore')

    activities = activities.rename(columns={'date': 'date_day'})

    activities['month'] = pd.to_datetime(activities['month'].apply(str), format='%Y%m')
    activities = activities.rename(columns={'month': 'date'})

    # missing province cannot be imputed, but need to do something bc missing values
    # activities['province'].isnull().sum() # only 108
    activities['province'] = activities['province'].fillna(XX)
    activities['province'] = activities['province'].astype(int)

    # make string columns lowercase
    activities = activities.applymap(lambda s: s.lower() if type(s) == str else s)

    # make columns lowercase
    activities.columns = [col.lower() for col in activities.columns]

    # activities['product_name'] = activities['product_name'].str.replace('_REDACTED', '')

    # REDACTED
    # REDACTED
    # REDACTED
    # REDACTED
    # REDACTED

    # REDACTED
    # REDACTED
    # REDACTED
    # REDACTED

    activities.loc[activities['activity_type'].isin(['REDACTED', 'REDACTED']),
                   'activity_type'
                   ] = 'channel5'

    # drop handful of faxes and samples
    activities = activities.loc[activities['activity_type'].isin(CHANNELS)]

    # add sales regions as done for sales
    create_sales_region(activities)

    activities['province'] = 'P' + activities['province'].astype(str)

    # missing sales regions:
    # activities['sales_region'].isnull().sum() # REDACTED
    # activities.loc[activities['team']=='1st team', 'sales_region'].isnull().sum() # REDACTED

    ### Specialties ###

    # change REDACTED to specialism1 since REDACTED
    # "specialism3" and "REDACTED" are the same, so make both "specialism3"
    activities['specialties_top'] = activities['specialty'].replace({"REDACTED": 'specialism1',
                                                                     'REDACTED': 'specialism3',
                                                                     'REDACTED': 'specialism1',
                                                                     'REDACTED': 'specialism1'})

    activities.loc[~activities['specialties_top'].isin(GOOD_SPECIALISMS), 'specialties_top'] = 'other'


    activities.to_csv(f'data/1_prepped_data/activity_202001-{end_date}.csv', index=False)
    return activities


def prep_initiations(file_name, sheet_name=None, end_date=None):
    """Read and prepare REDACTED initiations data.

    Parameters
    ----------
    file_name : str
        The filename within the ./data/0_raw_data directory
    sheet_name : str, optional
        The sheet name if the file is an Excel workbook
    end_date : str
        The final year-month in the dataset, like 202204

    Returns
    -------
        A list of dataframes
    """

    df_init = pd.read_excel(f'data/0_raw_data/initiations/{file_name}')

    df_init.columns = [col.lower() for col in df_init.columns]

    df_init['month'] = pd.to_datetime(df_init['month'].apply(str), format='%Y%m')
    df_init = df_init.rename(columns={'month': 'date'})
    df_init = df_init.rename(columns={'projected_initiations': 'initiations'})
    good_cols = ['date', 'province', 'specialism', 'classification', 'initiations', 'products_XX']

    # make string columns values lowercase
    df_init = df_init.applymap(lambda s: s.lower() if type(s) == str else s)

    df_init = df_init.loc[df_init['products_XX'].isin(PRODUCTS), good_cols]

    # rename products_XX to products
    df_init = df_init.rename(columns={'products_XX': 'product'})

    # make teams out of specialisms
    df_init['team'] = df_init['specialism']
    df_init['team'] = df_init['team'].replace({'REDACTED': '1st team', 'specialist': '2nd team',
                                               'REDACTED': float('nan')})

    # similar to create_sales_region()
    df_init.loc[(df_init['specialism'] == 'huisarts') & (
            (df_init['province'] < XX) |
            ((df_init['province'] > XX) & (df_init['province'] < XX))
    ),
                'sales_region'
    ] = '1st team REDACTED'
    df_init.loc[(df_init['specialism'] == 'huisarts') & (
            ((df_init['province'] > XX) & (df_init['province'] < XX)) |
            ((df_init['province'] > XX) & (df_init['province'] < XX))
    ),
                'sales_region'
    ] = '1st team REDACTED'
    # 2nd team is only 1 sales regions
    df_init.loc[df_init['specialism'] == 'specialist', 'sales_region'] = '2nd team national'

    df_init['province'] = 'P' + df_init['province'].astype(str)

    df_init.to_csv(f'data/1_prepped_data/initiations_2018-{end_date}.csv', index=False)
    return df_init


def agg_sales(sales):
    """Aggregate the units dataframes by lists in granularities.

    Parameters
    ----------
    sales : list
        A list of dataframes

    Returns
    -------
        A list of dataframes
    """

    # Aggregate sales by product_simp
    # sales = pd.read_csv('data/1_prepped_data/sales_2020-2022XX.csv', parse_dates=['date'])
    dfs_sales = []

    for granule in GRANULARITIES:
        sales_simp = sales.groupby(['date', 'product'] + granule) \
                          .agg({'Units': 'sum', 'CURRENCY': 'sum'}) \
                          .reset_index()

        # Make 3 target lags
        for shift in range(1, 4):
            sales_simp[f'Units_{shift}M'] = sales_simp.groupby(['product'] + granule)['Units'].shift(shift)
        # add mean target last 2 & 3 months
        sales_simp[f'Units(2 month avg)'] = sales_simp[[f'Units_1M', f'Units_2M']].mean(axis=1)
        sales_simp[f'Units(3 month avg)'] = sales_simp[['Units_1M', 'Units_2M', 'Units_3M']] \
            .mean(axis=1)

        # add TimeOnMarket
        sales_simp['TimeOnMarket'] = sales_simp.groupby(['product'] + granule).cumcount()
        sales_simp.loc[sales_simp['product'] == 'product1', 'TimeOnMarket'] += XX
        sales_simp.loc[sales_simp['product'] == 'product2', 'TimeOnMarket'] -= XX
        sales_simp.loc[sales_simp['product'] == 'product3', 'TimeOnMarket'] += XX
        sales_simp.loc[sales_simp['product'] == 'product4', 'TimeOnMarket'] += XX

        dfs_sales.append(sales_simp)
        # check
        # [display(df.head()) for df in dfs_sales]

        # add Covid lockdowns

    return dfs_sales


def agg_initiations(initiations):
    """Aggregate init dataframes by lists within INIT_GRANULARITIES.

    Parameters
    ----------
    initiations : list
        A list of dataframes

    Returns
    -------
        A list of dataframes
    """

    dfs_init = []

    for granule in INIT_GRANULARITIES:
        init_simp = initiations.groupby(['date', 'product'] + granule) \
                          .agg({'initiations': 'sum'}) \
                          .reset_index()

        # Make 3 target lags
        for shift in range(1, 4):
            init_simp[f'initiations_{shift}M'] = init_simp.groupby(['product'] + granule)['initiations'].shift(shift)
        # add mean target last 2 & 3 months
        init_simp[f'initiations(2 month avg)'] = init_simp[[f'initiations_1M', f'initiations_2M']].mean(axis=1)
        init_simp[f'initiations(3 month avg)'] = init_simp[['initiations_1M', 'initiations_2M', 'initiations_3M']].mean(axis=1)

        # add TimeOnMarket
        init_simp['TimeOnMarket'] = init_simp.groupby(['product'] + granule).cumcount()
        # init_simp.loc[init_simp['product'] == 'product1', 'TimeOnMarket'] += XX  # REDACTED
        init_simp.loc[init_simp['product'] == 'product2', 'TimeOnMarket'] -= XX  # REDACTED
        # init_simp.loc[init_simp['product'] == 'product3', 'TimeOnMarket'] += 24  # REDACTED
        init_simp.loc[init_simp['product'] == 'product4', 'TimeOnMarket'] += (XX-XX)

        dfs_init.append(init_simp)

    return dfs_init


def agg_activities(activities, granularities):
    """Aggregate activity dfs by lists within the right granularities.

    Parameters
    ----------
    activities : list
        A list of dataframes
    granularities : list
        The predefined `GRANULARITIES` or `INIT_GRANULARITIES`

    Returns
    -------
        A list of dataframes
    """

    # Aggregate activity table to be monthly and create lags
    # activities = pd.read_csv('data/1_prepped_data/activity_202001-2022XXXX.csv', parse_dates=['month', 'date'])
    # monthly activity by 1st/2nd team, team-region, top specialties, team-region & province, province
    dfs_act = []

    for granule in GRANULARITIES:

        act_monthly = activities.groupby(['product'] + granule + ['date', 'activity_type']) \
            .agg('count') \
            .reset_index()

        act_monthly = act_monthly.loc[act_monthly['product'].isin(PRODUCTS)]

        # drop non-aggregation features except for 'date' to have something to count
        act_monthly = act_monthly.drop(columns=['onekeyid', 'owner', 'call_id', 'specialty'], errors='ignore')

        # rename date to what it is, a count of activities
        act_monthly = act_monthly.rename(columns={'date_day': 'count_activities'})

        act_monthly = act_monthly.pivot_table(index=['date', 'product'] + granule,
                                              columns=['activity_type'],
                                              values='count_activities',
                                              fill_value=0,
                                              dropna=False) \
            .reset_index() \
            .rename_axis(None, axis=1)

        # good way to check:
        # print(act_monthly.shape)

        # make activity lags
        for channel in CHANNELS:
            # make shifts for each product_name team
            for shift in range(1, 5):
                act_monthly[f'{channel}_{shift}M'] = act_monthly.groupby(['product'] + granule)[channel].shift(shift)
            # take average
            act_monthly[f'{channel}_2M_avg'] = act_monthly[[f'{channel}_1M', f'{channel}_2M']] \
                .mean(axis=1)
            act_monthly[f'{channel}_3M_avg'] = act_monthly[[f'{channel}_1M', f'{channel}_2M', f'{channel}_3M']] \
                .mean(axis=1)
            act_monthly[f'{channel}_4M_avg'] = act_monthly[[f'{channel}_1M', f'{channel}_2M',
                                                            f'{channel}_3M', f'{channel}_4M']] \
                .mean(axis=1)
            act_monthly[f'{channel}_2-3M_avg'] = act_monthly[[f'{channel}_2M', f'{channel}_3M']] \
                .mean(axis=1)
            act_monthly[f'{channel}_2-4M_avg'] = act_monthly[[f'{channel}_2M', f'{channel}_3M', f'{channel}_4M']] \
                .mean(axis=1)
            act_monthly[f'{channel}_3-4M_avg'] = act_monthly[[f'{channel}_3M', f'{channel}_4M']] \
                .mean(axis=1)

        # fill missing activities with 0
        # act_monthly.loc[:, CHANNELS] = act_monthly.loc[:, CHANNELS].fillna(0)
        # fill all missing activities with 0
        act_monthly = act_monthly.fillna(0)

        dfs_act.append(act_monthly)

    return dfs_act


def merge_sales_and_activities(dfs_sales, dfs_act):
    """Merge units and activity dataframes, add totals, and impute.

    Parameters
    ----------
    dfs_sales : list
        A list of units dataframes
    dfs_act : list
        A list of activity dataframes

    Returns
    -------
        A list of dataframes
    """

    # Merge sales and activity
    dfs_merged = []
    for i in range(len(GRANULARITIES)):
        sales_merged = pd.merge(dfs_sales[i],
                                dfs_act[i].rename(columns={'month': 'date'}),
                                how='left',
                                on=['date', 'product'] + GRANULARITIES[i])
        # total activities (cumulative)
        for channel in CHANNELS:
            sales_merged[f'{channel}_total'] = sales_merged.groupby(['product']+GRANULARITIES[i])[f'{channel}_1M'] \
                .cumsum()
            sales_merged[f'{channel}_total'] = sales_merged.groupby(['product']+GRANULARITIES[i])[f'{channel}_total'] \
                .fillna(method='ffill')
        sales_merged.loc[:, CHANNELS] = sales_merged.loc[:, CHANNELS].fillna(0)
        units_columns = sales_merged.columns[sales_merged.columns.str.startswith('Units')].to_list()
        sales_merged.loc[:, units_columns] = sales_merged.loc[:, units_columns].fillna(0)
        dfs_merged.append(sales_merged)

    # [display(df.head()) for df in dfs_merged]

    return dfs_merged


def merge_initiations_and_activities(dfs_init, dfs_act):
    """Merge inits and activity dataframes, add totals, and impute.

    TODO combine with merge_sales_and_activities()

    Parameters
    ----------
    dfs_init : list
        A list of initiation dataframes
    dfs_act : list
        A list of activity dataframes

    Returns
    -------
        A list of dataframes
    """
    dfs_merged = []
    for i in range(len(INIT_GRANULARITIES)):
        df_merged = pd.merge(dfs_init[i],
                             dfs_act[i],  # .rename(columns={'month': 'date'}),
                             how='left',
                             on=['date', 'product'] + INIT_GRANULARITIES[i])
        # total activities (cumulative)
        for channel in CHANNELS:
            df_merged[f'{channel}_total'] = df_merged.groupby(['product'] + INIT_GRANULARITIES[i])[f'{channel}_1M'] \
                .cumsum()
            df_merged[f'{channel}_total'] = df_merged.groupby(['product'] + INIT_GRANULARITIES[i])[f'{channel}_total'] \
                .fillna(method='ffill')
        activity_columns = df_merged.columns[df_merged.columns.str.startswith(tuple(CHANNELS))].to_list()
        initiation_columns = df_merged.columns[df_merged.columns.str.startswith('initiations')].to_list()
        cols_to_impute = initiation_columns + activity_columns
        df_merged.loc[:, cols_to_impute] = df_merged.loc[:, cols_to_impute].fillna(0)
        dfs_merged.append(df_merged)

    # [print(df.head()) for df in dfs_merged]

    return dfs_merged


def add_product2_dfs(dfs_merged):
    """Add versions of final dfs filtered by product2 for convenience"""
    for i in range(len(GRANULARITIES)):
        dfs_merged.append(dfs_merged[i].loc[dfs_merged[i]['product'] == 'product2'])
    return dfs_merged


def save_units_aggregations(dfs_merged, save_folder='units'):
    """Save units sales files in ./data/2_aggregated_data

    Parameters
    ----------
    dfs_merged : list
        List of finalized dataframes. Must match length of file_names
    save_folder : str
        'units', 'initiations', or 'units_product1_historical'
    """

    file_names = [
        '0_units_prods.csv',
        '1_units_prods_team.csv',
        '2_units_prods_team_region.csv',
        '3_units_prods_spec.csv',
        '4_units_prods_team_spec_region.csv',
        '5_units_prods_province_region.csv',
        '6_units_prods_team_province_region.csv',
        '7_units_prods_team_spec_region_province.csv']
    if len(dfs_merged) == 16:
        file_names += ['0_units_product2.csv',
                       '1_units_product2_team.csv',
                       '2_units_product2_team_region.csv',
                       '3_units_product2_spec.csv',
                       '4_units_product2_team_spec_region.csv',
                       '5_units_product2_province_region.csv',
                       '6_units_product2_team_province_region.csv',
                       '7_units_product2_team_spec_region_province.csv'
                       ]
    for i in range(len(file_names)):
        dfs_merged[i].to_csv(f'data/2_aggregated_data/{save_folder}/' + file_names[i], index=False)


def save_init_aggregations(dfs_merged):
    """Save initiation files in ./data/2_aggregated_data

    Parameters
    ----------
    dfs_merged : list
        List of finalized dataframes. Must match length of file_names
    """

    file_names = [
        '0_init_prods.csv',
        '1_init_prods_team.csv',
        '2_init_prods_team_region.csv',
        '3_init_prods_province.csv',
        '4_init_prods_province_team.csv',
        '5_init_prods_province_team_region.csv',
        '0_init_product2.csv',
        '1_init_product2_team.csv',
        '2_init_product2_team_region.csv',
        '3_init_product2_province.csv',
        '4_init_product2_province_team.csv',
        '5_init_product2_province_team_region.csv',
    ]
    for i in range(len(file_names)):
        dfs_merged[i].loc[dfs_merged[i]['date'] >= '2020'] \
            .to_csv('data/2_aggregated_data/initiations/' + file_names[i], index=False)

# def clean_series():
    # drop tiny series

    ### TS data prep ###
    # series start and end:
    # act_monthly.groupby(['product_name', 'team']).agg({'month':['min','max']})


if __name__ == '__main__':
    import sys
    # units pipeline parameters:
    # "units" "REDACTED.xlsx" "sales"
    # "REDACTED.xlsx" "activities" "units"
    if sys.argv[1] == 'units':
        sales = prep_sales(file_name=sys.argv[2], sheet_name=sys.argv[3])
        activities = prep_activities(file_name=sys.argv[4], sheet_name=sys.argv[5])
        dfs_sales = agg_sales(sales)
        dfs_act_sales = agg_activities(activities, granularities=GRANULARITIES)
        dfs_merged_sales = merge_sales_and_activities(dfs_sales, dfs_act_sales)
        if sys.argv[6] == 'units':
            dfs_merged_sales = add_product2_dfs(dfs_merged_sales)
        save_units_aggregations(dfs_merged_sales, save_folder=sys.argv[6])

    # initiations pipeline parameters:
    # "initiations" "REDACTED.xlsx" "None" "REDACTED.xlsx" "activities"
    elif sys.argv[1] == 'initiations':
        initiations = prep_initiations(file_name=sys.argv[2])
        activities = prep_activities(file_name=sys.argv[4], sheet_name=sys.argv[5])
        dfs_act_inits = agg_activities(activities, granularities=INIT_GRANULARITIES)
        dfs_init = agg_initiations(initiations)
        dfs_merged_inits = merge_initiations_and_activities(dfs_init, dfs_act_inits)
        dfs_merged_inits = add_product2_dfs(dfs_merged_inits)
        save_init_aggregations(dfs_merged_inits)

    # product1 historical scenario pipeline parameters:
    # "units" "REDACTED.xlsx" "sales"
    # "REDACTED.xlsx" "activities" "units_product1_historical"
