import os

from Researchers.Forwards.ClassForwardOneAsset import find_bonds_hedge, find_gold_hedge, find_aggressive_etf, \
    run_aggressive_etf, spy_or_qqq, find_best_hyper_mom
from Researchers.Forwards.MakeSynteticAsset import make_syntetic_asset, merge_two_assets
from Researchers.SpecialDividends.MainResearch.MakeTrades import run_special_div
from GitHub.PortsForSiteCopy.SMA_and_Mom import SMAandMomentum
from GitHub.PortsForSiteCopy.SMA_and_Mom import start as sma_vix_start
from GitHub.PortsForSite.Target_Vol import TargetVolatility, print_port_shares
from GitHub.PortsForSite.Target_Vol import start as tv_start
from datetime import datetime, timedelta
from APIs.API_Tiingo.multithread import download_tickers as tiingo_downloader
from APIs.API_Brokers.IB import IbAdvisor
from ib_insync import *
from APIs.API_Pandas import pandas_loader
from FinanceAndMl_libs import finance_ml as fm

import numpy as np
import pandas as pd

cur_disc = os.getcwd().split('\\')[0]


def make_hedge() -> None:
    find_bonds_hedge()
    find_gold_hedge(True)
    make_syntetic_asset(
        "VIPSX,VUSTX_ratio_sortino_cagr_mom_week.csv", 'VUSTX', 'VIPSX', 'ratio', 'mom', 'bars', 'week', 'sortino',
        'cagr'
    )
    make_syntetic_asset(
        "DGS3_Close_adj,LBMA_GOLD_ratio_cagr_cagr_mom_month.csv", 'LBMA_GOLD', 'DGS3_Close_adj', 'ratio', 'mom', 'bars',
        'month', 'cagr', 'cagr'
    )
    merge_two_assets(
        "VUSTX_VIPSX_ratio_mom_bars_week_sortino_cagr.csv",
        "LBMA_GOLD_DGS3_Close_adj_ratio_mom_bars_month_cagr_cagr.csv"
    )

    return None


def make_agreess_etf() -> None:
    find_aggressive_etf()
    make_syntetic_asset(
        "VISVX,^NDX_ratio_cagr_cagr_mom_month.csv", '^NDX', 'VISVX', 'ratio', 'mom', 'bars', 'month', 'cagr', 'cagr'
    )

    return None


def run_stocks_strata(update_etf: bool = True) -> None:
    # Получим текущие веса по акциям
    df = pd.read_excel(rf"{cur_disc}\Биржа\Stocks. BigData\Projects\Strategies\data\StocksData.xlsx")
    df = pd.concat([
        df[['Ticker', 'Unnamed: 10']].dropna(),
        df[['Ticker.1', 'Unnamed: 24']].dropna().rename(columns={'Ticker.1': 'Ticker', 'Unnamed: 24': "Unnamed: 10"})
    ]).set_index('Ticker')
    df['Unnamed: 10'] = df['Unnamed: 10'].round(4)
    print(df['Unnamed: 10'].sum())
    print(df.to_dict(orient='dict')['Unnamed: 10'])

    # GrowthValueStocks
    if update_etf:
        tiingo_downloader('etf', ['SPY', 'GLD'], 2)
        tiingo_downloader('mutual', ['VUSTX'], 1)
        fm.download_tickers(['^VIX'], 'yahoo', True)
        pd.read_csv(rf'{cur_disc}\Биржа\Stocks. BigData\Цены\Дейли\yahoo\^VIX.csv', parse_dates=['Date'], index_col='Date')\
            .to_csv(rf"{cur_disc}\Биржа\Stocks. BigData\Цены\Дейли\tos\VIX.csv")

    portfolios = {
        'high_risk':
            {'SPY': 1.0, 'VUSTX': 0},
        'mid_save':
            {'SPY': .7, 'VUSTX': .15, 'GLD': .15},
        # 'intermed_save':
        #     {'SPY': .5, 'VUSTX': .30, 'GLD': .20},
        'high_save':
            {'SPY': .0, 'VUSTX': .6, 'GLD': .4},
    }
    test_port = SMAandMomentum(
        balance_start=100_000,
        portfolios=portfolios,
        rebalance='monthly',
        trade_rebalance_at='close',
        date_start=datetime(1995, 1, 1),
        date_end=datetime(2025, 12, 14),
        signal_stocks='SPY',
        signal_bonds='VUSTX',
        benchmark='SPY',
        sma_period=180,
        forsed_rebalance=True,
        calc_adviser_comm=True,
        comm_for_profit=.2,
        comm_for_enter=0.025,
    )
    df_strategy, df_yield_by_years, chart_name = sma_vix_start(test_port, False)
    from GitHub.PortsForSiteCopy.libs import trading_lib as tl
    from GitHub.PortsForSiteCopy.SMA_and_Mom import calc_benchmark
    tl.plot_capital_plotly(
        test_port.FOLDER_WITH_IMG + chart_name,
        list(df_strategy.index),
        list(df_strategy.Capital),
        calc_benchmark(test_port, list(df_strategy.index)),
        test_port.bench,
        df_yield_by_years,
        portfolios,
        True
    )

    return None


def run_halal_etf(update_etf: bool = True) -> None:
    if update_etf:
        tiingo_downloader('etf', ['SPY', 'SPUS', 'SGOL', 'SPSK'], 4)
        tiingo_downloader('mutual', ['VUSTX'], 1)
        fm.download_tickers(['^VIX'], 'yahoo', True)
        pd.read_csv(rf'{cur_disc}\Биржа\Stocks. BigData\Цены\Дейли\yahoo\^VIX.csv', parse_dates=['Date'], index_col='Date') \
            .to_csv(rf"{cur_disc}\Биржа\Stocks. BigData\Цены\Дейли\tos\VIX.csv")
    # portfolios = {
    #     'high_risk':
    #         {'SPSIEUN': 1.0, 'GOLD_H': 0, 'DJSUKTXR': .0},
    #     'mid_save':
    #         {'SPSIEUN': .8, 'GOLD_H': .1, 'DJSUKTXR': .1},
    #     'intermed_save':
    #         {'SPSIEUN': .5, 'GOLD_H': .0, 'DJSUKTXR': .5},
    #     'high_save':
    #         {'DJSUKTXR': .4, 'GOLD_H': .6}
    # }
    portfolios = {
        'high_risk':
            {'SPUS': 1.0, 'SGOL': 0, 'SPSK': .0},
        'mid_save':
            {'SPUS': .8, 'SGOL': .1, 'SPSK': .1},
        'intermed_save':
            {'SPUS': .5, 'SGOL': .0, 'SPSK': .5},
        'high_save':
            {'SPSK': .4, 'SGOL': .6}
    }
    test_port = SMAandMomentum(
        balance_start=100_000,
        portfolios=portfolios,
        rebalance='monthly',
        trade_rebalance_at='close',
        date_start=datetime(1995, 12, 25),
        date_end=datetime(2025, 12, 14),
        signal_stocks='SPY',
        signal_bonds='VUSTX',
        momentum_stocks=10,
        momentum_bonds=2,
        benchmark='SPY',
        sma_period=180,
        forsed_rebalance=True,
        calc_adviser_comm=True,
        comm_for_profit=.2,
        comm_for_enter=0.025,
    )
    df_strategy, df_yield_by_years, chart_name = sma_vix_start(test_port, False)

    return None


def run_index_follow(update_etf: bool = True):
    if update_etf:
        fm.download_tickers(['QQQ', 'TLT', 'GLD', 'SPY'])
        fm.download_tickers(['^VIX'], 'yahoo', True)
        pd.read_csv(rf'{cur_disc}\Биржа\Stocks. BigData\Цены\Дейли\yahoo\^VIX.csv', parse_dates=['Date'], index_col='Date') \
            .to_csv(rf"{cur_disc}\Биржа\Stocks. BigData\Projects\GitHub\PortsForSite\historical_data\VIX.csv")

    """ Не оправдало себя по доходности 
    from Researchers.TrendFollow import sma_mom
    sma_lens = np.arange(0, 100, 1)
    tickers = ['SPY', 'QQQ', 'DIA']
    sma_mom.make_momentum_rotation(sma_lens, tickers, load_tickers=True, save_file=True)
    """

    portfolios = {
        'risk_on':
            {'QQQ': 0.8, 'SPY': .2},
        'risk_off':
            {'TLT': .6, 'GLD': .4}
    }
    test_port = TargetVolatility(
        portfolios=portfolios,
        balance_start=100_000,
        date_start=datetime(1995, 12, 15),
        date_end=datetime(2022, 3, 1),
        benchmark='SPY',
        vol_target=30,
        rebalance='monthly',
        vol_calc_period='month',
        vol_calc_range=1,
        forsed_rebalance=True,
        use_margin=False,
        calc_adviser_comm=False,
        comm_for_profit=.2,
        comm_for_enter=.025,
        float_target_vol=True,
    )
    df_strategy, df_yield_by_years, chart_name = tv_start(test_port, download=False)
    print_port_shares(df_strategy)

    from GitHub.PortsForSite.libs import trading_lib as tl
    from GitHub.PortsForSite.Target_Vol import calc_benchmark
    tl.plot_capital_plotly(
        test_port.FOLDER_WITH_IMG + chart_name,
        list(df_strategy.Date),
        list(df_strategy.Capital),
        calc_benchmark(test_port, list(df_strategy['Date'])),
        test_port.bench,
        df_yield_by_years,
        portfolios,
        True
    )


def run_val_depo(update_etf: bool = True, vol_target: int = 10, riskon_ticker: str = 'QQQ'):
    if update_etf:
        fm.download_tickers(['QQQ', 'SPY', 'TLT', 'IJS'])

    portfolios = {
        'risk_on':
            {riskon_ticker: 1.0},
        'risk_off':
            {'TLT': 1.0}
    }

    test_port = TargetVolatility(
        portfolios=portfolios,
        balance_start=100_000,
        date_start=datetime(1999, 8, 29),
        date_end=datetime(2023, 3, 23),
        benchmark='SPY',
        vol_target=vol_target,
        rebalance='monthly',
        vol_calc_period='month',
        vol_calc_range=1,
        forsed_rebalance=True,
        use_margin=False,
        calc_adviser_comm=False,
        comm_for_profit=.2,
        comm_for_enter=.025,
    )
    df_strategy, df_yield_by_years, chart_name = tv_start(test_port, download=False)
    print_port_shares(df_strategy)


def run_multiasset():
    # Stocks. BigData/Projects/Researchers/Forwards/AllAssetsRotation.ipynb
    # Подготовим данные
    # print(f"Внимание! PCY - можно заменить на CEMB")
    # find_best_hyper_mom('VFINX', 'PREMX', 'cagr', 'cagr', ['tiingo'], 'month', 'SPY', 'PCY', True)
    # find_best_hyper_mom('VFINX', 'DX-Y.NYB', 'sortino', 'sortino', ['tiingo', 'yahoo'], 'week', 'SPY', 'UUP', True)
    # find_best_hyper_mom('VFINX', '^NDX', 'cagr', 'cagr', ['tiingo', 'yahoo'], 'month', 'SPY', 'QQQ', True)
    # find_best_hyper_mom('VFINX', 'VEIEX', 'cagr', 'cagr', ['tiingo'], 'week', 'SPY', 'EEM', True)
    # find_best_hyper_mom('VFINX', 'VFSTX', 'cagr', 'cagr', ['tiingo'], 'month', 'SPY', 'VCSH', True)
    # print(f"Внимание! Аналог VFISX - 80%SHY, 20%IEI")
    # find_best_hyper_mom('VFINX', 'VFISX', 'cagr', 'cagr', ['tiingo'], 'month', 'SPY', 'SHY', True)
    # find_best_hyper_mom('VFINX', 'VUSTX', 'cagr', 'cagr', ['tiingo'], 'month', 'SPY', 'TLT', True)
    # pandas_loader.run_pandas_datareader('USDPM', data_source='quandl', ticker='GOLD', sourse='LBMA')
    # find_best_hyper_mom('VFINX', 'LBMA_GOLD', 'cagr', 'cagr', ['tiingo'], 'month', 'SPY', 'GLD', True)

    # Получим данные
    # tickers = [
    #     'VFINX_PREMX_ratio_mom_bars_month_cagr_cagr',
    #     'VFINX_VFSTX_ratio_mom_bars_month_cagr_cagr',
    #     'VFINX_VFISX_ratio_mom_bars_month_cagr_cagr',
    #     'VFINX_VUSTX_ratio_mom_bars_month_cagr_cagr',
    #     'VFINX_LBMA_GOLD_ratio_mom_bars_month_cagr_cagr',
    #     'VFINX_^NDX_ratio_mom_bars_month_cagr_cagr',
    #     'VFINX_VEIEX_ratio_mom_bars_week_cagr_cagr',
    #     'VFINX_DX-Y.NYB_ratio_mom_bars_week_sortino_sortino'
    # ]
    tickers = [
        'SPY_PCY_ratio_mom_bars_month_cagr_cagr',
        'SPY_VCSH_ratio_mom_bars_month_cagr_cagr',
        'SPY_SHY_ratio_mom_bars_month_cagr_cagr',
        'SPY_TLT_ratio_mom_bars_month_cagr_cagr',
        'SPY_GLD_ratio_mom_bars_month_cagr_cagr',
        'SPY_QQQ_ratio_mom_bars_month_cagr_cagr',
        'SPY_EEM_ratio_mom_bars_week_cagr_cagr',
        'SPY_UUP_ratio_mom_bars_week_sortino_sortino'
    ]
    bench = 'SPY'  # VFINX
    s_asset = 'EEM'  # VEIEX
    dict_data = fm.get_tickers(tickers + [bench])
    dict_data = fm.make_same_index(dict_data, 'adjClose')

    # Создадим входы
    port_df = pd.DataFrame()
    for key in tickers:
        ticker_name = key.split('_ratio')[0].split(f'{bench}_')[1]
        port_df[f'{ticker_name}_enter'] = (dict_data[key]['cur_ticker'] != bench)
    port_df[f'{bench}_enter'] = np.where(port_df[f'{s_asset}_enter'] == False, True, False)
    port_df = port_df.astype(int)
    port_df['mult'] = 1 / port_df.sum(axis=1)

    # Получим динамику цен
    need_tickers = [col.replace('_enter', '') for col in port_df.columns if '_enter' in col]
    need_data = fm.get_tickers(need_tickers)
    need_data = fm.make_same_index(need_data, 'adjClose')
    for ticker in need_tickers:
        port_df[f'{ticker}_gain'] = need_data[ticker]['adjClose'].shift(-1) / need_data[ticker]['adjClose'] - 1

    # Рассчитаем позиции
    enter_cols = [col for col in port_df.columns if '_enter' in col]
    enter_df = port_df[enter_cols].apply(lambda s: ",".join(s[s == 1].index.tolist()), axis=1)

    cap_gain_cols = []
    for ticker in need_tickers:
        reg_ticker = ticker.replace('^', '')
        idx_mask = enter_df[enter_df.str.contains(f'{reg_ticker}_enter')].index
        port_df.loc[idx_mask, f'cap_gain_{ticker}'] = port_df[f'{ticker}_gain'] * port_df['mult']
        no_idx_mask = set(port_df.index) - set(idx_mask)
        port_df.loc[no_idx_mask, f'cap_gain_{ticker}'] = None
        cap_gain_cols.append(f'cap_gain_{ticker}')

    port_df = port_df['2009-11-23':].copy()
    port_df['mult_gain'] = port_df[cap_gain_cols].sum(axis=1) + 1
    port_df['port'] = np.cumprod(port_df['mult_gain'])

    # График
    # port_df['bh'] = (port_df[pd.Series(enter_cols).str.replace("_enter", '_gain').to_list()] *
    #                  (1 / (len(need_tickers) - 1))).sum(axis=1)
    # port_df['bh'] = np.cumprod(port_df[f'bh'] + 1)
    # port_df['bench'] = np.cumprod(port_df[f'{bench}_gain'] + 1)
    # port_df['dd'] = (port_df['port'] / port_df['port'].expanding().max() - 1) * 100
    # fm.make_chart_with_dd(
    #     port_df,
    #     chart_name=f'{bench}-{s_asset}',
    #     port_col='port',
    #     port_dd_col='dd',
    #     bh_col='bh',
    #     bh_name='bh',
    #     bench_col=f'bench',
    #     bench_name=bench,
    # ).show()
    # years = (port_df.index[-1] - port_df.index[0]).days / 365
    # print((port_df['port'][-1] ** (1 / years) - 1) * 100, port_df['port'].pct_change().std(ddof=1) * np.sqrt(252))
    # port_group = port_df.groupby(pd.Grouper(freq='Y'))
    # print((port_group['port'].last() / port_group['port'].first() - 1) * 100)

    print(port_df[enter_cols + ['mult']].iloc[-10:])

    return None


if __name__ == "__main__":
    # Создание хэджа и запрос по ситуации с бондами и золотом
    make_hedge()

    """ Agress. ETF Strata
    Бонды чекаем каждый понедельник. 
    Золото и риск чекаем раз в месяц. 
    Каждый день чекаем волу за вчера. 
    # """
    # make_agreess_etf()
    # ports = {'risk_on': {'QQQ': .1}, 'risk_off': {'TLT': .1}}
    # run_aggressive_etf(ports)

    """ Stocks. Strata
    Риск-он выше SMA180. 
    70% риска, если между SMA180 и SMA180 * 1.04.
    50% риска - если мы ниже SMA180 и чек по VIX.
    0% в иных ситуациях.
    Сложный хэдж используем - если идёт цикл подъёма ставки. 
    """
    # run_stocks_strata(True)

    """ Halal ETF. Strata
    По СМА+VIX (как стратегия на акциях). Раз в месяц. 
    """
    # run_halal_etf(False)

    """ Следование за индексом. Strata
    Риск-он берём по форвару между SPY и QQQ. Где вес лидера 80%. 

    Если выше сма200 * 1.04, TV 30%.
    Если между сма200 и сма200 * 1.04, TV 30% * 0.5.
    Если ниже сма200 TV 30% * 0.1.
    Сложный хэдж используем - если идёт цикл подъёма ставки.
    """
    # spy_or_qqq()
    # run_index_follow(True)

    """ Валютный депозит. Strata
    Рисковая часть ротация QQQ - IJS. Как у агрессивных ETF. 
    Защитная часть - gold0.4_bonds.
    Ведём по воле. 6%, 8% или 10%. 
    Чек раз в месяц.
    """
    # run_val_depo(vol_target=8, riskon_ticker='QQQ')

    """ Мультиклассовая. Strata
    Порфтель принимает в себя 9 активов.
    100% портфеля может находится или в VFINX, или в VEIEX. В портфеле не может находиться и VFINX, и VEIEX.
    Таким образом максимальное количество позиций - 8. 
    Если один из оставшихся 7 активов имеет лучший форвардный моментум, чем VFINX, то 
        1 / (Количество лучших активов + VFINX/VEIEX) - вес каждого актива в портфеле. 
        
    Доллар и VEIEX чекается раз в неделю. 
    Иные раз в месяц.
    """
    # run_multiasset()

    """ Special Dividends
    Сначала запускается парсер новостей (смотри, чтобы логин и пароль были актуальны). До 9:15 вручную проверяются 
        новости по ссылкам и проставляется значение дивов и ex-div date. Затем, запускается функция ниже с режимом
        action='position', чтобы выставить ордера на открытие. После открытия функция ниже запускается ещё раз с 
        режимом action='stop', чтобы выставить стопы по позициям, которые были сегодня открыты.
    Парсер новостей E:\Биржа\Stocks. BigData\Projects\Researchers\SpecialDividends\MainResearch\04. NewsFinder.py
    
    action = position/orders/cancel_mocs
    """
    # run_special_div(action='orders')

    """ Минимальные просадки. США и развивающиеся.
    VIGRX VEIEX VFITX VFISX UUP
    Momentum Timing: 2Month
    Exclude First: No
    VolaPeriod: 20 days
    Assets to hold: 2
    Allocation Weights: Invers. Vola
    Frequency: Monthly
    CAGR 10.9% Stdev 9.5% DD 13%
    Jan 1995 - Jul 2022
    
    https://www.portfoliovisualizer.com/test-market-timing-model?s=y&coreSatellite=false&timingModel=9&timePeriod=4&startYear=1973&firstMonth=1&endYear=2022&lastMonth=12&calendarAligned=true&includeYTD=false&initialAmount=10000&periodicAdjustment=0&adjustmentAmount=0&inflationAdjusted=true&adjustmentPercentage=0.0&adjustmentFrequency=4&symbols=VIGRX+VEIEX+VFITX+VFISX&singleAbsoluteMomentum=false&volatilityTarget=9.0&downsideVolatility=false&outOfMarketStartMonth=5&outOfMarketEndMonth=10&outOfMarketAssetType=1&movingAverageSignal=1&movingAverageType=1&multipleTimingPeriods=false&periodWeighting=2&windowSize=2&windowSizeInDays=20&movingAverageType2=1&windowSize2=10&windowSizeInDays2=105&excludePreviousMonth=false&normalizeReturns=false&volatilityWindowSize=-1&volatilityWindowSizeInDays=20&assetsToHold=2&allocationWeights=1&riskControlType=0&riskWindowSize=10&riskWindowSizeInDays=0&stopLossMode=0&stopLossThreshold=2.0&stopLossAssetType=1&rebalancePeriod=1&separateSignalAsset=false&tradeExecution=0&leverageType=0&leverageRatio=0.0&debtAmount=0&debtInterest=0.0&maintenanceMargin=25.0&leveragedBenchmark=false&comparedAllocation=0&benchmark=VFINX&timingPeriods%5B0%5D=5&timingUnits%5B0%5D=2&timingWeights%5B0%5D=100&timingUnits%5B1%5D=2&timingWeights%5B1%5D=0&timingUnits%5B2%5D=2&timingWeights%5B2%5D=0&timingUnits%5B3%5D=2&timingWeights%5B3%5D=0&timingUnits%5B4%5D=2&timingWeights%5B4%5D=0&volatilityPeriodUnit=2&volatilityPeriodWeight=0
    """

    """ Минимальные просадки. США.
    QQQ IJS ^GOLD VUSTX UUP
    VolaPeriod: 3 month
    Allocation Weights: RP
    Sharp: 1.23
    
    https://www.portfoliovisualizer.com/test-market-timing-model?s=y&coreSatellite=false&timingModel=9&timePeriod=4&startYear=2008&firstMonth=1&endYear=2022&lastMonth=12&calendarAligned=true&includeYTD=false&initialAmount=10000&periodicAdjustment=0&adjustmentAmount=0&inflationAdjusted=true&adjustmentPercentage=0.0&adjustmentFrequency=4&symbols=QQQ+IJS+%5EGOLD+VUSTX+UUP&singleAbsoluteMomentum=false&volatilityTarget=9.0&downsideVolatility=false&outOfMarketStartMonth=5&outOfMarketEndMonth=10&outOfMarketAssetType=1&movingAverageSignal=1&movingAverageType=1&multipleTimingPeriods=false&periodWeighting=2&windowSize=1&windowSizeInDays=105&movingAverageType2=1&windowSize2=10&windowSizeInDays2=105&excludePreviousMonth=false&normalizeReturns=false&volatilityWindowSize=3&volatilityWindowSizeInDays=20&assetsToHold=5&allocationWeights=4&riskControlType=0&riskWindowSize=10&riskWindowSizeInDays=0&stopLossMode=0&stopLossThreshold=2.0&stopLossAssetType=1&rebalancePeriod=1&separateSignalAsset=false&tradeExecution=0&leverageType=0&leverageRatio=100.0&debtAmount=0&debtInterest=3.0&maintenanceMargin=25.0&leveragedBenchmark=false&comparedAllocation=0&benchmark=-1&benchmarkSymbol=VFINX&timingPeriods%5B0%5D=5&timingUnits%5B0%5D=2&timingWeights%5B0%5D=100&timingUnits%5B1%5D=2&timingWeights%5B1%5D=0&timingUnits%5B2%5D=2&timingWeights%5B2%5D=0&timingUnits%5B3%5D=2&timingWeights%5B3%5D=0&timingUnits%5B4%5D=2&timingWeights%5B4%5D=0&volatilityPeriodUnit=1&volatilityPeriodWeight=0
    """

    """ Облиги во время кризиса. Перейти, когда TIPS прогорят.
    Стратегия. 
    Покупаем корп. облигации, 4-7 лет, когда ставка выросла. 
    Как только появляются намёки на снижение ставки, из-за кризиса, покупаем облиги с плечом 50%, чтобы заработать на 
    росте облигационных цен из-за снижения ставки и выхода из кризиса, а купон должен х2 перекрывать стоимость текущего 
    плеча. 
    По мере подтверждения снижения ставки добираем плечо, вплоть до 300%. 
    
    По мере снижения ставки, купон останется прежним, а стоимость обслуживания плеча уменьшится. 
    Доходность около 6%. В худшем случае инвестор получит её. В лучшем случае, около 10-15% годовых. 
    
    Дополнение - можно в периоды роста рынка сидеть с плечом х2 в "RiskParity US Sectors, Gold, VUSTX".
        Как только SPY заходит за СМА, сбрасывать порт и набирать корп. облигации.
        По мере выхода из кризиса, сбрасывать выросшие облиги и возвращаться в порт. 
    """

    """ Болванка для "RiskParity US Sectors, Gold, VUSTX". UUP?
    Пока SPY выше СМА200 можно держать порт с плечом x1.8. Как только ниже, плечо сбрасываем. 
    
    Примерные расчёты показывают доходность 15-17%, при предельных просадках 20%, с 2000 года. 
    
    XLE XLB XLI XLK XLV XLP XLU XLY XLF ^GOLD VUSTX
    https://www.portfoliovisualizer.com/test-market-timing-model?s=y&coreSatellite=false&timingModel=9&timePeriod=4&startYear=1988&firstMonth=1&endYear=2022&lastMonth=12&calendarAligned=true&includeYTD=false&initialAmount=10000&periodicAdjustment=0&adjustmentAmount=0&inflationAdjusted=true&adjustmentPercentage=0.0&adjustmentFrequency=4&symbols=XLE+XLB+XLI+XLK+XLV+XLP+XLU+XLY+XLF+%5EGOLD+VUSTX&singleAbsoluteMomentum=false&volatilityTarget=9.0&downsideVolatility=false&outOfMarketStartMonth=5&outOfMarketEndMonth=10&outOfMarketAssetType=1&movingAverageSignal=1&movingAverageType=1&multipleTimingPeriods=false&periodWeighting=2&windowSize=1&windowSizeInDays=105&movingAverageType2=1&windowSize2=10&windowSizeInDays2=105&excludePreviousMonth=false&normalizeReturns=false&volatilityWindowSize=3&volatilityWindowSizeInDays=20&assetsToHold=10&allocationWeights=4&riskControlType=0&riskWindowSize=10&riskWindowSizeInDays=0&stopLossMode=0&stopLossThreshold=2.0&stopLossAssetType=1&rebalancePeriod=1&separateSignalAsset=false&tradeExecution=0&leverageType=1&leverageRatio=85&debtAmount=0&debtInterest=4.5&maintenanceMargin=25.0&leveragedBenchmark=false&comparedAllocation=0&benchmark=-1&benchmarkSymbol=VFINX&timingPeriods%5B0%5D=5&timingUnits%5B0%5D=2&timingWeights%5B0%5D=100&timingUnits%5B1%5D=2&timingWeights%5B1%5D=0&timingUnits%5B2%5D=2&timingWeights%5B2%5D=0&timingUnits%5B3%5D=2&timingWeights%5B3%5D=0&timingUnits%5B4%5D=2&timingWeights%5B4%5D=0&volatilityPeriodUnit=1&volatilityPeriodWeight=0
    
    XLE XLB XLI XLK XLV XLP XLU XLY XLF UGL TMF
    https://www.portfoliovisualizer.com/test-market-timing-model?s=y&coreSatellite=false&timingModel=9&timePeriod=4&startYear=2010&firstMonth=1&endYear=2022&lastMonth=12&calendarAligned=true&includeYTD=false&initialAmount=10000&periodicAdjustment=0&adjustmentAmount=0&inflationAdjusted=true&adjustmentPercentage=0.0&adjustmentFrequency=4&symbols=XLE+XLB+XLI+XLK+XLV+XLP+XLU+XLY+XLF+UGL+TMF&singleAbsoluteMomentum=false&volatilityTarget=9.0&downsideVolatility=false&outOfMarketStartMonth=5&outOfMarketEndMonth=10&outOfMarketAssetType=1&movingAverageSignal=1&movingAverageType=1&multipleTimingPeriods=false&periodWeighting=2&windowSize=1&windowSizeInDays=105&movingAverageType2=1&windowSize2=10&windowSizeInDays2=105&excludePreviousMonth=false&normalizeReturns=false&volatilityWindowSize=3&volatilityWindowSizeInDays=20&assetsToHold=10&allocationWeights=4&riskControlType=0&riskWindowSize=10&riskWindowSizeInDays=0&stopLossMode=0&stopLossThreshold=2.0&stopLossAssetType=1&rebalancePeriod=1&separateSignalAsset=false&tradeExecution=0&leverageType=0&leverageRatio=50.0&debtAmount=0&debtInterest=4.5&maintenanceMargin=25.0&leveragedBenchmark=false&comparedAllocation=0&benchmark=-1&benchmarkSymbol=VFINX&timingPeriods%5B0%5D=5&timingUnits%5B0%5D=2&timingWeights%5B0%5D=100&timingUnits%5B1%5D=2&timingWeights%5B1%5D=0&timingUnits%5B2%5D=2&timingWeights%5B2%5D=0&timingUnits%5B3%5D=2&timingWeights%5B3%5D=0&timingUnits%5B4%5D=2&timingWeights%5B4%5D=0&volatilityPeriodUnit=1&volatilityPeriodWeight=0
    
    FSENX FSCSX FSRPX FSDPX FSPHX FIDSX FDFAX FKUTX FSDAX  ^GOLD VUSTX
    https://www.portfoliovisualizer.com/test-market-timing-model?s=y&coreSatellite=false&timingModel=9&timePeriod=4&startYear=1988&firstMonth=1&endYear=2022&lastMonth=12&calendarAligned=true&includeYTD=false&initialAmount=10000&periodicAdjustment=0&adjustmentAmount=0&inflationAdjusted=true&adjustmentPercentage=0.0&adjustmentFrequency=4&symbols=FSENX+FSCSX+FSRPX+FSDPX+FSPHX+FIDSX+FDFAX+FKUTX+FSDAX++%5EGOLD+VUSTX&singleAbsoluteMomentum=false&volatilityTarget=9.0&downsideVolatility=false&outOfMarketStartMonth=5&outOfMarketEndMonth=10&outOfMarketAssetType=1&movingAverageSignal=1&movingAverageType=1&multipleTimingPeriods=false&periodWeighting=2&windowSize=1&windowSizeInDays=105&movingAverageType2=1&windowSize2=10&windowSizeInDays2=105&excludePreviousMonth=false&normalizeReturns=false&volatilityWindowSize=3&volatilityWindowSizeInDays=20&assetsToHold=10&allocationWeights=4&riskControlType=0&riskWindowSize=10&riskWindowSizeInDays=0&stopLossMode=0&stopLossThreshold=2.0&stopLossAssetType=1&rebalancePeriod=1&separateSignalAsset=false&tradeExecution=0&leverageType=1&leverageRatio=50.0&debtAmount=0&debtInterest=4.5&maintenanceMargin=25.0&leveragedBenchmark=false&comparedAllocation=0&benchmark=-1&benchmarkSymbol=VFINX&timingPeriods%5B0%5D=5&timingUnits%5B0%5D=2&timingWeights%5B0%5D=100&timingUnits%5B1%5D=2&timingWeights%5B1%5D=0&timingUnits%5B2%5D=2&timingWeights%5B2%5D=0&timingUnits%5B3%5D=2&timingWeights%5B3%5D=0&timingUnits%5B4%5D=2&timingWeights%5B4%5D=0&volatilityPeriodUnit=1&volatilityPeriodWeight=0
    """
