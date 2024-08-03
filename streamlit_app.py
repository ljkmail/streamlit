#%%
# =============================================================================
#                  강우예측자료를 활용한 제주도 지하수위 예측 서비스
# =============================================================================

# 분석할 지하수관측소 강우강측소 선정
import pandas as pd
import keyring
# 지하수위 관측소(관측소코드)
# 제주노형 95534, 제주동홍 95535, 제주조천 95537, 제주한경 95536
# 강우(asos) 관측소(지점코드)
# 제주 184, 서귀포 189, 성산 188, 고산 185

dic_static = dict(gw = [95534, 95535, 95537, 95536], rain = [184, 189, 188, 185])
#%% keyring
keyring.set_password(
    service_name = "apis.data.go.kr", 
    username = "ljkmail4", 
    password = "0t3eQ20DNg7U9+IcznP4R0jrLlOHQVtGAbmM1WlDdNTRqdpuDcmQpgz5s8cr/24U1F48RT0jgcVTw7kGoCPyAg=="
)
keyring.set_password(
    service_name = "apihub.kma.go.kr",
username = "ljkmail4",
password = "_NL4EYPlSCOS-BGD5VgjvQ",
)
keyring.set_password(
    service_name = "www.gims.go.kr",
username = "ljkmail4",
password = 'L0eM%2BBjtmlHYcF5I5lOvfJ%2F0TBcPs11GMO1P0ENzjkoPaJQMrx3pVqPQ%2FXNBWYBQ'
)

# %% streamlit
import streamlit as st
import streamlit.components.v1 as components

st.markdown("<h3 style='text-align: center;'>지하수 수위 예측(제주도)</h3>", unsafe_allow_html=True)
with st.sidebar:
    st.header("관측정설정")
    gennum_options = ['제주노형', '제주동홍', '제주조천', '제주한경']
    selected_gennum = st.selectbox(label="관측정 이름을 선택하세요",
                                    options=gennum_options)  
    run = st.button('분석시작!')
    progress_bar = st.progress(0, text="분석대기중입니다.")

gennum_to_file = {
    '제주노형': './output_streamlit/graph_제주노형.html',
    '제주동홍': './output_streamlit/graph_제주동홍.html',
    '제주조천': './output_streamlit/graph_제주조천.html',
    '제주한경': './output_streamlit/graph_제주한경.html'
}

html_path = gennum_to_file.get(selected_gennum, None)

if html_path:
    try:
        with open(f'./output_streamlit/graph_{selected_gennum}.html', 'r', encoding='utf-8') as html_file:
            source_code = html_file.read()
            components.html(source_code, height=400)
    except FileNotFoundError:
            st.error(f"{selected_gennum}의 그래프 파일을 찾을 수 없습니다.")
else:
     st.info("관측정을 선택하고 '분석시작!' 버튼을 눌러주세요.")        

if run:   
    #%% 강우 예측 자료 불러오기
    from datetime import datetime
    import json
    import requests
    import io
    from tqdm import tqdm

    # 제주 강우관측소 좌표(rain_location_grid.xlsx에서 찾음(지하수관측소 기준))
    # 제주 asos 기상관측소 station번호(240601_asos_staion.csv 에서 찾음)
    dic_nx_ny_rain = dict(stn = [184, 189, 188, 185], 
                        nx = [52, 53, 55, 46], 
                        ny = [38, 33, 39, 35],
                        name = ['제주', '서귀포', '성산포', '고산'])

    df_rain_future = pd.DataFrame()

    key1 = keyring.get_password("apis.data.go.kr", "ljkmail4")
    today = datetime.today().strftime("%Y%m%d")

    for i in tqdm(range(0, len(dic_nx_ny_rain)), desc='강우 예측자료 읽는중'):

        progress_bar.progress(i / len(dic_nx_ny_rain), 
                              text = "강우 예측자료를 읽어오는 중입니다(1/4).")

        url_future = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
        params ={'serviceKey' : key1, 'pageNo' : '4', 'numOfRows' : '1000', 
                'dataType' : 'JSON', 'base_date' : today, 'base_time' : "0500", 
                'nx' : dic_nx_ny_rain['nx'][i], 'ny' : dic_nx_ny_rain['ny'][i] }
        
        # 'fcstValue' 열에서 '강수없음'을 숫자로 바꿀 수 없으면 0으로 대체하는 코드
        def replace_non_numeric(x):
            try:
                return float(x)
            except ValueError:
                return 0
        
        dic_rain_future_temp = json.loads(requests.get(url_future, params=params).text)["response"]
        df_rain_future_temp = pd.DataFrame(dic_rain_future_temp["body"]["items"]["item"])
        df_rain_future_temp = df_rain_future_temp[df_rain_future_temp["category"] == "PCP"]
        df_rain_future_temp = df_rain_future_temp[['fcstDate', 'fcstTime', 'fcstValue']]
        df_rain_future_temp['YYMMDDHHMI'] = df_rain_future_temp['fcstDate'] + df_rain_future_temp['fcstTime']
        df_rain_future_temp['fcstValue'] = df_rain_future_temp['fcstValue'].str.replace('mm', '')
        df_rain_future_temp['fcstValue'] = df_rain_future_temp['fcstValue'].apply(replace_non_numeric)
        df_rain_future_temp = df_rain_future_temp[['YYMMDDHHMI', 'fcstValue']]
        df_rain_future_temp['stn'] = dic_nx_ny_rain['stn'][i]
        df_rain_future_temp['name'] = dic_nx_ny_rain['name'][i]
        df_rain_future_temp.rename(columns = {'fcstValue' : 'RN'}, inplace = True)
        df_rain_future_temp['RN'] = pd.to_numeric(df_rain_future_temp['RN'])
        df_rain_future_temp['YYMMDDHHMI'] = pd.to_datetime(df_rain_future_temp['YYMMDDHHMI'])
        
        df_rain_future = pd.concat([df_rain_future, df_rain_future_temp])
    
    #%% 오늘 현재시간까지의 시간 강우자료 불러오기

    key2 = keyring.get_password("apihub.kma.go.kr", "ljkmail4")

    tm1 = datetime.today().strftime('%Y%m%d0000')
    tm2 = datetime.today().strftime('%Y%m%d%H00')

    df_rain_current = pd.DataFrame()

    for i in tqdm(range(0, len(dic_nx_ny_rain)), desc='오늘 강우자료 읽는중'):

        progress_bar.progress(i / len(dic_nx_ny_rain), 
                              text = "오늘 강우자료를 읽어오는 중입니다(2/4).")
        
        url_today = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php'
        params ={'tm1' : tm1, 'tm2' : tm2, 'stn' : dic_nx_ny_rain['stn'][i], 'authKey' : key2, 
        'help' : '0', 'obs' : 'RN'}
        
        df_rain_current_temp = pd.read_csv(io.StringIO(requests.get(url_today, params=params).text.replace('#', '')), 
        skiprows = 2, sep = '\s+')
        df_rain_current_temp = df_rain_current_temp.drop([0, len(df_rain_current_temp) - 1])
        df_rain_current_temp = df_rain_current_temp[['YYMMDDHHMI', 'RN']]
        df_rain_current_temp['stn'] = dic_nx_ny_rain['stn'][i]
        df_rain_current_temp['name'] = dic_nx_ny_rain['name'][i]
        df_rain_current_temp['RN'] = pd.to_numeric(df_rain_current_temp['RN'])
        df_rain_current_temp['RN'].replace(-9.0, 0, inplace = True)
        df_rain_current_temp['YYMMDDHHMI'] = pd.to_datetime(df_rain_current_temp['YYMMDDHHMI'])
        
        df_rain_current = pd.concat([df_rain_current, df_rain_current_temp])
        
    # 현재와 미래 강우 합치기
    df_rain_future_edit1 = df_rain_future[df_rain_future['YYMMDDHHMI'] > df_rain_current['YYMMDDHHMI'].max()]
    df_rain_future_edit2 = pd.concat([df_rain_current, df_rain_future_edit1]).set_index('YYMMDDHHMI')
    df_rain_future_edit3 = pd.pivot(df_rain_future_edit2, columns = 'name',  values = ['RN'])   

    df_rain_future_currnet = df_rain_future_edit3.resample('1D').sum()
    df_rain_future_currnet.reset_index(inplace = True)
    df_rain_future_currnet.columns = df_rain_future_currnet.columns.droplevel(level=1)

    df_rain_future_currnet.columns = ['YMD', '185', '189', '188', '184']


    #%% 과거~어제까지 일 강우자료 불러오기
    from datetime import timedelta

    key2 = keyring.get_password("apihub.kma.go.kr", "ljkmail4")
    url_rain_day = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php'

    # 1995년부터 2024.5.31까지 자료 다운로드

    # df_rain_past = pd.date_range(start='1995-01-01', end='2024-05-31')
    # df_rain_past = pd.DataFrame(df_rain_past)
    # df_rain_past.rename(columns={0 : 'YMD'}, inplace=True)


    # for i in tqdm(range(0, len(dic_nx_ny_rain))):
    #     params_rain_day1 = {'tm1' : '19941231', 'tm2' : '20240531', 
    #                         'stn' : dic_nx_ny_rain['stn'][i], 
    #                         'authKey' : key2, 'help' : '0', 'obs' : 'RN'}
        
    #     df_rain_past_temp1 = pd.read_csv(
    #         io.StringIO(requests.get(url_rain_day, 
    #                                   params=params_rain_day1).text.replace('#', '')), 
    #         skiprows = 2, sep = '\s+')
    #     df_rain_past_temp1 = df_rain_past_temp1[['YYMMDD', 'RN']][3:]
    #     df_rain_past_temp1 = df_rain_past_temp1[:len(df_rain_past_temp1)-1]
    #     df_rain_past_temp1['YYMMDD'] = pd.to_datetime(df_rain_past_temp1['YYMMDD'])
    #     df_rain_past_temp1['RN'] = pd.to_numeric(df_rain_past_temp1['RN'])
    #     df_rain_past_temp1['RN'].replace(-9.0, 0, inplace = True)
    #     df_rain_past_temp1['RN'].fillna(0, inplace = True)
    #     df_rain_past_temp1.rename(columns = {'RN' : dic_nx_ny_rain['stn'][i]}, inplace =True)
    #     df_rain_past = pd.merge(left=df_rain_past, right=df_rain_past_temp1, how='left',
    #                             left_on='YMD', right_on='YYMMDD').drop(columns = ['YYMMDD'])

    # df_rain_past.to_csv('./input/df_rain_past.csv', index=False)

    # 위의 코드 저장된 자료
    df_rain_past = pd.read_csv('./input/df_rain_past.csv')
    df_rain_past['YMD'] = pd.to_datetime(df_rain_past['YMD'])

    # df_rain_past 이후부터 어제까지 자료 가져오기

    df_rain_past2 = pd.date_range(start=(pd.to_datetime(df_rain_past['YMD'].max()) + timedelta(days = 1)).strftime('%Y-%m-%d'),
                                end=(datetime.today() - timedelta(days = 1)).strftime('%Y-%m-%d'))
    df_rain_past2 = pd.DataFrame({'YMD' : df_rain_past2})

    for i in tqdm(range(0, len(dic_nx_ny_rain)), desc='과거 강우 읽는중'):

        progress_bar.progress(i / len(dic_nx_ny_rain), 
                              text = "과거 강우자료를 읽어오는 중입니다(3/4).")
        
        params_rain_day1 = {'tm1' : (pd.to_datetime(df_rain_past['YMD'].max())).strftime('%Y%m%d'), 
                            'tm2' : (datetime.today() - timedelta(days = 1)).strftime('%Y%m%d'), 
                            'stn' : dic_nx_ny_rain['stn'][i], 
                            'authKey' : key2, 'help' : '0', 'obs' : 'RN'}
        
        df_rain_past_temp1 = pd.read_csv(
            io.StringIO(requests.get(url_rain_day, 
                                    params=params_rain_day1).text.replace('#', '')), 
            skiprows = 2, sep = '\s+')
        df_rain_past_temp1 = df_rain_past_temp1[['YYMMDD', 'RN']][3:]
        df_rain_past_temp1 = df_rain_past_temp1[:len(df_rain_past_temp1)-1]
        df_rain_past_temp1['YYMMDD'] = pd.to_datetime(df_rain_past_temp1['YYMMDD'])
        df_rain_past_temp1['RN'] = pd.to_numeric(df_rain_past_temp1['RN'])
        df_rain_past_temp1['RN'].replace(-9.0, 0, inplace = True)
        df_rain_past_temp1['RN'].fillna(0, inplace = True)
        df_rain_past_temp1.rename(columns = {'RN' : dic_nx_ny_rain['stn'][i]}, inplace =True)
        df_rain_past2 = pd.merge(left=df_rain_past2, right=df_rain_past_temp1, how='left',
                                left_on='YMD', right_on='YYMMDD').drop(columns = ['YYMMDD'])

    #%% 과거 현재 미래 강우량 합치기
    df_rain_past2.columns = df_rain_past.columns

    df_rain = pd.concat([df_rain_past, df_rain_past2, df_rain_future_currnet])

    df_rain.set_index('YMD').plot()

    #%% 지하수 자료 불러오고 과거자료와 합치기

    df_gw_past_raw = pd.read_csv('./input/gw_yunbo.csv')

    # 제주노형 95534, 제주동홍 95535, 제주조천 95537, 제주한경 95536
    df_gw_past1 = df_gw_past_raw[df_gw_past_raw['GENNUM'].isin([95534, 95535, 95537, 95536])].set_index('YMD')
    df_gw_past2 = pd.pivot(df_gw_past1, columns='GENNUM', values=['ELEV'])
    df_gw_past2.columns = df_gw_past2.columns.droplevel(level = 0)
    df_gw_past2['YMD'] = df_gw_past2.index
    df_gw_past2['YMD'] = pd.to_datetime(df_gw_past2['YMD'])
    df_gw_past2 = df_gw_past2.reset_index(drop =True)

    df_date = pd.date_range(start=df_gw_past2['YMD'].min(), end=df_gw_past2['YMD'].max())
    df_date = pd.DataFrame({'YMD' : df_date})
    df_gw_past3 = pd.merge(left = df_date, right = df_gw_past2, on = 'YMD')
    df_gw_past3 = df_gw_past3.interpolate(method = 'linear')

    # gims api에서 연보자료 이후부터 현재까지 지하수자료 불러오기

    df_gw_current = pd.date_range(start = df_gw_past3['YMD'].max() + timedelta(days = 1),
                                end = datetime.today())
    df_gw_current = pd.DataFrame({'YMD' : df_gw_current})

    key_gw = keyring.get_password("www.gims.go.kr", "ljkmail4")

    begindate = (df_gw_past3['YMD'].max() + timedelta(days=1)).strftime('%Y%m%d')
    enddate = datetime.today().strftime('%Y%m%d')

    url_gw = 'http://www.gims.go.kr/api/data/observationStationService/getGroundwaterMonitoringNetwork?KEY=' + key_gw + '&type=JSON'

    for i in tqdm(range(0, len(dic_static['gw'])), desc='과거 지하수자료 읽는중'):

        progress_bar.progress(i / len(dic_static['gw']), 
                              text = "과거 지하수자료를 읽어오는 중입니다(4/4).")

        params_gw = {'gennum' : str(dic_static['gw'][i]), 'begindate' : begindate, 'enddate' : enddate}
        
        df_gw_api = pd.DataFrame(json.loads(requests.get(url_gw, params=params_gw).text)['response']['resultData'])
        
        df_gw_api_temp = df_gw_api[['ymd', 'gennum', 'elev']].copy()
        df_gw_api_temp.rename(columns = {'ymd' : 'YMD', 'gennum' : 'GENNUM', 'elev' : 'ELEV'}, inplace = True)
        df_gw_api_temp['YMD'] = pd.to_datetime(df_gw_api_temp['YMD'])
        df_gw_api_temp['ELEV'] = pd.to_numeric(df_gw_api_temp['ELEV'])
        df_gw_api_temp = df_gw_api_temp[['YMD', 'ELEV']]
        
        df_gw_current = pd.merge(left=df_gw_current, right=df_gw_api_temp, on = 'YMD', how = 'left')
        df_gw_current['ELEV'] = df_gw_current['ELEV'].interpolate(method = 'linear')
        df_gw_current.rename(columns = {'ELEV' : dic_static['gw'][i]}, inplace = True)
        
    # 제주동홍(95535)는 23.8.30~23.8.31 1.56m가 내려갔고 23.10.24~23.10.25 1.48m 올라간 것 수정
    date_start = pd.to_datetime('2023-08-31')
    date_end = pd.to_datetime('2023-10-24')

    df_gw_current.loc[(df_gw_current['YMD'] >= date_start) & 
                    (df_gw_current['YMD'] <= date_end), 95535] += 1.56

    df_gw = pd.concat([df_gw_past3, df_gw_current]).dropna()

    df_gw.set_index('YMD').plot()

    #%% 지하수자료와 강우자료 합치기, 시계열자료 정규화, 모형훈련 함수 생성, 그래프 함수 생성
    from darts import TimeSeries
    from darts.utils.timeseries_generation import datetime_attribute_timeseries
    from darts.dataprocessing.transformers import Scaler
    from darts.models import TFTModel
    from darts.metrics import mae
    import plotly.graph_objects as go
    import plotly.io as pio
    import numpy as np
    import os
    import torch

    # 모형훈련 함수 생성
    # gw_code에는 GENNUM, rain_code에는 asos 번호, learn_date는 모형학습 시작일자, 
    # n_epochs는 학습횟수, predict_period은 예측 기간(일단위)
    # model_save을 True로 하면 모델을 model_tft_gw_{gw_code}.pt로 저장, False로 하면 저장한 파일 읽어옴
    def fun_model(gw_code, rain_code, learn_date, n_epochs, predict_period, model_save):
        # 적정한 모형 생성을 위해 학습기간을 설정
        df_rain_sliced  = df_rain[df_rain['YMD'] >= pd.to_datetime(learn_date)]
        df_gw_rain = pd.merge(df_rain_sliced, df_gw, on = 'YMD', how='left')
        ts_gw_raw = TimeSeries.from_dataframe(df_gw_rain[df_gw_rain['YMD'] <= datetime.today()], 
                                            'YMD', [gw_code])
        
        # 스케일링
        scaler_gw = Scaler()
        scaler_gw.fit(ts_gw_raw)
        ts_gw_scaled = scaler_gw.transform(ts_gw_raw)
        
        ts_covariates = TimeSeries.from_dataframe(df_gw_rain, 'YMD', str(rain_code))
        ts_covariates_editd1 = ts_covariates.stack(
            datetime_attribute_timeseries(ts_covariates, 'year')).stack(
                datetime_attribute_timeseries(ts_covariates, 'month')).stack(
                    datetime_attribute_timeseries(ts_covariates, 'day'))
        
        scaler_covariates = Scaler()
        scaler_covariates.fit(ts_covariates_editd1)
        ts_covariates_scaled = scaler_covariates.transform(ts_covariates_editd1)
        
        ts_gw_scaled_train, ts_gw_scaled_val = ts_gw_scaled.split_after(
            pd.to_datetime(datetime.today() - timedelta(days = predict_period)))
        
        model_tft = {}
        if model_save == True:
            model_tft[str(gw_code)] = TFTModel(input_chunk_length = 365*2, 
                                    output_chunk_length = predict_period, 
                                    hidden_size = 64, lstm_layers = 1, 
                                    n_epochs = n_epochs)
            
            model_tft[str(gw_code)].fit(ts_gw_scaled_train[str(gw_code)], 
                                        future_covariates = ts_covariates_scaled[str(rain_code)])
            model_tft[str(gw_code)].save(f'./model/model_tft_gw_{gw_code}.pt')
        else:
            model_tft[str(gw_code)] = TFTModel.load(f'./model/model_tft_gw_{gw_code}.pt',
                                                    map_location=torch.device('cpu'))
        
        # 예측
        ts_pred_scaled = model_tft[str(gw_code)].predict(len(ts_gw_scaled_val[str(gw_code)]) + 3, 
                                                    series = ts_gw_scaled_train[str(gw_code)], 
                                    future_covariates = ts_covariates_scaled[str(rain_code)])
        ts_pred_scaled_sample = model_tft[str(gw_code)].predict(len(ts_gw_scaled_val[str(gw_code)]) + 3, 
                                                    series = ts_gw_scaled_train[str(gw_code)], 
                                    future_covariates = ts_covariates_scaled[str(rain_code)],
                                    num_samples = 500)
        ts_pred = scaler_gw.inverse_transform(ts_pred_scaled)
        ts_pred_sample = scaler_gw.inverse_transform(ts_pred_scaled_sample)
        # Mean Absolute Error
        value_mae = round(mae(ts_gw_scaled_val, ts_pred_scaled), 2)
        
        return ts_gw_raw, ts_pred, ts_pred_sample, value_mae

    # 그래프 함수 생성
    # pio.renderers.default = 'browser'

    def fun_graph(ts_pred, ts_pred_sample, ts_gw_raw, value_mae, gw_korean):
        
        df_val_pred = ts_pred.pd_dataframe()
        
        df_val = df_val_pred[df_val_pred.index <= df_val_pred.index[-3]]
        df_val['YMD'] = df_val.index
        df_val = df_val.reset_index(drop = True)
        
        df_pred = df_val_pred[df_val_pred.index >= df_val_pred.index[-3]]
        df_pred['YMD'] = df_pred.index
        df_pred = df_pred.reset_index(drop = True)
        
        # 몬테카를로 샘플링
        df_pred_sample = ts_pred_sample.pd_dataframe()
        df_pred_sample_conf = pd.DataFrame({
            'upper' : df_pred_sample.apply(lambda x: np.percentile(x, 95), axis = 1),
            'lower' : df_pred_sample.apply(lambda x: np.percentile(x, 5), axis = 1)
            })
        df_pred_sample_conf['YMD'] = df_pred_sample_conf.index
        df_pred_sample_conf = df_pred_sample_conf.reset_index(drop = True)
        
        df_measured = ts_gw_raw.tail(365).pd_dataframe()
        df_measured['YMD'] = df_measured.index
        df_measured = df_measured.reset_index(drop = True)
        
        # 예측일자 추출
        fig_gw = go.Figure()
        fig_gw.add_trace(go.Scatter(
            mode = 'lines', 
            x = df_val['YMD'], y = df_val[df_val.columns[0]], 
            name = 'validation', 
            line = dict(color = 'steelblue')))
        fig_gw.add_trace(go.Scatter(
            mode = 'lines', 
            x = df_pred['YMD'], y = df_pred[df_pred.columns[0]], 
            name = 'predicted', 
            line = dict(color = 'orangered')))
        fig_gw.add_trace(go.Scatter(
            mode = 'lines', 
            x = df_measured['YMD'], y = df_measured[df_pred.columns[0]], 
            name = 'measured',
            line = dict(color = 'black'),
            opacity = 0.4))
        
        # 신뢰구간 추가
        fig_gw.add_trace(go.Scatter(
            x=df_pred_sample_conf['YMD'], 
            y=df_pred_sample_conf['upper'], 
            mode='lines', 
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig_gw.add_trace(go.Scatter(
            x=df_pred_sample_conf['YMD'], 
            y=df_pred_sample_conf['lower'], 
            fill='tonexty', 
            mode='lines', 
            line=dict(width=0),
            fillcolor='rgba(85, 85, 85, 0.2)',
            showlegend=True,
            name='5%~95% percentiles',
            hoverinfo='skip'
        ))
        
        fig_gw.add_annotation(dict(xref = 'paper', yref = 'paper', x = 0.5, y = 1.05,
                                text = f'관측소명:{gw_korean}, MAE: {value_mae}',
                                showarrow = False,
                                font = dict(size = 20)))
        fig_gw.update_layout(xaxis=dict(title='YMD', titlefont = dict(size=15), 
                                        tickfont = dict(size=10), tickformat = '%y-%m-%d'), 
                            yaxis=dict(title='groundwater level', 
                                        titlefont = dict(size=15), 
                                        tickfont = dict(size=10)),
                            template='plotly_white',
                            legend=dict(font = dict(size=10)))
        
        if not os.path.exists('output_streamlit'):
            os.makedirs('output_streamlit')
            
        fig_gw.write_html(f'./output_streamlit/graph_{gw_korean}.html')
        
        # # 그래프 저장
        # plt.rcParams['font.family'] = 'Malgun Gothic'
        # plt.figure(figsize=(15, 10))
        # plt.plot(df_val['YMD'], df_val[df_val.columns[0]], label='Validation', color='steelblue')
        # plt.plot(df_pred['YMD'], df_pred[df_pred.columns[0]], label='Predicted', color='orangered')
        # plt.plot(df_measured['YMD'], df_measured[df_pred.columns[0]], label='Measured', color='black', alpha=0.4)
        
        # plt.fill_between(df_pred_sample_conf['YMD'], df_pred_sample_conf['lower'], df_pred_sample_conf['upper'], 
        #                 color='gray', alpha=0.2, label='5%~95% percentiles')
        
        # plt.title(f'관측소명: {gw_korean}, MAE: {value_mae}', fontsize=20)
        # plt.xlabel('YMD', fontsize=15)
        # plt.ylabel('Groundwater Level', fontsize=15)
        # plt.legend(fontsize=12)
        # plt.xticks(rotation=45)
            
        # plt.savefig(f'./output/graph_{gw_korean}.jpg')
        # plt.close()
    
    if selected_gennum == '제주노형':
        # 모형훈련, 예측 및 그래프 확인(지하수(제주노형 95534), 강우(제주 184))
            
        # 모형훈련 및 예측
        ts_gw_raw_제주노형, ts_pred_제주노형, ts_pred_sample_제주노형, value_mae_제주노형 = fun_model(
            gw_code = 95534, rain_code = 184, learn_date = '2021-01-01', 
            n_epochs = 30, predict_period = 30, model_save = False)

        # 그래프 확인
        fun_graph(ts_pred_제주노형, ts_pred_sample_제주노형, ts_gw_raw_제주노형, 
                value_mae_제주노형, gw_korean = '제주노형')
        try:
            with open(f'./output_streamlit/graph_{selected_gennum}.html', 
                      'r', encoding='utf-8') as html_file:
                source_code = html_file.read()
                components.html(source_code, height=400)
        except FileNotFoundError:
                st.error(f"{selected_gennum}의 그래프 파일을 찾을 수 없습니다.")  
    
    if selected_gennum == '제주동홍':
        # 모형훈련(지하수(제주동홍 95535), 강우(서귀포 189))

        # 모형훈련 및 예측
        ts_gw_raw_제주동홍, ts_pred_제주동홍, ts_pred_sample_제주동홍, value_mae_제주동홍 = fun_model(
            gw_code = 95535, rain_code = 189, learn_date = '2020-01-01', 
            n_epochs = 30,  predict_period = 30, model_save = False)

        # 그래프 확인
        fun_graph(ts_pred_제주동홍, ts_pred_sample_제주동홍, ts_gw_raw_제주동홍, 
                value_mae_제주동홍, gw_korean = '제주동홍')
        
        html_path = gennum_to_file[selected_gennum]
        try:
            with open(f'./output_streamlit/graph_{selected_gennum}.html', 
                      'r', encoding='utf-8') as html_file:
                source_code = html_file.read()
                components.html(source_code, height=400)
        except FileNotFoundError:
                st.error(f"{selected_gennum}의 그래프 파일을 찾을 수 없습니다.")  
                
    if selected_gennum == '제주조천':
        # 모형훈련(지하수(제주조천 95537), 강우(성산 188))      

        ts_gw_raw_제주조천, ts_pred_제주조천, ts_pred_sample_제주조천, value_mae_제주조천 = fun_model(
            gw_code = 95537, rain_code = 188, learn_date = '2010-01-01', 
            n_epochs = 20,  predict_period = 90, model_save = False)    

        # 그래프 확인
        fun_graph(ts_pred_제주조천, ts_pred_sample_제주조천, ts_gw_raw_제주조천, 
                value_mae_제주조천, gw_korean = '제주조천')
        html_path = gennum_to_file[selected_gennum]
        try:
            with open(f'./output_streamlit/graph_{selected_gennum}.html', 
                      'r', encoding='utf-8') as html_file:
                source_code = html_file.read()
                components.html(source_code, height=400)
        except FileNotFoundError:
                st.error(f"{selected_gennum}의 그래프 파일을 찾을 수 없습니다.")   

    if selected_gennum == '제주한경':
        # 모형훈련(지하수(제주한경 95536), 강우(고산 185))

        ts_gw_raw_제주한경, ts_pred_제주한경, ts_pred_sample_제주한경, value_mae_제주한경 = fun_model(
            gw_code = 95536, rain_code = 185, learn_date = '2010-01-01', 
            n_epochs = 20,predict_period = 90, model_save = False)

        # 그래프 확인
        fun_graph(ts_pred_제주한경, ts_pred_sample_제주한경, ts_gw_raw_제주한경, 
                value_mae_제주한경, gw_korean = '제주한경')
        html_path = gennum_to_file[selected_gennum]
        try:
            with open(f'./output_streamlit/graph_{selected_gennum}.html', 
                      'r', encoding='utf-8') as html_file:
                source_code = html_file.read()
                components.html(source_code, height=400)
        except FileNotFoundError:
                st.error(f"{selected_gennum}의 그래프 파일을 찾을 수 없습니다.")  

    progress_bar.progress(100, text="분석완료! 아래의 그래프를 확인하세요.")