import random

import altair as alt
import streamlit as st
import pandas as pd
import numpy as np


def main():
    forecast_details = fetch_forecasts()
    forecasts = forecast_details['data']
    reporting_date = forecast_details['reporting_date']

    st.title('Forecasts')
    st.altair_chart(
        make_forecast_line_chart(forecasts, reporting_date), 
        use_container_width=True
    )

    st.subheader('Mean Absolute Percentage Error')
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric(label='Last 7 days', value=f'{get_forecast_mape_last_n_days(7):.2f}%')
    with metric_cols[1]:
        st.metric(label='Last 30 days', value=f'{get_forecast_mape_last_n_days(30):.2f}%')
    with metric_cols[2]:
        st.metric(label='Last 90 days', value=f'{get_forecast_mape_last_n_days(90):.2f}%')


def make_forecast_line_chart(forecasts, reporting_date):
    plot_data = forecasts.melt(
        id_vars=['date'], 
        value_vars=['y', 'y_pred'], 
        var_name='series', 
        value_name='value',
    )

    hover = alt.selection_single(
        fields=['date'],
        nearest=True,
        on='mouseover',
        empty='none',
    )

    lines = alt.Chart(plot_data) \
        .mark_line() \
        .encode(
            x='date',
            y='value',
            color='series',
            strokeDash=alt.condition(
                (alt.datum.series == 'y_pred'),
                alt.value([5,5]),
                alt.value([0]),
            ),
        )
    
    points = lines.transform_filter(hover).mark_circle(size=65)

    tooltips = alt.Chart(plot_data) \
        .mark_rule() \
        .encode(
            x='date',
            y='value',
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip('date', title='date'),
                alt.Tooltip('value', title='value'),
            ],
        ) \
        .add_selection(hover)

    chart = (lines + points + tooltips).interactive()
    return chart


@st.cache
def fetch_forecasts():
    forecasts = pd.read_csv('data/forecasts.csv', dtype='str')
    forecasts['date'] = pd.to_datetime(forecasts['date'])
    forecasts['y'] = pd.to_numeric(forecasts['y'])
    forecasts['y_pred'] = pd.to_numeric(forecasts['y_pred'])

    # pddt = pandas datetime
    reporting_date_pddt = forecasts[forecasts['y'].notnull()]['date'].max()

    reporting_date = reporting_date_pddt.to_pydatetime().strftime('%Y-%m-%d')

    start_date = max(forecasts['date'].min(),
                     reporting_date_pddt - pd.Timedelta(365 *1, unit='d'))

    forecasts = forecasts[forecasts['date'] >= start_date].copy()

    # random.seed(42)
    # np.random.seed(42)
    # dates = pd.date_range('2022-01-01', '2023-01-01', freq='D').values
    # num_timesteps = len(dates)
    # y = np.zeros(shape=(num_timesteps,))
    # for i in range(1, num_timesteps):
    #     y[i] = y[i-1] + np.random.normal()
    
    # y_pred = y + np.random.normal()

    # data = pd.DataFrame({
    #     'date': dates,
    #     'y': y,
    #     'y_pred': y_pred
    # })

    # data.loc[data['date'] >= '2022-12-01', 'y'] = np.nan

    return {'data': forecasts, 'reporting_date': reporting_date}


@st.cache
def get_forecast_mape_last_n_days(n):
    forecast_details = fetch_forecasts()
    forecasts = forecast_details['data']
    reporting_date = pd.to_datetime(forecast_details['reporting_date'])
    n_days_ago = reporting_date - pd.Timedelta(n, unit='d')
    forecasts_last_n_days = forecasts[forecasts['date'] > n_days_ago]
    return mape(forecasts_last_n_days['y'], forecasts_last_n_days['y_pred'])


def mape(y, y_pred):
    eps = np.finfo(np.float64).eps
    return np.mean((y - y_pred) / np.maximum(np.abs(y), eps)) * 100.0


main()

