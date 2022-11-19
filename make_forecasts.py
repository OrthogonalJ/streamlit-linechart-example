import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric, add_changepoints_to_plot


page_views = pd.read_csv('data/wikipedia_page_views/pageviews-20150701-20210906.csv', dtype='str')
page_views.Date = pd.to_datetime(page_views.Date)
page_views.loc[:, 'Main Page'] = pd.to_numeric(page_views.loc[:, 'Main Page'])

model_data = page_views.rename(columns={'Date': 'ds', 'Main Page': 'y'})
model = Prophet(changepoint_prior_scale=1.0, seasonality_mode='multiplicative')
model.fit(model_data)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
forecast = forecast.merge(model_data, on='ds', how='left')
forecast = forecast.loc[:, ['ds', 'y', 'yhat']]
forecast.rename(columns={'ds': 'date', 'yhat': 'y_pred'}, inplace=True)
forecast.to_csv('data/forecasts.csv', index=False)
