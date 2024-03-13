import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from visualizations import *
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

df = pd.read_csv('df_final.csv')
df['time'] = pd.to_datetime(df['time'])
# df = df.set_index('time')

# Sidebar for navigation
page = st.sidebar.selectbox("Select a Page", ["Dashboard Overview", "Electricity Pricing Trends", "Data Insights and Exploration","Renewable v/s Non-Renewable Energy", "Interactive Chart"])

# Page 1: Dashboard Overview
if page == "Dashboard Overview":
    st.title("IE Dataton: Bluetab")
    st.markdown('''# The Challenge: Predicting Electricity Prices
        ''')
    st.markdown('''
                In the dynamic world of energy, the ability to forecast, demand and understand the factors affecting prices is essential for efficient resource management and informed decision-making. In this challenge, we will specifically focus on the Spanish market, using real data on energy demand and prices from 2015 to 2018, as well as climatological records in the main Spanish cities during those years.
                ''')
    st.markdown('''
                The goal of this challenge is to:
                - Explore and analyze the data.
                - Provide energy price forecasting.
                - Proposal of an analytical and predictive model explaining what advantages it would provide in a real world.
                ''')
    st.markdown(''' ## Initial Models: ''')
    fig, metrics = plot_and_evaluate_predictions(df)
    st.markdown(metrics)
    st.plotly_chart(fig)
    st.markdown('''
                ## Metric Analysis:
                - Root Mean Square Error (RMSE): The RMSE is 13.25, which means that the standard deviation of the prediction errors is around 13.25 price units. The RMSE gives a sense of how much error the system typically makes in its predictions. For electricity prices, depending on the price scale, an RMSE of over 13 might be considered high.
                - R-squared (R²): The R² value is approximately 0.13, which is quite low. This suggests that only around 13% of the variance in the actual price can be explained by the model's predictions. An R² value closer to 1 indicates a model that can explain more of the variance. In this context, it suggests the model is not explaining much of the variability in the electricity prices.
                - Mean Absolute Error (MAE): The MAE is around 10.49, which indicates that the average absolute error in the predictions is about 10.49 price units. This is another way to measure error, and it shows that the predictions are, on average, about 10.49 units away from the actual price values.
                ## Our Impressions:
                - The model appears to have a general sense of the trend but is not very accurate in predicting the actual prices, especially considering the volatility of electricity prices.
                - Given the low R² value, the model may not be very useful in its current form for reliable price forecasting.
                - The errors indicated by RMSE and MAE are relatively large, which can be problematic for budgeting, planning, or trading based on these predictions.
                - It would be beneficial to consider model improvements or additional data features that could better capture the peaks and volatility in the price.
                ''')
    st.markdown(''' 
                ## Approaching the Data:
                - We begin with two .csv files, one containing electricity prices and the other containing weather data for the following cities:
                    - Barcelona
                    - Bilbao
                    - Madrid
                    - Sevilla
                    - Valencia
                
                    The data was hourly and spanned from 2014-12-31 to 2018-12-31.
                ''')
    st.markdown('''
                The data was preprocessed and merged into a single dataframe. The final dataframe contains the following columns:
                - time
                - generation biomass
                - generation fossil brown coal/lignite
                - generation fossil gas
                - generation fossil hard coal
                - generation fossil oil
                - generation hydro pumped storage consumption
                - generation hydro run-of-river and poundage
                - generation hydro water reservoir
                - generation nuclear
                - generation other
                - generation other renewable
                - generation solar
                - generation waste
                - generation wind onshore
                - total load actual
                - price day ahead
                - price actual
                - temp_Barcelona
                - temp_min_Barcelona
                - temp_max_Barcelona
                - pressure_Barcelona
                - humidity_Barcelona
                - wind_speed_Barcelona
                - wind_deg_Barcelona
                - rain_1h_Barcelona
                - clouds_all_Barcelona
                - temp_Bilbao
                - temp_min_Bilbao
                - temp_max_Bilbao
                - pressure_Bilbao
                - humidity_Bilbao
                - wind_speed_Bilbao
                - wind_deg_Bilbao
                - rain_1h_Bilbao
                - snow_3h_Bilbao
                - clouds_all_Bilbao
                - temp_Madrid
                - temp_min_Madrid
                - temp_max_Madrid
                - pressure_Madrid
                - humidity_Madrid
                - wind_speed_Madrid
                - wind_deg_Madrid
                - rain_1h_Madrid
                - snow_3h_Madrid
                - clouds_all_Madrid
                - temp_Seville
                - temp_min_Seville
                - temp_max_Seville
                - pressure_Seville
                - humidity_Seville
                - wind_speed_Seville
                - wind_deg_Seville
                - rain_1h_Seville
                - clouds_all_Seville
                - temp_Valencia
                - temp_min_Valencia
                - temp_max_Valencia
                - pressure_Valencia
                - humidity_Valencia
                - wind_speed_Valencia
                - wind_deg_Valencia
                - rain_1h_Valencia
                - snow_3h_Valencia
                - clouds_all_Valencia
                - hour
                - weekday
                - month
                - business hour
                - temp_range_Barcelona
                - temp_range_Bilbao
                - temp_range_Madrid
                - temp_range_Seville
                - temp_range_Valencia
                - temp_weighted
                - generation coal all
                ''')
    st.markdown('''
                ## Time Series Analysis:
                ''')
    st.image('images/acf.png', caption='Autocorrelation Function (ACF) Plot')
    st.image('images/pacf.png', caption='Partial Autocorrelation Function (PACF) Plot')
    st.markdown('''
                 After the data was preprocessed and merged, we performed a time series analysis to understand the trends and patterns in the electricity prices. The analysis included:
                - We performed Augmented Dickey-Fuller (ADF) test to check for stationarity of the time series using  the adfuller function from the statsmodels package. Stationarity is a crucial assumption in time series analysis, indicating that the statistical properties of the time series do not change over time.
                - The output of the ADF test includes the test statistic and the p-value. The ADF test statistic is a number that can be negative or positive; a more negative value suggests that the null hypothesis of a unit root can be rejected, indicating stationarity. The p-value helps to decide whether to reject the null hypothesis; a low p-value (typically < 0.05) indicates that the null hypothesis can be rejected.
                - The ADF Statistic is -9.147016232851161, which is quite negative.
                - The p-value is 2.7504934849347068e-15, which is very close to zero.
                ## Interpretation:
                - The negative ADF statistic indicates that the time series is likely stationary. The p-value being significantly below 0.05 allows us to reject the null hypothesis of a unit root, reinforcing the conclusion that the time series is stationary.
                - The ACF plot that you've shown indicates a periodic correlation structure, as the autocorrelation coefficient is high at regular intervals. This could suggest seasonality in the data.
                - The data appears ready for machine learning methods that are suited for stationary time series. However, choosing the right model and parameters would require a deeper analysis of these plots, considering possible seasonality, and performing model selection and validation processes.
                ''')
    st.markdown('''
                ## Models Used:
                - ARIMA
                - SARIMA
                - Random Forest
                - LightGBM
                - XGBoost
                ''')
    st.markdown('''
                ## Feature Importance:
                - The Random Forest model has the highest feature importance for the following features:
                ''')
    st.image('images/rffeature_importance.png', caption='Random Forest Feature Importance Plot')
    st.markdown('''
                - The LightGBM model has the highest feature importance for the following features:
                ''')
    st.image('images/lgbmfeature_importance.png', caption='LightGBM Feature Importance Plot')
    st.markdown('''
                - The XGBoost model has the highest feature importance for the following features:
                ''')
    st.image('images/xgboostfeature_importance.png', caption='XGBoost Feature Importance Plot')
    st.markdown('''
                ## Results:
                ---
              
                | Model              | Metric   | Value     |
                |--------------------|----------|-----------|
                | **Default Predictions** | RMSE | 13.2498 |
                |                    | MAE     | 10.4853   |
                |                    | R² Score| 0.1298    |
                | **ARIMA**          | MAE     | 8.8412    |
                |                    | RMSE    | 9.6174    |
                |                    | MAPE    | 12.8147%  |
                |                    | R² Score| -1.1222   |
                | **SARIMA**         | MAE     | 1.9211    |
                |                    | RMSE    | 2.2754    |
                |                    | MAPE    | 2.7590%   |
                |                    | R² Score| 0.8812    |
                | **Random Forest**  | MSE     | 4.6915    |
                |                    | RMSE    | 2.1660    |
                |                    | R² Score| 0.9647    |
                |                    | MAE     | 1.5770    |
                | **LightGBM (Train)** | MAE   | 1.4205    |
                |                    | RMSE    | 1.9686    |
                |                    | R² Score| 0.9809    |
                | **LightGBM (Test)** | MAE    | 1.5004    |
                |                    | RMSE    | 2.0556    |
                |                    | R² Score| 0.9682    |
                | **XGBoost**        | MAE     | 1.6696    |
                |                    | RMSE    | 2.2526    |
                |                    | MSE     | 5.0743    |
                |                    | R² Score| 0.9618    |
                ---
                ''')
 


# Page 2: Electricity Pricing Trends
elif page == "Electricity Pricing Trends":
    st.title("Electricity Pricing Trends")
    st.markdown('''
                ## Electricity Pricing Trends Over Time
                ''')
    fig = plot_interactive_electricity_prices(df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
                Looking at the graph of electricity prices over time, here's an analysis of the observable trends:
                - Seasonality: There appears to be a seasonal pattern in electricity prices, with peaks and troughs occurring at regular intervals. This could suggest higher prices during certain times of the year, which might correlate with increased demand due to heating or cooling needs depending on the region.
                - Volatility: The electricity prices show quite a bit of volatility, with some very sharp spikes and drops. The presence of these extreme values could indicate sudden changes in demand, issues with supply, or other market-affecting events.
                - Trend: There seems to be no clear long-term upward or downward trend over the years; the prices fluctuate around a certain level. This indicates that the average price of electricity has been relatively stable over the period depicted in the graph.
                - Anomalies: There are a few notable spikes that stand out above the general 'noise' level of the rest of the data points. Identifying the exact timing and potential causes for these anomalies would require additional context or data, such as news events, changes in production costs, or policy changes affecting the electricity market.
                ''')
    st.markdown('''
                ## Average Electricity Price by Hour of the Day
                ''')
    fig = interactive_line_graph_from_df(df, 'hour', 'price actual', title='Average Actual Price by Hour')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
                The plot shows the relationship between the hour of the day (ranging from 0 to 23 on the x-axis) and the average electricity price (on the y-axis, in €/MWh). The plot reveals that the lowest prices occur in the early morning hours, around 4-5 AM, after which the prices start to rise, peaking at around 9 AM. There's a slight dip following this peak, then prices rise again, reaching the highest average price between 7 PM and 8 PM. After this second peak, prices decline sharply towards midnight. We can infer that the time of day has a significant impact on electricity prices, with the highest prices occurring during the late afternoon and early evening hours due to office hours starting and ending respectively. This information can help us better produce electricity to deal with dynamic usage patterns.
                ''')
    fig = interactive_boxplot_by_hour(df, 'hour', 'price actual', title='Distribution of Actual Price by Hour')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
                 The x-axis represents different hours of the day from 0 to 23, and the y-axis represents the range of electricity prices in €/MWh. Each box represents the interquartile range (IQR) of prices for each hour, with the median marked by a line within the box. The whiskers extend to the rest of the distribution, excluding outliers, which are plotted as individual points. The prices are generally higher and more variable during the day, especially from late morning to evening (around 9 AM to 7 PM), and they tend to be lower and less variable late at night and early morning. The highest median prices and the widest range of prices, indicating greater volatility, occur during the evening hours. There are also numerous outliers indicating sporadic spikes in prices throughout the day.
                ''')
    st.markdown(''' 
                ## Load and Price:
                ''')
    fig = plot_comparison_over_time(df)
    st.plotly_chart(fig, use_container_width=True)




# Page 3: Data Insights and Exploration
elif page == "Data Insights and Exploration":
    st.title("Data Insights and Exploration")
    st.markdown('''
                ## Energy Distribution:
                ''')
    st.markdown('''
                ## Nuclear Energy:
                ''')
    fig = interactive_nuclear_generation_plot(df, 'time', 'generation nuclear')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
                The plot displays the nuclear energy generation in megawatts (MW) on the y-axis across a timeline on the x-axis, which appears to range from January 2015 to sometime in 2019. The plot features sharp vertical drops to the bottom of the y-axis at various points, suggesting periods of reduced output or shutdowns in nuclear generation. These could be indicative of maintenance periods, refueling, or unexpected outages. The energy generation seems quite consistent when not in these lowered states, implying a stable output during normal operational periods.
                ''')
    fig = boxplot_nuclear_generation_by_hour(df, 'hour', 'generation nuclear')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
                The box itself shows the interquartile range (IQR), which contains the middle 50% of the data for that hour. The horizontal line within each box indicates the median price for that hour. The whiskers extending from the boxes represent the range of the data excluding outliers, which are shown as individual points outside of the whiskers.

                Inferences from the graph could include:
                - Price Consistency: The median prices do not fluctuate dramatically from hour to hour, suggesting relatively stable prices throughout the day.
                - Variability: The interquartile ranges are similar across most hours, which means the variability in price is consistent, except for a few hours where the boxes appear slightly taller, indicating more variability in prices during those hours.
                - Outliers: There are several outliers, particularly during certain hours, indicating occasional spikes or drops in the price that deviate significantly from the typical price range.
                - Peak Prices: Without the actual data, it's hard to determine peak price hours definitively, but it seems there are no significant spikes in median prices at any specific hour, as indicated by the relatively even height of the median lines across all hours.
                ''')
    st.markdown('''
                ## Solar Energy:
                ''')
    fig = plot_solar_generation_by_hour_interactive(df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
                The plot shows the solar energy generation in megawatts (MW) on the y-axis across a timeline on the x-axis, with a range from January 2015 to sometime in 2019. The plot features sharp vertical drops to the bottom of the y-axis at various points, suggesting periods of reduced output or shutdowns in solar generation. These could be indicative of nighttime, cloudy weather, or maintenance periods. The energy generation seems quite consistent when not in these lowered states, implying a stable output during normal operational periods.
                
                Here's the interpretation of the graph:
                - Low Nighttime Generation: There's minimal to no solar generation at night (from around 0 to 5 hours), as indicated by the boxes at the bottom axis, which is expected because solar panels don't generate electricity without sunlight.
                - Rise and Fall Pattern: Starting from the early hours of the morning (around 6), the solar generation begins to increase, peaking around midday (which seems to be around hours 10 to 15), and then decreases as the day progresses towards evening (hours 16 to 20).
                - Variability: The length of the boxes indicates the variability of generation within each hour. Morning and evening hours show more variability in solar generation, likely due to the changes in sunlight intensity during sunrise and sunset.
                - Outliers: There are many outliers, particularly during daylight hours, which may suggest days with exceptionally high or low generation, potentially due to weather conditions like overcast or exceptionally clear days.
                - Consistent Peak Generation: The middle of the day shows consistently higher generation, which is characteristic of solar power corresponding to when the sun is highest.
                ''')
    st.markdown('''
                ## Hydro Energy:
                ''')
    fig = plot_hydro_generation_by_hour(df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
                The plot shows the hydro energy generation in megawatts (MW) on the y-axis across a timeline on the x-axis, with a range from January 2015 to in 2019.
                
                Obervations:
                - The central tendency of hydro generation varies throughout the day. It appears to be lower during the early hours (midnight to early morning), increases and remains fairly consistent during the middle of the day, and then reduces slightly in the late hours.
                - The median hydro generation (indicated by the line inside each box) seems relatively stable from around 6 AM to 6 PM, suggesting a consistent output of hydro generation during these hours.
                - There is significant variability in hydro generation, as indicated by the length of the boxes and the whiskers. The longer boxes and whiskers during certain hours (like around noon and late evening) suggest more variability in generation during these times.
                - There are a number of outliers at various hours (shown as individual points above and below the boxes), indicating occasional spikes or drops in hydro generation beyond the typical range.
                ''')
    st.markdown('''
                Inferences:
                - The consistent median during daylight hours might suggest a correlation with higher demand or operational efficiencies that occur during these times.
                - The variability and outliers could be influenced by a number of factors, such as changes in water supply, demand fluctuations, or operational issues.
                - The reduced generation in the early hours could be due to lower electricity demand, leading to a decreased need for generation, or it could be a strategic decision to conserve water resources for peak hours.
                - The evening hours show a decrease in generation, which might be due to a combination of reduced demand and a strategy to store water for the next day.
                ''')
    st.markdown('''
                ## Wind Energy:
                ''')
    fig = plot_wind_generation_by_hour(df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
                Observations:
                - The median wind generation (the line in the middle of each box) varies throughout the day, suggesting that there are patterns to the wind generation over the 24-hour cycle.
                - There appears to be a significant spread in the interquartile range (the height of each box), indicating variability in wind generation during each hour. This variability might suggest fluctuating wind conditions or changes in wind generation capacity.
                - Numerous outliers (points beyond the whiskers of the boxes) are visible throughout the day, indicating instances of wind generation that are much higher or lower than the typical range for that hour.
                - The wind generation does not show a clear peak hour; rather, it has several hours with comparably high medians and a wide spread, which may suggest that the wind generation is not solely dependent on the time of day.
                ''')
    st.markdown('''
                Inferences:
                - The presence of outliers could be attributed to environmental factors that cause wind speed to increase or decrease significantly, affecting the power generation.
                - The wide range of generation across most hours suggests that wind generation is less predictable and more susceptible to rapid changes in wind conditions compared to other forms of generation, like hydro or solar, which tend to follow more predictable daily patterns.
                - There doesn't appear to be a strong diurnal pattern (day vs. night difference) in wind generation, which differs from solar generation that's dependent on sunlight hours.
                - Wind generation facilities might be operating near their capacity at several points during the day, as indicated by the higher outlier values.
                ''')
    st.markdown('''
                ## Fossil Fuel Generation:
                ''')
    fig = plot_fossil_generation_by_hour(df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
                Observations:
                - The median generation of fossil fuels, indicated by the line in the middle of each box, shows some variation throughout the day. However, there is no clear trend of increase or decrease observable in the median values across different hours.
                - The spread of the interquartile range (IQR) varies, which reflects the variability in fossil fuel generation at different hours. The IQR seems narrower during early morning hours and wider in the afternoon and evening hours.
                - There are numerous outliers throughout the day, more prominently during the afternoon and evening hours, which may indicate sporadic spikes in generation(Peak usage).
                - The range of the whiskers, which indicates variability outside the upper and lower quartiles, also varies by hour. This could suggest fluctuations in the operational capacity or demand for fossil fuel generation throughout the day.
                ''')
    st.markdown('''
                Inferences:
                - The presence of outliers and a wide range during certain hours might suggest that the demand for fossil fuel generation can be unpredictable and may respond to factors like peak consumption periods, availability of other generation sources, or operational dynamics of the power plants.
                - The relatively consistent median values across hours suggest that fossil fuel plants provide a steady baseline of electricity generation, potentially due to their role in maintaining grid stability in the face of fluctuating generation from renewable sources.
                - The variability in generation could also be influenced by market factors such as changes in fuel prices or regulatory policies impacting the operation of fossil fuel plants.
                ''')

    

# Page 4: Renewable v/s Non-Renewable Energy
elif page == "Renewable v/s Non-Renewable Energy":
    st.title("Renewable v/s Non-Renewable Energy")
    
    fig_daily = plot_average_daily_generation(df)
    st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown('''
                Observations:
                - Non-renewable energy generation is consistently higher than renewable generation throughout the day. This suggests that the grid relies more heavily on non-renewable sources.
                - Renewable energy generation peaks during the middle of the day, which could be due to higher sunlight availability for solar power. This peak is typically around noon to early afternoon, which aligns with the sun’s highest position.
                - Non-renewable generation remains relatively flat with a slight increase during daytime hours and a decrease after 21:00. This could indicate base load generation from non-renewable sources with slight adjustments to meet peak demands.
                - Both renewable and non-renewable generation show a decline after 21:00, which likely corresponds to a decrease in overall electricity demand as businesses close and residential energy use drops.
                ''')
    st.markdown('''
                Inferences:
                - The higher baseline of non-renewable generation suggests that the energy grid may be designed to depend on non-renewables for a constant power supply, possibly due to the intermittent nature of some renewable sources.
                - The significant difference between renewable and non-renewable generation could indicate that there’s either limited renewable energy capacity or that renewables are supplementary to a non-renewable base load.
                - The peak in renewable generation is most likely influenced by solar generation, given that wind energy would not have such a pronounced midday peak. This is also supported by the rapid decline in renewable generation as the sun sets.
                - Policy or economic factors could be in play; for example, non-renewables may still be more economically viable or there may be insufficient infrastructure to capture and store renewable energy effectively throughout the day.
                ''')
    fig_hourly = plot_average_hourly_generation(df)
    st.plotly_chart(fig_hourly, use_container_width=True)
    st.markdown('''
                Observations:
                - Non-renewable energy generation shows significant day-to-day variability but maintains higher values than renewable energy generation most of the time.
                - Renewable energy generation also exhibits daily fluctuations and seems to follow a seasonal pattern, with certain periods showing increased generation, which may correlate with seasons of higher renewable energy potential (like sunny or windy seasons).
                - There are noticeable spikes in non-renewable energy generation that could correspond to periods of increased demand or reduced renewable generation capacity.
                - The non-renewable generation seems to have a downward trend over time, particularly noticeable in the reduction of peak values, while renewable generation appears to have a slightly increasing trend.
                ''')
    st.markdown('''
                Inferences:
                - The higher levels of non-renewable generation suggest that the energy system relies heavily on these sources to meet its daily energy demands, possibly due to their reliability and controllable output compared to the variable nature of renewable sources.    
                - The seasonal patterns in renewable generation might be influenced by weather-related factors, such as increased solar generation during summer months and possibly more wind generation during certain seasons.
                - The decline in non-renewable generation peaks and the modest increase in renewable generation over time could be indicative of a gradual transition to renewable sources, reflecting investments in renewable energy infrastructure and a move towards sustainability.
                - Spikes in non-renewable generation could be attributed to increased energy demand during extreme weather events, which often require the use of fast-responding non-renewable energy plants.
                ''')
    st.markdown('''
                ## Energy Generation by Source:
                ''')
    fig = plot_avg_generation_per_day(df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
                Observations:
                - Both renewable and non-renewable generation show fluctuations throughout the week with non-renewable generation consistently outpacing renewable generation.
                - The average generation for both renewable and non-renewable energy seems to dip mid-week around Wednesday.
                - Weekend generation for both types does not fall significantly compared to weekdays, suggesting a steady demand for energy even on weekends.
                ''')
    st.markdown('''
                Inferences:
                - The consistent outperformance of non-renewable generation suggests a heavy reliance on these sources across all days, likely due to their ability to provide stable, controllable output.    
                - The mid-week dip could be related to industrial or commercial operations that are reduced mid-week, leading to lower energy demand. Alternatively, it could reflect weekly operational or maintenance practices within the energy generation facilities.
                - The relatively stable generation on weekends indicates that the reduction in commercial and industrial activity is potentially offset by increased residential use, or that baseline power generation levels are maintained regardless of day-to-day fluctuations in demand.
                ''')
    st.markdown('''
                ## Generation by Source:
                ''')
    fig = generate_energy_generation_graph(df)
    st.plotly_chart(fig, use_container_width=True)



    

