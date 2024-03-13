# visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objs as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import mplcursors
import streamlit as st


sns.set_theme(style="whitegrid")

def plot_electricity_prices(df):
    """Plot electricity prices over time."""
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['price actual'], label='Electricity Price')
    ax.set_title('Electricity Price Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    plt.legend()
    return fig

def plot_generation_data(df):
    """Plot electricity generation data over time."""
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['generation biomass'], label='Generation Data')
    ax.set_title('Electricity Generation Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Generation')
    plt.legend()
    return fig


# generation_visualization.py
import plotly.graph_objects as go

# generation_visualization.py
import plotly.graph_objects as go

def plot_generation_stack(df):
    """
    Plots an interactive stacked area chart of electricity generation by source with smooth lines.

    Parameters:
    - df: pandas.DataFrame containing the dataset with time and generation columns.

    Returns:
    - fig: Plotly Figure object with smoothed lines.
    """
    # Filter to include only generation columns
    generation_columns = [col for col in df.columns if 'generation' in col and 'total' not in col]

    # Creating the figure
    fig = go.Figure()

    # Adding each generation source as a trace with smoothed lines
    for column in generation_columns:
        fig.add_trace(go.Scatter(x=df['time'], y=df[column], mode='lines', name=column, stackgroup='one',
                                 line_shape='spline',  # Use spline for smooth lines
                                 hoverinfo='x+y+name'))

    # Updating the layout for a better look and specifying hover label properties
    fig.update_layout(
        title='Electricity Generation Mix Over Time',
        xaxis_title='Time',
        yaxis_title='Generation (MWh)',
        hovermode='x unified',
        autosize=False,
        width=1400,  # Set figure width
        height=600,  # Set figure height
        hoverlabel=dict(
            namelength=-1  # Display full text in hover labels; no truncation
        )
    )

    return fig

def plot_generation_mix(df):
    """Plot the mix of electricity generation over time."""
    generation_columns = [col for col in df.columns if 'generation' in col and 'total' not in col and 'all' not in col]
    fig = go.Figure()
    for gen_type in generation_columns:
        fig.add_trace(go.Scatter(x=df['time'], y=df[gen_type], mode='lines', stackgroup='one', name=gen_type))
    fig.update_layout(title='Electricity Generation Mix Over Time', xaxis_title='Time', yaxis_title='Generation (MWh)')
    return fig

def plot_electricity_prices(df):
    """Plot comparison of day ahead and actual electricity prices over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['price day ahead'], mode='lines', name='Price Day Ahead'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['price actual'], mode='lines', name='Price Actual'))
    fig.update_layout(title='Electricity Prices Over Time', xaxis_title='Time', yaxis_title='Price (EUR/MWh)')
    return fig

def plot_load_vs_generation(df):
    """Compare total load actual with the sum of all generation types."""
    # Assuming 'generation coal all' represents the total generation here for simplicity
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['total load actual'], mode='lines', name='Total Load Actual'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['generation coal all'], mode='lines', name='Total Generation'))
    fig.update_layout(title='Total Load vs Total Generation Over Time', xaxis_title='Time', yaxis_title='MWh')
    return fig

def plot_price_vs_temperature(df):
    """Plot actual electricity price and temperatures in Barcelona, Bilbao, Madrid, and Seville over time."""
    fig = go.Figure()
    # Add Price Actual Trace
    fig.add_trace(go.Scatter(x=df['time'], y=df['price actual'], mode='lines', name='Price Actual', yaxis='y1'))
    # Add Temperature Traces for each city
    cities = ['Barcelona', 'Bilbao', 'Madrid', 'Seville']
    for city in cities:
        temp_avg = df[f'temp_{city}']
        fig.add_trace(go.Scatter(x=df['time'], y=temp_avg, mode='lines', name=f'Temp {city}', yaxis='y2'))
    # Update Layout for Dual Y-Axis
    fig.update_layout(
        title='Actual Electricity Price vs Temperature in Cities Over Time',
        xaxis_title='Time',
        yaxis=dict(
            title='Price Actual (EUR/MWh)',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
        ),
        yaxis2=dict(
            title='Temperature (°C)',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        )
    )
    return fig

# plot_functions.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_price_vs_temperature_avg(df):
    """
    Plot actual electricity price and average temperatures of Barcelona, Bilbao, Madrid, and Seville over time.
    """
    # Calculate average temperature from selected cities
    temp_columns = ['temp_Barcelona', 'temp_Bilbao', 'temp_Madrid', 'temp_Seville']
    df['temp_avg'] = df[temp_columns].mean(axis=1)
    
    fig = go.Figure()

    # Add Price Actual Trace
    fig.add_trace(go.Scatter(x=df['time'], y=df['price actual'], mode='lines', name='Price Actual', yaxis='y1'))

    # Add Average Temperature Trace
    fig.add_trace(go.Scatter(x=df['time'], y=df['temp_avg'], mode='lines', name='Avg Temp (4 cities)', yaxis='y2', line=dict(color='red')))

    # Update Layout for Dual Y-Axis
    fig.update_layout(
        title='Actual Electricity Price vs Average Temperature Over Time',
        xaxis_title='Time',
        yaxis=dict(
            title='Price Actual (EUR/MWh)',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
        ),
        yaxis2=dict(
            title='Average Temperature (°C)',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right',
        ),
        autosize=False,
        width=800,  # Adjust width as needed
        height=600,  # Adjust height as needed
    )

    return fig


def plot_and_evaluate_predictions(df_energy):
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(df_energy['price actual'], df_energy['price day ahead']))
    # Calculate R-squared
    r_squared = r2_score(df_energy['price actual'], df_energy['price day ahead'])
    # Calculate MAE
    mae = mean_absolute_error(df_energy['price actual'], df_energy['price day ahead'])

    # Create traces with a hovertemplate
    trace0 = go.Scatter(
        x = df_energy.index,
        y = df_energy['price day ahead'],
        mode = 'lines',
        name = 'Predicted Price',
        line = dict(color = 'blue', width = 2, dash='dot'),
        hovertemplate =
        'Predicted Price: %{y:.2f}<extra></extra>'
    )
    trace1 = go.Scatter(
        x = df_energy.index,
        y = df_energy['price actual'],
        mode = 'lines',
        name = 'Actual Price',
        line = dict(color = 'red', width = 2),
        hovertemplate =
        'Actual Price: %{y:.2f}<extra></extra>'
    )

    # Layout
    layout = go.Layout(
        title = 'Comparison of Predicted and Actual Prices',
        xaxis = dict(title = 'Index'),
        yaxis = dict(title = 'Price'),
        hovermode = 'x unified'
    )

    # Figure
    fig = go.Figure(data=[trace0, trace1], layout=layout)

    # Metrics output
    metrics = {
        'RMSE': rmse,
        'R-squared': r_squared,
        'MAE': mae
    }

    return fig, metrics


def plot_interactive_electricity_prices(df):
    fig = px.line(df, x='time', y='price actual', 
                  labels={'time': 'Date', 'price actual': 'Price'},
                  title='Electricity Prices Over Time')
    fig.update_traces(mode='lines+markers', hoverinfo='all')
    # The fig is returned and not shown here; Streamlit will handle the display
    return fig


def interactive_boxplot_by_hour(df, x_column, y_column, title='Distribution of Actual Price by Hour'):
    """
    Creates an interactive box plot of a specified y value by hour.

    Parameters:
    - df: pandas DataFrame containing the data.
    - x_column: str, name of the column in df for x-axis (hour).
    - y_column: str, name of the column in df for y-axis (value to plot).
    - title: str, title of the plot.
    """
    fig = px.box(df, x=x_column, y=y_column, points="all",
                 labels={
                     x_column: 'Hour of Day',
                     y_column: 'Price (€/MWh)'
                 },
                 title=title)
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=list(range(24)), title='Hour of Day'),
        yaxis=dict(title='Price (€/MWh)'),
        hovermode='closest'
    )
    fig.update_traces(hoverinfo="all", hovertemplate="Hour: %{x}<br>Price: %{y}")
    
    return fig

def interactive_line_graph_from_df(df, hour_column, price_column, title='Average Actual Price by Hour'):
    """
    Creates an interactive line graph showing the average price by hour from a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - hour_column: str, name of the column in df that represents the hour of the day.
    - price_column: str, name of the column in df that represents the price.
    - title: str, title of the plot.
    """
    # Group by the hour column and calculate the mean price
    avg_price_by_hour = df.groupby(hour_column)[price_column].mean()
    
    # Reset the index if avg_price_by_hour is a Series to ensure it has a column to plot
    if isinstance(avg_price_by_hour, pd.Series):
        avg_price_by_hour = avg_price_by_hour.reset_index()

    # Create a Plotly graph object for the line graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=avg_price_by_hour[hour_column], y=avg_price_by_hour[price_column],
                             mode='lines+markers',  # Line plot with markers
                             line=dict(color='blue'),  # Line color
                             marker=dict(color='blue', size=8),  # Marker style
                             ))

    # Update layout with titles and axis labels
    fig.update_layout(title=title,
                      xaxis_title='Hour of Day',
                      xaxis=dict(tickmode='array', tickvals=list(range(24))),
                      yaxis_title='Average Price (€/MWh)',
                      hovermode='x unified')  # Unified hover

    # Update traces for hover info
    fig.update_traces(hoverinfo="x+y", hovertemplate='Hour: %{x}<br>Average Price: %{y} €/MWh')
    return fig

def interactive_nuclear_generation_plot(df, time_column, generation_column):
    """
    Creates an interactive line graph showing nuclear generation over time.

    Parameters:
    - df: pandas DataFrame containing the data.
    - time_column: str, name of the column in df that represents the time.
    - generation_column: str, name of the column in df that represents nuclear generation.
    """
    # Ensure the DataFrame is sorted by time and the time column is in datetime format
    df = df.sort_values(time_column)
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Create the figure
    fig = go.Figure()

    # Optionally apply a rolling window for smoothing the data
    window_size = 24  # for example, a 24-hour rolling window
    df['smoothed_generation'] = df[generation_column].rolling(window=window_size, center=True).mean()

    # Add a trace for the line graph using the smoothed data
    fig.add_trace(go.Scatter(
        x=df[time_column], 
        y=df['smoothed_generation'], 
        mode='lines',  # Ensures that lines are drawn
        name='Total Nuclear',
        line=dict(shape='spline')  # Use 'spline' for smoothed lines
    ))

    # Update layout with titles and axis labels
    fig.update_layout(
        title='Total Nuclear Generation Over Time',
        xaxis_title='Time',
        yaxis_title='Generation (MW)',
        hovermode='x unified'  # Unified hover
    )

    # Update traces for hover info
    fig.update_traces(hoverinfo="x+y", hovertemplate='Time: %{x}<br>Generation: %{y} MW')

    return fig


def boxplot_nuclear_generation_by_hour(df, hour_column, generation_column):
    """
    Creates a Plotly boxplot of nuclear generation by hour.

    Parameters:
    - df: pandas DataFrame containing the data.
    - hour_column: str, name of the column in df that represents the hour.
    - generation_column: str, name of the column in df that represents nuclear generation.

    Returns:
    - fig: The Plotly figure object containing the plot.
    """
    fig = px.box(df, x=hour_column, y=generation_column, 
                 title='Distribution of Nuclear Generation by Hour', 
                 labels={hour_column: 'Hour of Day', generation_column: 'Generation (MW)'})
    
    # Update the layout if necessary
    fig.update_layout(xaxis=dict(title='Hour of Day', tickmode='array', tickvals=list(range(24))),
                      yaxis=dict(title='Generation (MW)'),
                      showlegend=False)
    
    # Return the figure object for use with st.plotly_chart in Streamlit
    return fig

def plot_solar_generation_by_hour_interactive(df):
    # Create the Plotly figure
    fig = px.box(df, x='hour', y='generation solar',
                 labels={'generation solar': 'Solar Generation (MWh)', 'hour': 'Hour of Day'},
                 title='Interactive Distribution of Solar Generation by Hour')
    # Customize the figure as needed
    fig.update_layout(hovermode='closest')  # Enhance hover interaction
    return fig

def plot_hydro_generation_by_hour(df):
    # Calculate the total hydro generation
    df['hydro_total'] = df['generation hydro pumped storage consumption'] + df['generation hydro run-of-river and poundage'] + df['generation hydro water reservoir']
    
    # Create the Plotly figure for the box plot
    fig = px.box(df, x='hour', y='hydro_total', labels={'hydro_total': 'Hydro Generation (MWh)', 'hour': 'Hour of Day'},
                 title='Interactive Distribution of Hydro Generation by Hour')
    # Customize the figure as needed, e.g., updating layout
    fig.update_layout(hovermode='closest')  # Enhance hover interaction
    return fig

def plot_wind_generation_by_hour(df):
       
    # Create the Plotly figure for the box plot
    fig = px.box(df, x='hour', y='generation wind onshore', labels={'generation wind onshore': 'Wind Generation (MWh)', 'hour': 'Hour of Day'},
                 title='Interactive Distribution of Wind Generation by Hour')
    # Customize the figure as needed, e.g., updating layout
    fig.update_layout(hovermode='closest')  # Enhance hover interaction
    return fig

def plot_fossil_generation_by_hour(df):
    # Calculate the total hydro generation
    df['fossil_total'] = df['generation fossil brown coal/lignite'] + df['generation fossil gas'] + df[ 'generation fossil hard coal']+ df[ 'generation fossil oil']
    
    # Create the Plotly figure for the box plot
    fig = px.box(df, x='hour', y='fossil_total', labels={'fossil_total': 'Hydro Generation (MWh)', 'hour': 'Hour of Day'},
                 title='Interactive Distribution of Fossil Fuel Electricity Generation by Hour')
    # Customize the figure as needed, e.g., updating layout
    fig.update_layout(hovermode='closest')  # Enhance hover interaction
    return fig

def plot_average_hourly_generation(df):
    # Calculate the total for non-renewables and renewables
    df['total_non_renewables'] = df['generation biomass'] + df[ 'generation fossil brown coal/lignite'] + df['generation fossil gas'] + df['generation fossil hard coal'] + df['generation fossil oil'] + df[ 'generation nuclear'] + df[ 'generation other']
    df['total_renewables'] = df['generation hydro pumped storage consumption'] + df['generation hydro run-of-river and poundage'] + df['generation hydro water reservoir'] + df[ 'generation other renewable'] + df['generation solar'] + df['generation waste'] + df['generation wind onshore']
    
    # Group by hour of day and calculate the mean
    hourly_means = df[['total_non_renewables', 'total_renewables']].groupby(df['hour']).mean()
    
    # Create the figure for hourly means
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_means.index, y=hourly_means['total_non_renewables'],
                             mode='lines+markers', name='Non-Renewable Generation',
                             line=dict(color='red')))
    fig.add_trace(go.Scatter(x=hourly_means.index, y=hourly_means['total_renewables'],
                             mode='lines+markers', name='Renewable Generation',
                             line=dict(color='green')))
    
    # Update layout
    fig.update_layout(
        title='Average Hourly Energy Generation',
        xaxis_title='Hour of Day',
        yaxis_title='Average Generation (MW)',
        legend_title='Generation Type',
        hovermode='closest'
    )
    
    return fig

def plot_average_daily_generation(df):
    # Calculate the total for non-renewables and renewables
    df['total_non_renewables'] = df['generation biomass'] + df[ 'generation fossil brown coal/lignite'] + df['generation fossil gas'] + df['generation fossil hard coal'] + df['generation fossil oil'] + df[ 'generation nuclear'] + df[ 'generation other']
    df['total_renewables'] = df['generation hydro pumped storage consumption'] + df['generation hydro run-of-river and poundage'] + df['generation hydro water reservoir'] + df[ 'generation other renewable'] + df['generation solar'] + df['generation waste'] + df['generation wind onshore']
    df.set_index('time', inplace=True)
    
    # Group by day and calculate the mean
    daily_means = df[['total_non_renewables', 'total_renewables']]
    
    # Create the figure for daily means
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_means.index, y=daily_means['total_non_renewables'],
                             mode='lines+markers', name='Non-Renewable Generation',
                             line=dict(color='red')))
    fig.add_trace(go.Scatter(x=daily_means.index, y=daily_means['total_renewables'],
                             mode='lines+markers', name='Renewable Generation',
                             line=dict(color='green')))
    
    # Update layout
    fig.update_layout(
        title='Average Daily Energy Generation',
        xaxis_title='Day',
        yaxis_title='Average Generation (MW)',
        legend_title='Generation Type',
        hovermode='closest'
    )
    
    return fig

def plot_avg_generation_per_day(df):
    # Ensure the 'time' column is in datetime format
    df.index = pd.to_datetime(df.index)

    # Create a 'day_of_week' column representing the day of the week
    df['day_of_week'] = df.index.dayofweek

    # Group by 'day_of_week' and calculate the average generation
    avg_per_day_of_week = df.groupby('day_of_week')[['total_non_renewables', 'total_renewables']].mean()

    # Map integer days to actual day names
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    avg_per_day_of_week.index = avg_per_day_of_week.index.map(lambda x: days[x])

    # Create the figure
    fig = go.Figure(data=[
        go.Bar(name='Non-Renewable Generation', x=avg_per_day_of_week.index, y=avg_per_day_of_week['total_non_renewables'], marker_color='red'),
        go.Bar(name='Renewable Generation', x=avg_per_day_of_week.index, y=avg_per_day_of_week['total_renewables'], marker_color='green')
    ])
    
    # Update layout for the figure
    fig.update_layout(
        barmode='group',
        title='Average Energy Generation per Day of the Week',
        xaxis_title='Day of the Week',
        yaxis_title='Average Generation (MW)',
        xaxis=dict(tickangle=45),
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
    )

    return fig

def plot_comparison_over_time(df):
    fig = go.Figure()
    # Add traces for price and load
    fig.add_trace(go.Scatter(x=df.index, y=df['price actual'], name='Price Actual (€/MWh)', yaxis='y1', mode='lines', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['total load actual'], name='Total Load Actual (MW)', yaxis='y2', mode='lines', line=dict(width=2)))

    # Layout for two y-axes
    fig.update_layout(
        title='Comparison of Price and Total Load Over Time',
        xaxis_title='Time',
        yaxis=dict(title='Price Actual (€/MWh)', side='left', showgrid=False),
        yaxis2=dict(title='Total Load Actual (MW)', side='right', overlaying='y', showgrid=False),
        legend=dict(x=0.03, y=0.97, bordercolor="Black", borderwidth=1),
        hovermode='x'
    )

    return fig

def generate_energy_generation_graph(df):
    """
    Generates an interactive stacked graph with smoothed lines, visualizing weekly average energy generation 
    from various sources over time, with detailed hover information.
    
    Parameters:
    - df: DataFrame containing the energy generation data and a 'time' column.
    
    Returns:
    - A Plotly Figure object containing the interactive stacked graph with smoothed lines and weekly averages.
    """
    # Ensure the 'time' column is in datetime format and set it as the DataFrame index
    # df['time'] = pd.to_datetime(df['time'])
    # df.set_index('time', inplace=True)

    # Define the energy generation columns to be plotted
    columns = [
        'generation biomass', 
        'generation fossil brown coal/lignite',
        'generation fossil gas', 
        'generation fossil hard coal',
        'generation fossil oil', 
        'generation hydro pumped storage consumption',
        'generation hydro run-of-river and poundage',
        'generation hydro water reservoir', 
        'generation nuclear',
        'generation other', 
        'generation other renewable', 
        'generation solar',
        'generation waste', 
        
    ]

    # Aggregate data by week for each column
    weekly_df = df[columns].resample('W').mean()

    # Reset the index to get the 'time' column back for plotting
    weekly_df.reset_index(inplace=True)

    # Create the figure
    fig = go.Figure()

    # Add each generation type as a separate trace with a detailed hover template
    for column in columns:
        fig.add_trace(
            go.Scatter(
                x=weekly_df['time'], 
                y=weekly_df[column], 
                name=column, 
                stackgroup='one',
                mode='lines',  # Use lines instead of markers
                line_shape='spline',  # Smooth the line
                hoverinfo='x+y+name',
                hovertemplate=f"{column}:<br>%{{x}}<br>%{{y:.2f}} MWh<extra></extra>",
            )
        )

    # Update the layout to enhance the graph
    fig.update_layout(
        title='Weekly Average Energy Generation Over Time',
        xaxis_title='Time',
        yaxis_title='Generation (MWh)',
        hovermode="x unified",
        xaxis={'showgrid': False, 'type': 'date'},
        yaxis={'showgrid': False},
        legend_title_text='Energy Source'
    )

    return fig


def interactive_comparison_chart(df):
    """
    Interactive chart in Streamlit for comparing selected columns with options for data aggregation frequency,
    including a time range slider for selecting start and end times, and a button to revert to the full time series.
    
    Parameters:
    - df: The DataFrame containing the data, including a 'time' column.
    """
    # Convert 'time' to datetime if not already and remove timezone information for simplicity
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    
    # UI for column selection
    all_columns = df.columns.drop('time')  # Assumes 'time' is not a column to be compared
    selected_columns = st.multiselect('Select columns to display:', options=all_columns, default=[all_columns[0]])
    
    # UI for frequency selection
    aggregation = st.radio('Select aggregation level:', options=['Hourly', 'Daily', 'Weekly'], index=0)

    # Convert datetimes to Unix timestamps for the slider
    min_time = df['time'].min().timestamp()
    max_time = df['time'].max().timestamp()
    
    # Time range selection using Unix timestamps
    start_time, end_time = st.slider("Select time range:", 
                                     min_value=int(min_time), 
                                     max_value=int(max_time), 
                                     value=(int(min_time), int(max_time)), 
                                     format="MM/DD/YY")
    
    # Convert Unix timestamps back to datetimes for filtering
    start_time = pd.to_datetime(start_time, unit='s')
    end_time = pd.to_datetime(end_time, unit='s')
    
    # Filter dataframe based on selected time range
    df_filtered = df[(df['time'] >= start_time) & (df['time'] <= end_time)]

    # Data aggregation based on the selected frequency
    df_filtered.set_index('time', inplace=True)
    if aggregation == 'Daily':
        df_agg = df_filtered.resample('D').mean()
    elif aggregation == 'Weekly':
        df_agg = df_filtered.resample('W').mean()
    else:  # Hourly or default
        df_agg = df_filtered  # Assuming data is already in hourly frequency if not daily or weekly

    df_agg.reset_index(inplace=True)
    
    # Plotting
    fig = go.Figure()
    for column in selected_columns:
        if column in df_agg.columns:
            fig.add_trace(go.Scatter(x=df_agg['time'], y=df_agg[column], mode='lines+markers', name=column,
                                     hoverinfo='x+y+name', hovertemplate=f"{column}: %{{y:.2f}}<extra></extra>"))
    
    fig.update_layout(title='Data Comparison Over Time',
                      xaxis_title='Time',
                      yaxis_title='Values',
                      hovermode="x unified",
                      legend_title_text='Data Type')
    return fig
