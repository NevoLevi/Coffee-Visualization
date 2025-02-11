import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import requests
import io
import numpy as np
from typing import Tuple, List, Optional
from streamlit_plotly_events import plotly_events
from plotly.subplots import make_subplots


# Data loading and preprocessing
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Load and preprocess all required datasets."""
    datasets = {
        'psd_coffee': 'data/psd_coffee.csv',
        'arabica_clean': 'data/df_arabica_clean.csv',
        'importers_consumption': 'data/Coffee_importers_consumption.csv',  # Update with correct path
        're_export': 'data/Coffee_re_export.csv',
        'domestic_consumption': 'data/Coffee_domestic_consumption.csv',
        'coffee_import': 'data/Coffee_import.csv'  # New import file
    }

    data = {name: pd.read_csv(file) for name, file in datasets.items()}

    # Scale the data as required
    data['psd_coffee'].iloc[:, 2:] *= 1000  # Multiply relevant columns by 1000
    for df_name in ['importers_consumption', 're_export', 'domestic_consumption', 'coffee_import']:
        df = data[df_name]
        if df_name == 'domestic_consumption':
            year_columns = df.columns[2:]  # Exclude 'Country', 'Coffee type', include Total domestic consumption
        else:
            year_columns = df.columns[1:]  # Exclude 'Country', include Total domestic consumption
        df[year_columns] = df[year_columns] / 60
        data[df_name] = df

    # Clean up column names for relevant datasets
    for df_name in ['importers_consumption', 're_export', 'domestic_consumption', 'coffee_import']:
        data[df_name].columns = data[df_name].columns.str.strip()

    # Load world map data
    world_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    response = requests.get(world_url)
    world = gpd.read_file(io.StringIO(response.content.decode('utf-8')))

    # Load population data
    population = pd.read_csv('data/WorldPopulation2023.csv')

    return tuple(data.values()) + (world, population)


def create_top_producers_chart(psd_coffee: pd.DataFrame) -> go.Figure:
    """Create a line chart of top coffee producing countries."""
    top_producers = psd_coffee.groupby('Country')['Production'].sum().nlargest(5).index
    top_producers_data = psd_coffee[psd_coffee['Country'].isin(top_producers)]

    return px.line(
        top_producers_data,
        x='Year',
        y='Production',
        color='Country',
        title="Top 5 Coffee Producing Countries (1960-2023)",
        labels={'Production': 'Production (60kg bags)'}
    )


def create_altitude_quality_scatter(arabica_clean: pd.DataFrame, selected_countries: List[str]) -> go.Figure:
    """Create an interactive scatter plot of altitude vs coffee quality."""
    filtered_data = arabica_clean[arabica_clean['Country of Origin'].isin(selected_countries)]

    fig = px.scatter(
        filtered_data,
        x='Altitude',
        y='Total Cup Points',
        color='Country of Origin',
        hover_data=['Variety', 'Processing Method'],
        title="Altitude vs. Coffee Quality"
    )

    fig.update_layout(
        xaxis_title="Altitude (meters above sea level)",
        yaxis_title="Total Cup Points",
        legend_title="Country of Origin"
    )

    return fig


def create_processing_quality_chart(arabica_clean: pd.DataFrame) -> go.Figure:
    """Create a box plot of processing method vs coffee quality."""
    # Filter for specific processing methods
    methods = ['Washed / Wet', 'Natural / Dry', 'Pulped natural / honey']
    filtered_data = arabica_clean[arabica_clean['Processing Method'].isin(methods)]

    # Create a box plot
    fig = px.box(
        filtered_data,
        x='Processing Method',
        y='Total Cup Points',
        color='Processing Method',
        title="Coffee Quality by Processing Method",
        labels={'Total Cup Points': 'Quality Score'},
        category_orders={'Processing Method': methods},
        points=False
    )

    fig.update_layout(showlegend=False)

    # Add sample size annotations
    for method in methods:
        sample_size = filtered_data[filtered_data['Processing Method'] == method].shape[0]
        fig.add_annotation(
            x=method,
            y=filtered_data[filtered_data['Processing Method'] == method]['Total Cup Points'].max(),
            text=f"n={sample_size}",
            showarrow=False,
            yshift=10
        )

    return fig


def categorize_color(color: str) -> str:
    """Categorize the color of coffee beans."""
    if pd.isna(color):
        return 'Unknown'
    color = color.lower()
    if 'blue' in color:
        return 'Blue-green'
    elif 'yello' in color and 'green' in color:
        return 'Yellow-green'
    elif 'green' in color:
        return 'Green'
    elif 'yellow' in color:
        return 'Yellow'
    elif 'brow' in color:
        return 'Brown'
    else:
        return 'Other'


def create_color_quality_chart(arabica_clean: pd.DataFrame):
    """Create a box plot of bean color vs coffee quality with matched colors."""
    arabica_clean['Color Category'] = arabica_clean['Color'].apply(categorize_color)

    # Define a color map that matches category names
    color_map = {
        'Blue-green': '#83b7a8',  # Blue-green color
        'Yellow-green': '#9acd32',  # Yellow-green color
        'Green': '#008000',  # Green color
        'Yellow': '#EFE719',  # Yellow color
        'Brown': '#8b4513'  # Brown color
    }

    # Filter out 'Unknown' and 'Other' categories
    filtered_data = arabica_clean[arabica_clean['Color Category'].isin(color_map.keys())]

    fig = px.box(
        filtered_data,
        x='Color Category',
        y='Total Cup Points',
        color='Color Category',
        title="Coffee Quality by Bean Color",
        labels={'Total Cup Points': 'Quality Score'},
        category_orders={'Color Category': list(color_map.keys())},
        color_discrete_map=color_map,
        points=False
    )

    fig.update_layout(showlegend=False)

    # Add sample size annotations
    for color in filtered_data['Color Category'].unique():
        sample_size = filtered_data[filtered_data['Color Category'] == color].shape[0]
        fig.add_annotation(
            x=color,
            y=filtered_data[filtered_data['Color Category'] == color]['Total Cup Points'].max(),
            text=f"n={sample_size}",
            showarrow=False,
            yshift=10
        )

    return fig


def create_flavor_profile_radar(arabica_clean: pd.DataFrame, countries: List[str]) -> go.Figure:
    """Create a radar chart of coffee flavor profiles by country with scaling from 6 to 10."""
    attributes = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']

    # Filter data for selected countries
    selected_data = arabica_clean[arabica_clean['Country of Origin'].isin(countries)][
        attributes + ['Country of Origin', 'Total Cup Points']]

    # Define the new scaling range
    scale_min = 6
    scale_max = 10

    # Apply custom scaling to the attributes
    def scale_attribute(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value) * (scale_max - scale_min) + scale_min

    scaled_data = selected_data.copy()
    for attribute in attributes:
        min_value = selected_data[attribute].min()
        max_value = selected_data[attribute].max()
        scaled_data[attribute] = selected_data[attribute].apply(scale_attribute, args=(min_value, max_value))

    fig = go.Figure()

    for country in countries:
        country_data = scaled_data[scaled_data['Country of Origin'] == country][attributes].mean()
        fig.add_trace(go.Scatterpolar(
            r=country_data.values,
            theta=attributes,
            fill='toself',
            name=country
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[scale_min, scale_max])),
        showlegend=True,
        title="Average Coffee Flavor Profile by Country (Scaled 6-10)"
    )

    return fig


def create_parallel_coordinates_plot(arabica_clean: pd.DataFrame, countries: List[str]) -> go.Figure:
    """Create a parallel coordinates plot of coffee flavor profiles by country."""
    attributes = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']

    # Filter data for selected countries
    selected_data = arabica_clean[arabica_clean['Country of Origin'].isin(countries)][
        attributes + ['Country of Origin']]

    fig = px.parallel_coordinates(selected_data, color="Country of Origin", dimensions=attributes,
                                  color_discrete_sequence=px.colors.qualitative.Bold)

    fig.update_layout(
        title="Coffee Flavor Profiles by Country (Parallel Coordinates Plot)"
    )

    return fig


def create_flavor_profile_bar_plot(arabica_clean: pd.DataFrame, country1: str, country2: str) -> go.Figure:
    """Create a bar plot of coffee flavor profiles by country with dynamic scaling."""
    attributes = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']

    # Filter data for the selected countries
    selected_data = arabica_clean[arabica_clean['Country of Origin'].isin([country1, country2])]

    # Calculate the dynamic scale based on the selected countries
    min_value = selected_data[attributes].mean().min()
    max_value = selected_data[attributes].mean().max()

    fig = go.Figure()

    for country in [country1, country2]:
        country_data = selected_data[selected_data['Country of Origin'] == country][attributes].mean()
        fig.add_trace(go.Bar(
            x=attributes,
            y=country_data.values,
            name=country
        ))

    fig.update_layout(
        title="Average Coffee Flavor Profile by Country",
        xaxis_title="Attributes",
        yaxis_title="Score",
        yaxis=dict(range=[min_value - 0.5, max_value + 0.5]),
        barmode='group'
    )

    return fig


def display_total_cup_points(arabica_clean: pd.DataFrame, country1: str, country2: str):
    """Display the average total cup points for the selected countries."""
    selected_data = arabica_clean[arabica_clean['Country of Origin'].isin([country1, country2])]
    avg_cup_points = selected_data.groupby('Country of Origin')['Total Cup Points'].mean()

    st.markdown("<h4 style='text-align: center;'>Average Total Cup Points</h4>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{country1}: {avg_cup_points[country1]:.2f}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{country2}: {avg_cup_points[country2]:.2f}</h3>", unsafe_allow_html=True)


def create_import_trends_chart(psd_coffee: pd.DataFrame, selected_countries: List[str],
                               all_countries: bool) -> go.Figure:
    """Create a line chart of coffee import trends by type."""
    # Filter data for selected countries
    df = psd_coffee[psd_coffee['Country'].isin(selected_countries)]

    # Group by year and sum the import values
    df_grouped = df.groupby('Year').agg({
        'Bean Imports': 'sum',
        'Rst,Ground Dom. Consum': 'sum',
        'Soluble Imports': 'sum'
    }).reset_index()

    # Create the line chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_grouped['Year'], y=df_grouped['Bean Imports'],
        mode='lines',
        line=dict(width=2, color='rgb(131, 90, 241)'),
        name='Green Coffee Beans'
    ))
    fig.add_trace(go.Scatter(
        x=df_grouped['Year'], y=df_grouped['Rst,Ground Dom. Consum'],
        mode='lines',
        line=dict(width=2, color='rgb(111, 231, 219)'),
        name='Roasted and Ground Coffee'
    ))
    fig.add_trace(go.Scatter(
        x=df_grouped['Year'], y=df_grouped['Soluble Imports'],
        mode='lines',
        line=dict(width=2, color='rgb(255, 183, 77)'),
        name='Soluble Coffee'
    ))

    fig.update_layout(
        title='Coffee Import Trends by Type' + (' (All Countries)' if all_countries else ''),
        xaxis_title='Year',
        yaxis_title='Import Volume (60kg bags)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def create_multiline_chart(psd_coffee: pd.DataFrame, selected_countries: List[str]) -> go.Figure:
    """Create a multi-line chart with dual Y-axes for Arabica and Robusta production."""
    fig = go.Figure()

    for country in selected_countries:
        country_data = psd_coffee[psd_coffee['Country'] == country]

        # Arabica production line
        fig.add_trace(go.Scatter(
            x=country_data['Year'],
            y=country_data['Arabica Production'],
            name=f'{country} - Arabica',
            line=dict(width=2)
        ))

        # Robusta production line
        fig.add_trace(go.Scatter(
            x=country_data['Year'],
            y=country_data['Robusta Production'],
            name=f'{country} - Robusta',
            line=dict(width=2, dash='dash')
        ))

    fig.update_layout(
        title='Arabica and Robusta Production by Country',
        xaxis_title='Year',
        yaxis_title='Production Volume (60kg bags)',
        legend_title='Country and Coffee Type',
        hovermode='x unified'
    )

    return fig



def create_top_consumers_per_capita_chart(psd_coffee: pd.DataFrame, population: pd.DataFrame) -> go.Figure:
    """Create a bar chart of top coffee consuming countries per capita."""
    latest_year = psd_coffee['Year'].max()
    consumption_data = psd_coffee[psd_coffee['Year'] == latest_year]

    # Merge with population data
    consumption_data = consumption_data.merge(population, left_on='Country', right_on='Country')

    # Calculate per capita consumption
    consumption_data['Consumption per Capita'] = 60 * (
            consumption_data['Domestic Consumption'] / consumption_data['Population2023'])

    # Select top 10 countries based on per capita consumption
    top_consumers_per_capita = consumption_data.nlargest(10, 'Consumption per Capita')

    return px.bar(
        top_consumers_per_capita,
        x='Country',
        y='Consumption per Capita',
        title=f"Top 10 Coffee Consuming Countries per Capita in {latest_year}",
        labels={'Consumption per Capita': 'Consumption per Capita (kg/person)'}
    )


def create_help_tooltip(help_text: str) -> str:
    """Create a help tooltip with the given text."""
    return f"""
    <div class="tooltip">
        <span class="help-icon">?</span>
        <span class="tooltiptext">{help_text}</span>
    </div>
    """


def subheader_with_tooltip(header_text: str, tooltip_text: str) -> None:
    """Create a subheader with an inline help tooltip."""
    st.markdown(f"""
    <style>
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: help;
        margin-left: 5px;
    }}
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 60%;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: fixed;
        z-index: 1;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        opacity: 0;
        transition: opacity 0.3s;
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    .tooltip .help-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background-color: #f0f0f0;
        color: #333;
        font-size: 14px;
        font-weight: bold;
    }}
    .subheader-container {{
        display: flex;
        align-items: center;
    }}
    </style>
    <div class="subheader-container">
        <h3>{header_text}</h3>{create_help_tooltip(tooltip_text)}
    </div>
    """, unsafe_allow_html=True)


def create_top_consumers_line_chart(importers_consumption: pd.DataFrame) -> go.Figure:
    """Create a line chart showing consumption trends for top 7 importing countries."""
    # Select top 7 countries based on total import consumption
    top_countries = importers_consumption.nlargest(7, 'Total_import_consumption')

    # Prepare the data for plotting
    years = importers_consumption.columns[1:-1]  # Exclude 'Country' and 'Total_import_consumption'
    fig = go.Figure()

    for _, row in top_countries.iterrows():
        fig.add_trace(go.Scatter(
            x=years,
            y=row[years],
            mode='lines+markers',
            name=row['Country']
        ))

    # Customize x-axis to show every 5 years
    fig.update_layout(
        title='Coffee Consumption Trends for Top 7 Importing Countries',
        xaxis_title='Year',
        yaxis_title='Consumption (60kg bags)',
        xaxis=dict(
            tickmode='array',
            tickvals=[year for year in years if int(year) % 5 == 0],  # Show every 5 years
            ticktext=[year for year in years if int(year) % 5 == 0]
        ),
        template='plotly_dark'
    )

    return fig

def create_continent_consumption_area(psd_coffee: pd.DataFrame) -> go.Figure:
    """Create a stacked area chart showing the proportion of global coffee consumption by continent over time."""
    continent_mapping = {
        'Albania': 'Europe', 'Algeria': 'Africa', 'Angola': 'Africa', 'Argentina': 'South America',
        'Armenia': 'Asia', 'Australia': 'Oceania', 'Benin': 'Africa', 'Bolivia': 'South America',
        'Bosnia and Herzegovina': 'Europe', 'Brazil': 'South America', 'Burundi': 'Africa',
        'Cameroon': 'Africa', 'Canada': 'North America', 'Central African Republic': 'Africa',
        'Chile': 'South America', 'China': 'Asia', 'Colombia': 'South America',
        'Congo (Brazzaville)': 'Africa', 'Congo (Kinshasa)': 'Africa', 'Costa Rica': 'North America',
        'Cote d\'Ivoire': 'Africa', 'Croatia': 'Europe', 'Cuba': 'North America',
        'Dominican Republic': 'North America', 'Ecuador': 'South America', 'Egypt': 'Africa',
        'El Salvador': 'North America', 'Equatorial Guinea': 'Africa', 'Ethiopia': 'Africa',
        'European Union': 'Europe', 'Gabon': 'Africa', 'Georgia': 'Asia', 'Ghana': 'Africa',
        'Guatemala': 'North America', 'Guinea': 'Africa', 'Guyana': 'South America', 'Haiti': 'North America',
        'Honduras': 'North America', 'India': 'Asia', 'Indonesia': 'Asia', 'Iran': 'Asia',
        'Jamaica': 'North America', 'Japan': 'Asia', 'Jordan': 'Asia', 'Kazakhstan': 'Asia',
        'Kenya': 'Africa', 'Korea, South': 'Asia', 'Kosovo': 'Europe', 'Laos': 'Asia',
        'Liberia': 'Africa', 'Madagascar': 'Africa', 'Malawi': 'Africa', 'Malaysia': 'Asia',
        'Mexico': 'North America', 'Montenegro': 'Europe', 'Morocco': 'Africa',
        'New Caledonia': 'Oceania', 'New Zealand': 'Oceania', 'Nicaragua': 'North America',
        'Nigeria': 'Africa', 'North Macedonia': 'Europe', 'Norway': 'Europe', 'Panama': 'North America',
        'Papua New Guinea': 'Oceania', 'Paraguay': 'South America', 'Peru': 'South America',
        'Philippines': 'Asia', 'Russia': 'Europe', 'Rwanda': 'Africa', 'Saudi Arabia': 'Asia',
        'Senegal': 'Africa', 'Serbia': 'Europe', 'Sierra Leone': 'Africa', 'Singapore': 'Asia',
        'South Africa': 'Africa', 'Sri Lanka': 'Asia', 'Switzerland': 'Europe', 'Taiwan': 'Asia',
        'Tanzania': 'Africa', 'Thailand': 'Asia', 'Togo': 'Africa', 'Trinidad and Tobago': 'North America',
        'Turkey': 'Asia', 'Uganda': 'Africa', 'Ukraine': 'Europe', 'United Kingdom': 'Europe',
        'United States': 'North America', 'Uruguay': 'South America', 'Venezuela': 'South America',
        'Vietnam': 'Asia', 'Yemen': 'Asia', 'Yemen (Sanaa)': 'Asia', 'Zambia': 'Africa', 'Zimbabwe': 'Africa'
    }

    df = psd_coffee.copy()
    df['Continent'] = df['Country'].map(continent_mapping)

    # Ensure 'Year' column is present and in the correct format
    if 'Year' not in df.columns:
        print("Columns in psd_coffee:", df.columns)
        raise ValueError("'Year' column not found in psd_coffee dataset")

    # Convert 'Year' to numeric if it's not already
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Ensure 'Domestic Consumption' column is present
    if 'Domestic Consumption' not in df.columns:
        print("Columns in psd_coffee:", df.columns)
        raise ValueError("'Domestic Consumption' column not found in psd_coffee dataset")

    df_grouped = df.groupby(['Continent', 'Year'])['Domestic Consumption'].sum().reset_index()

    fig = px.area(df_grouped, x='Year', y='Domestic Consumption', color='Continent',
                  title='Global Coffee Consumption by Continent',
                  labels={'Domestic Consumption': 'Coffee Consumption (60kg bags)'})
    return fig


def create_reexport_proportion_chart(coffee_import: pd.DataFrame, re_export: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing the proportion of re-exports to imports for major coffee trading countries."""
    imports = coffee_import.copy()
    imports.set_index('Country', inplace=True)
    imports = imports.drop(columns=['Total_import'])
    imports = imports.reset_index().melt(id_vars=['Country'], var_name='Year', value_name='Imports')

    reexports = re_export.copy()
    reexports.set_index('Country', inplace=True)
    reexports = reexports.drop(columns=['Total_re_export'])
    reexports = reexports.reset_index().melt(id_vars=['Country'], var_name='Year', value_name='Re-exports')

    df = imports.merge(reexports, on=['Country', 'Year'], how='outer')
    df['Year'] = pd.to_numeric(df['Year'])
    df['Re-export Proportion'] = df['Re-exports'] / df['Imports']

    # Select top 8 countries by total trade volume
    top_countries = df.groupby('Country')[['Imports', 'Re-exports']].sum().sum(axis=1).nlargest(8).index
    df_filtered = df[df['Country'].isin(top_countries)]

    latest_year = df_filtered['Year'].max()
    df_latest = df_filtered[df_filtered['Year'] == latest_year]

    fig = px.bar(df_latest, x='Country', y='Re-export Proportion',
                 title=f'Proportion of Re-exports to Imports for Major Coffee Trading Countries ({latest_year})',
                 labels={'Re-export Proportion': 'Re-export / Import Ratio'},
                 color='Re-export Proportion', color_continuous_scale='Viridis')

    fig.update_layout(xaxis_title='Country', yaxis_title='Re-export / Import Ratio')
    return fig


def create_consumption_production_bubble_top10(domestic_consumption: pd.DataFrame,
                                               psd_coffee: pd.DataFrame) -> px.scatter:
    """Create a bubble chart showing domestic consumption vs production for top 10 countries, with bubble size representing exports."""
    consumption = domestic_consumption.groupby('Country')['Total_domestic_consumption'].sum().reset_index()
    production = psd_coffee.groupby('Country')['Production'].sum().reset_index()
    exports = psd_coffee.groupby('Country')['Exports'].sum().reset_index()

    df = consumption.merge(production, on='Country', how='outer')
    df = df.merge(exports, on='Country', how='outer')
    df = df.dropna().nlargest(10, 'Total_domestic_consumption')

    # Create a formatted exports column for hover data
    df['Exports_formatted'] = df['Exports'].apply(format_value)

    fig = px.scatter(df, x='Production', y='Total_domestic_consumption', size='Exports',
                     hover_name='Country',
                     title='Top 10 Countries: Production vs Domestic Consumption (Bubble Size: Exports)',
                     labels={'Total_domestic_consumption': 'Domestic Consumption (60kg bags)',
                             'Production': 'Production (60kg bags)',
                             'Exports': 'Exports (60kg bags)'},
                     hover_data={'Exports_formatted': True, 'Exports': False})

    # Update hover template to rename 'Exports_formatted' to 'Exports' in the tooltip
    fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>' +
                                   'Production: %{x}<br>' +
                                   'Domestic Consumption: %{y}<br>' +
                                   'Exports: %{customdata[0]}<extra></extra>',
                      customdata=df[['Exports_formatted']].values)

    return fig


def create_consumption_production_trend_psd(psd_coffee: pd.DataFrame, selected_countries: List[str]) -> go.Figure:
    """Create a line chart with dual y-axes showing domestic consumption trends and production fluctuations over time using PSD coffee data."""
    filtered_data = psd_coffee[psd_coffee['Country'].isin(selected_countries)]
    df = filtered_data.groupby('Year').agg({
        'Domestic Consumption': 'sum',
        'Arabica Production': 'sum',
        'Robusta Production': 'sum'
    }).reset_index()

    df['Total Production'] = df['Arabica Production'] + df['Robusta Production']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=df['Year'], y=df['Domestic Consumption'], name='Consumption'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Total Production'], name='Production'),
                  secondary_y=False)

    fig.update_layout(title='Coffee Consumption and Production Trends',
                      xaxis_title='Year')
    fig.update_yaxes(title_text="Coffee (60kg bags)", secondary_y=False)

    return fig


def format_value(value):
    """Format value based on its magnitude."""
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"{value / 1e3:.2f}K"
    else:
        return str(value)


def preprocess_data(coffee_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the coffee data to categorize coffee types."""
    coffee_data['Coffee Type Category'] = coffee_data['Coffee type'].apply(
        lambda x: 'Both' if 'Robusta' in x and 'Arabica' in x else x.split('/')[0]
    )
    coffee_data['Total Domestic Consumption'] = coffee_data.filter(regex='^19|^20').sum(axis=1)
    return coffee_data

def create_coffee_type_map(coffee_data: pd.DataFrame, world: gpd.GeoDataFrame) -> px.choropleth:
    """Create a choropleth map showing coffee type and total domestic consumption."""
    coffee_data = preprocess_data(coffee_data)

    # Ensure all countries are included
    world['name'] = world['name'].str.strip()
    coffee_data = world[['name']].merge(coffee_data, how='left', left_on='name', right_on='Country').fillna('No data')

    # Convert total domestic consumption to billions and format as string
    coffee_data['Total Domestic Consumption'] = coffee_data['Total Domestic Consumption'].apply(
        lambda x: format_value(x) if x != 'No data' else 'No data'
    )

    fig = px.choropleth(
        coffee_data,
        geojson=world.set_index('name')['geometry'].__geo_interface__,  # Ensure geojson is correctly linked
        locations='name',  # Use the 'name' column for locations
        color="Coffee Type Category",
        hover_name="name",
        hover_data={
            "Total Domestic Consumption": True,
            "Coffee Type Category": False,
        },
        color_discrete_map={
            'Both': '#636EFA',  # Blue
            'Robusta': '#EF553B',  # Red
            'Arabica': '#00CC96',  # Green
            'No data': '#808080'  # Gray
        },
        projection="natural earth"
    )

    fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")

    return fig


def create_reexport_vs_domestic_chart(coffee_import: pd.DataFrame, re_export: pd.DataFrame, importers_consumption: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing the proportion of re-exports vs domestic consumption for top importing countries in 2019."""

    # Filter for the year 2019
    imports_2019 = coffee_import[['Country', '2019']].rename(columns={'2019': 'Imports_2019'})
    reexports_2019 = re_export[['Country', '2019']].rename(columns={'2019': 'Reexports_2019'})
    consumption_2019 = importers_consumption[['Country', '2019']].rename(columns={'2019': 'Consumption_2019'})

    # Merge data
    df = imports_2019.merge(reexports_2019, on='Country').merge(consumption_2019, on='Country')

    # Calculate proportions
    df['Reexport'] = df['Reexports_2019'] / df['Imports_2019']
    df['Consumption'] = df['Consumption_2019'] / df['Imports_2019']

    # Select top 10 importing countries by import volume in 2019
    top_countries = df.nlargest(8, 'Imports_2019')['Country']
    df_top = df[df['Country'].isin(top_countries)]

    # Melt the dataframe for plotting
    df_melted = df_top.melt(id_vars=['Country'], value_vars=['Reexport', 'Consumption'],
                            var_name='Proportion Type', value_name='Proportion')

    # Create the bar chart
    fig = px.bar(df_melted, x='Country', y='Proportion', color='Proportion Type', barmode='group',
                 title='Proportion of Re-exports vs Domestic Consumption for Top Importing Countries (2019)',
                 labels={'Proportion': 'Proportion of Imports'},
                color_discrete_map = {'Reexport Proportion': 'rgb(196, 12, 242)',
                                        'Consumption Proportion': 'rgb(234, 111, 238)'})

    fig.update_layout(xaxis_title='Country', yaxis_title='Proportion of Imports')

    return fig


def create_custom_tabs(tab_names):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(tab_names[0], key=tab_names[0], use_container_width=True):
            st.session_state.active_tab = tab_names[0]
    with col2:
        if st.button(tab_names[1], key=tab_names[1], use_container_width=True):
            st.session_state.active_tab = tab_names[1]
    with col3:
        if st.button(tab_names[2], key=tab_names[2], use_container_width=True):
            st.session_state.active_tab = tab_names[2]

    st.markdown("""
    <style>
    div.stButton > button {
        font-size: 24px;
        font-weight: bold;
        height: 3em;
        border: 2px solid #4CAF50;
        border-radius: 5px;
    }
    div.stButton > button:hover {
        background-color: #4CAF50;
        color: white;
    }
    div.stButton > button:focus:not(:active) {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)



def create_interactive_trading_map(psd_coffee: pd.DataFrame, world: gpd.GeoDataFrame,
                                   year_range: Tuple[int, int]) -> px.choropleth:
    """Create an interactive choropleth map with clickable countries to show aggregated coffee trading data."""
    trading_data = psd_coffee[(psd_coffee['Year'] >= year_range[0]) & (psd_coffee['Year'] <= year_range[1])]
    aggregated_data = trading_data.groupby('Country').agg({
        'Exports': 'sum',
        'Imports': 'sum',
        'Production': 'sum',
        'Domestic Consumption': 'sum'
    }).reset_index()

    # Add a small value to avoid log(0)
    aggregated_data['Trading'] = aggregated_data['Exports'] + aggregated_data['Imports']

    # Extract European Union data
    eu_data = aggregated_data[aggregated_data['Country'] == 'European Union']
    eu_data = eu_data.iloc[0]
    eu_data_dict = {
        'name': 'European Union',
        'Trading': eu_data['Trading'],
        'Log_Trading': np.log1p(eu_data['Trading']),
        'Latitude': 50.8503,  # Example latitude of Brussels, Belgium (EU headquarters)
        'Longitude': 4.3517  # Example longitude of Brussels, Belgium (EU headquarters)
    }

    # Ensure all countries are included in the world dataframe
    world['name'] = world['name'].str.strip()
    aggregated_data = world[['name']].merge(aggregated_data, how='left', left_on='name', right_on='Country').fillna(0)

    # Normalize Trading values for color scaling
    aggregated_data['Log_Trading'] = np.log1p(aggregated_data['Trading'])

    # Add European Union data to the aggregated dataset
    eu_df = pd.DataFrame([eu_data_dict])
    aggregated_data = pd.concat([aggregated_data, eu_df], ignore_index=True)

    # Define the custom steps and color scale
    steps = [1, 100000, 1000000, 10000000, 100000000, 1000000000]
    colors = [
        "#ffffe0", "#ffeda0", "#feb24c", "#f03b20", "#4d0000"
    ]  # Light yellow to dark red

    # Format trading values for hover data
    aggregated_data['Trading'] = aggregated_data['Trading'].apply(lambda x: f"{x:,.0f}")

    fig = px.choropleth(
        aggregated_data,
        geojson=world.set_index('name')['geometry'].__geo_interface__,
        locations=aggregated_data['name'],
        color="Log_Trading",
        hover_name="name",
        hover_data={"Trading": True, "Log_Trading": False},
        color_continuous_scale=colors,
        range_color=(np.log1p(steps[0]), np.log1p(steps[-1])),
        labels={"Log_Trading": "Log(Trading)"},
        projection="natural earth"
    )

    # Create custom tickvals and ticktext for the color axis
    tick_vals = [np.log1p(x) for x in steps]
    tick_text = [f"{x:,}" for x in steps]

    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Total Trading (60kg bags)",
            tickvals=tick_vals,
            ticktext=tick_text,
            xanchor="left",  # Fix the color bar position
            x=0.9,  # Position the color bar inside the map
            y=0.5,
            yanchor="middle",
            lenmode="fraction",
            len=0.5  # Adjust the length of the color bar
        )
    )

    fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")

    # Add a scatter plot layer for the European Union
    fig.add_scattergeo(
        lon=[eu_data_dict['Longitude']],
        lat=[eu_data_dict['Latitude']],
        text=[f"Trading: {eu_data_dict['Trading']:,.0f}"],
        mode='markers',
        marker=dict(size=10, color='brown', symbol='circle'),
        hovertemplate=f"European Union<br>Trading: {eu_data_dict['Trading']:,.0f}<extra></extra>",
        name='European Union',
        customdata=[{'name': 'European Union'}]  # Add custom data to identify the EU dot
    )

    # Update the layout to include all traces in the selection
    fig.update_layout(
        clickmode='event+select'
    )

    return fig


def display_data(country: Optional[str], aggregated_data: pd.DataFrame):
    """Display the aggregated data for the selected country or the world if no country is selected."""
    if country:
        country_data = aggregated_data[aggregated_data['name'] == country]
        if country_data.empty:
            st.write(f"No data available for {country}")
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Country", country)
            col2.metric("Total Production", f"{country_data['Production'].values[0]:,.0f}")
            col3.metric("Total Exports", f"{country_data['Exports'].values[0]:,.0f}")
            col4.metric("Total Imports", f"{country_data['Imports'].values[0]:,.0f}")
            col5.metric("Total Domestic Consumption", f"{country_data['Domestic Consumption'].values[0]:,.0f}")
    else:
        world_data = aggregated_data[['Production', 'Exports', 'Imports', 'Domestic Consumption']].sum().to_frame().T
        world_data['Country'] = 'World'
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Country", "World")
        col2.metric("Total Production", f"{world_data['Production'].values[0]:,.0f}")
        col3.metric("Total Exports", f"{world_data['Exports'].values[0]:,.0f}")
        col4.metric("Total Imports", f"{world_data['Imports'].values[0]:,.0f}")
        col5.metric("Total Domestic Consumption", f"{world_data['Domestic Consumption'].values[0]:,.0f}")


def main():
    st.set_page_config(layout="wide")

    st.title("Global Coffee Trends: From Bean to Cup")
    st.markdown(
        """
        <div style="font-size: 24px;">
        Dive into the world of coffee with our interactive dashboard. Explore how production, consumption, and quality intertwine across the globe. Uncover surprising patterns in coffee preferences, from the impact of bean color on flavor to the shifting tides of imports and exports. Whether you're a casual sipper or a coffee connoisseur, this data-driven journey will give you a fresh perspective on your daily brew.
        </div>
        """, unsafe_allow_html=True
    )

    # Add CSS for dynamic width of the map container
    st.markdown(
        """
        <style>
        .map-container {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Display the image
    st.image("coffee_image_best_new_final.png", width=800)

    # Load data
    psd_coffee, arabica_clean, importers_consumption, re_export, domestic_consumption, coffee_import, world, population = load_data()

    # Create custom tabs
    tab_names = ["Trends", "Arabica Coffee Quality", "Compare and Explore"]
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = tab_names[0]

    create_custom_tabs(tab_names)

    # Display content based on active tab
    if st.session_state.active_tab == "Trends":
        st.header("Trends")

        year_range = st.slider("Select Year Range", min_value=1960, max_value=2023, value=(1960, 2023), key="year_range_slider")

        # Aggregate data for all countries based on the selected year range
        trading_data = psd_coffee[(psd_coffee['Year'] >= year_range[0]) & (psd_coffee['Year'] <= year_range[1])]
        aggregated_data = trading_data.groupby('Country').agg({
            'Production': 'sum',
            'Exports': 'sum',
            'Imports': 'sum',
            'Domestic Consumption': 'sum'
        }).reset_index()

        # Ensure all countries are included
        world['name'] = world['name'].str.strip()
        aggregated_data = world[['name']].merge(aggregated_data, how='left', left_on='name', right_on='Country').fillna(0)

        # Add European Union data to the aggregated dataset
        eu_data = psd_coffee[psd_coffee['Country'] == 'European Union']
        eu_aggregated = eu_data.groupby('Country').agg({
            'Production': 'sum',
            'Exports': 'sum',
            'Imports': 'sum',
            'Domestic Consumption': 'sum'
        }).reset_index()
        eu_aggregated['name'] = 'European Union'
        aggregated_data = pd.concat([aggregated_data, eu_aggregated], ignore_index=True)

        # Initialize the session state for selected country if not already done
        if 'selected_country' not in st.session_state:
            st.session_state.selected_country = None

        # Create a placeholder for the country data display
        country_data_placeholder = st.empty()

        # Function to update the displayed data
        def update_display():
            with country_data_placeholder:
                display_data(st.session_state.selected_country, aggregated_data)

        # Button to clear the country selection and show world data
        if st.button("Show World Data"):
            st.session_state.selected_country = None
            st.experimental_rerun()  # Full rerun when the button is clicked

        # Initial display of data
        update_display()

        # Add a tooltip to the map section
        subheader_with_tooltip(
            "Interactive Coffee Trading Map",
            "This interactive map displays coffee trading data for selected years. "
            "Trading is Exports + Imports of the country. Click on a country to view its details. "
            "To reset the map to its initial state, click again on the last selected country, or adjust the year range slider."
        )

        # Create and display the interactive map
        fig = create_interactive_trading_map(psd_coffee, world, year_range)
        selected_point = plotly_events(fig, click_event=True)

        if selected_point:
            try:
                if 'customdata' in selected_point[0]:
                    # EU dot was clicked
                    selected_country = selected_point[0]['customdata']['name']
                else:
                    # Country on the map was clicked
                    point_number = selected_point[0]["pointNumber"]
                    selected_country = world.iloc[point_number]["name"]

                if st.session_state.selected_country != selected_country:
                    st.session_state.selected_country = selected_country
                    update_display()  # Update the displayed data without rerunning the app
            except (KeyError, IndexError) as e:
                st.error(f"Error: {e}. Event data: {selected_point}")


        # Two columns for the remaining charts
        col1, col2 = st.columns(2)

        with col1:
            subheader_with_tooltip(
                "Top Coffee Producing Countries",
                "This chart displays the top coffee producing countries over time. It helps identify the major players in coffee production and how their production levels have changed over the years."
            )
            fig_producers = create_top_producers_chart(psd_coffee)
            fig_producers.update_layout(height=400)
            st.plotly_chart(fig_producers, use_container_width=True)

        with col2:
            subheader_with_tooltip(
                "Arabica vs Robusta Production Trends",
                "This chart compares the production trends of Arabica and Robusta coffee for selected countries. Arabica is generally considered higher quality but more difficult to grow, while Robusta is hardier but often considered less flavorful. The balance between these can indicate changes in global coffee preferences and growing conditions."
            )

            all_countries = sorted(psd_coffee['Country'].unique())
            default_countries = ['Vietnam', 'Indonesia']
            default_countries = [country for country in default_countries if country in all_countries]

            selected_countries = st.multiselect(
                "Select up to 3 countries to analyze",
                options=all_countries,
                default=default_countries,
                max_selections=3,
                key="countries_multiselect"
            )

            if selected_countries:
                fig = create_multiline_chart(psd_coffee, selected_countries)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.write("Please select at least one country to visualize production trends.")

        col3, col4 = st.columns(2)

        with col3:
            subheader_with_tooltip(
                "Coffee Import Trends by Type",
                "This line chart shows import trends for different types of coffee over time. It can reveal changes in global coffee preferences, trade patterns, and processing trends."
            )
            all_countries = st.checkbox("Select All Countries", value=True, key="all_countries_checkbox")
            all_country_options = sorted(psd_coffee['Country'].unique())
            default_countries = ['United States', 'Germany', 'Japan', 'Italy', 'France']
            default_countries = [country for country in default_countries if country in all_country_options]

            if not all_countries:
                selected_countries = st.multiselect(
                    "Select specific countries to analyze",
                    options=all_country_options,
                    default=default_countries,
                    key="countries_multiselect_import"
                )
            else:
                selected_countries = all_country_options

            if selected_countries:
                fig = create_import_trends_chart(psd_coffee, selected_countries, all_countries)
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            subheader_with_tooltip(
                "Global Coffee Consumption and Production Trends",
                "This line chart shows global coffee consumption and production trends over time. It helps visualize how consumption and production have changed and how they relate to each other. You may select specific countries by ticking off 'Select All Countries'"
            )

            # Add the checkbox and multiselect for the 4th graph
            all_countries_4 = st.checkbox("Select All Countries", value=True,
                                          key="all_countries_checkbox_4")
            all_country_options_4 = sorted(psd_coffee['Country'].unique())
            default_countries_4 = ['Brazil', 'Vietnam', 'Colombia', 'Indonesia', 'Ethiopia']
            default_countries_4 = [country for country in default_countries_4 if country in all_country_options_4]

            if not all_countries_4:
                selected_countries_4 = st.multiselect(
                    "Select specific countries to analyze for Production vs Consumption",
                    options=all_country_options_4,
                    default=default_countries_4,
                    key="countries_multiselect_production_consumption"
                )
            else:
                selected_countries_4 = all_country_options_4

            if selected_countries_4:
                fig_consumption_production_trend = create_consumption_production_trend_psd(psd_coffee,
                                                                                           selected_countries_4)
                st.plotly_chart(fig_consumption_production_trend, use_container_width=True)
    elif st.session_state.active_tab == "Arabica Coffee Quality":
        st.header("Arabica Coffee Quality")
        # Add content for Arabica Coffee Quality tab
        col1, col2 = st.columns(2)
        with col1:
            subheader_with_tooltip(
                "Altitude vs. Coffee Quality",
                "This scatter plot shows the relationship between altitude and coffee quality. Generally, higher altitudes are associated with better quality coffee due to slower growth and more concentrated flavors. Look for patterns or clusters in the data.  You may select specific countries by ticking off 'Select All Countries'"
            )
            all_countries = sorted(arabica_clean['Country of Origin'].unique())
            selected_countries = st.multiselect(
                "Select up to 5 countries to analyze",
                options=all_countries,
                default=all_countries[:3],
                max_selections=5,
                key="arabica_countries_multiselect"
            )
            if selected_countries:
                fig_altitude = create_altitude_quality_scatter(arabica_clean, selected_countries)
                fig_altitude.update_layout(height=400)
                st.plotly_chart(fig_altitude, use_container_width=True)

        with col2:
            subheader_with_tooltip(
                "Coffee Quality by Processing Method",
                "This box plot compares coffee quality scores across different processing methods. The processing method can significantly affect the final flavor profile of the coffee. Observe which methods tend to produce higher quality scores and which have more variability."
            )
            fig_processing = create_processing_quality_chart(arabica_clean)
            fig_processing.update_layout(height=400)
            st.plotly_chart(fig_processing, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            subheader_with_tooltip(
                "Coffee Quality by Bean Color",
                "This chart shows how coffee quality varies with bean color. Bean color can be an indicator of roast level and overall bean health. Look for any correlations between certain colors and higher quality scores."
            )
            fig_color = create_color_quality_chart(arabica_clean)
            fig_color.update_layout(height=400)
            st.plotly_chart(fig_color, use_container_width=True)

        with col4:
            subheader_with_tooltip(
                "Average Coffee Flavor Profile by Country",
                "This bar chart shows the average flavor profile of coffee from selected countries. Each bar represents a different flavor attribute, and the chart allows for easy comparison of flavor profiles between countries."
            )

            country1 = st.selectbox("Select first country", options=all_countries, index=0, key="country1")
            country2 = st.selectbox("Select second country", options=all_countries, index=1, key="country2")

            if country1 and country2 and country1 != country2:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig_flavor = create_flavor_profile_bar_plot(arabica_clean, country1, country2)
                    fig_flavor.update_layout(height=400)
                    st.plotly_chart(fig_flavor, use_container_width=True)

                with col2:
                    display_total_cup_points(arabica_clean, country1, country2)
            else:
                st.write("Please select two different countries to compare.")

    elif st.session_state.active_tab == "Compare and Explore":
        st.header("Compare and Explore")
        # Add content for Compare and Explore tab
        col1, col2 = st.columns(2)
        with col1:
            subheader_with_tooltip(
                "Re-exports vs Domestic Consumption for Major Coffee Trading Countries",
                "This bar chart shows the re-exports and domestic consumption for major coffee trading countries side by side. It helps to understand how much imported coffee is consumed domestically versus re-exported."
            )
            fig_reexport_vs_domestic = create_reexport_vs_domestic_chart(coffee_import, re_export,
                                                                         importers_consumption)
            st.plotly_chart(fig_reexport_vs_domestic, use_container_width=True)

        with col2:
            subheader_with_tooltip(
                "Top 10 Coffee Consuming Countries per Capita",
                "This chart displays coffee consumption per capita, showing which countries consume the most coffee relative to their population size. This can reveal interesting patterns about coffee culture in different countries."
            )
            fig_consumers_capita = create_top_consumers_per_capita_chart(psd_coffee, population)
            fig_consumers_capita.update_layout(height=400)
            st.plotly_chart(fig_consumers_capita, use_container_width=True)

        subheader_with_tooltip(
            "Predominant Coffee Type Consumption by Country",
            "This world map shows the predominant type of coffee consumed in each country. Countries are colored based on whether they primarily consume Arabica, Robusta, or a mix of both. Hover over countries to see total cunsmuption in the country from 1990-2019"
        )
        fig_coffee_type_map = create_coffee_type_map(domestic_consumption, world)
        st.plotly_chart(fig_coffee_type_map, use_container_width=True)

        subheader_with_tooltip(
            "Top 10 Countries: Production vs Domestic Consumption",
            "This bubble chart compares coffee production and domestic consumption for the top 10 coffee-consuming countries. The size of each bubble represents the country's coffee exports."
        )
        fig_consumption_production = create_consumption_production_bubble_top10(domestic_consumption,
                                                                                psd_coffee)
        st.plotly_chart(fig_consumption_production, use_container_width=True)

if __name__ == "__main__":
    main()

