import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime
import base64
import io
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("final_combined_dataset.csv")

# Normalize data types and clean state names
df['state'] = df['state'].str.strip()
if 'year' in df.columns:
    df['year'] = df['year'].astype(str)

# Load GeoJSON for Indian states
geojson_path = r"C:\Users\Shravani\Desktop\DWM_Project\Human_trafficking\india-maps-data-main\geojson\india.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    india_geojson = json.load(f)

# Ensure state mapping consistency
state_name_mapping = {
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'Delhi UT': 'Delhi',
    'DNH  and  Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'A & N Islands': 'Andaman and Nicobar Islands',
    'Unknown': None
}

# Clean and map state names
df['state_clean'] = df['state'].replace(state_name_mapping)
df = df[df['state_clean'].notna()]
df['state_mapped'] = df['state_clean']

# ---------------------------
# Helper Functions
# ---------------------------
def generate_insights(filtered_df):
    """Generate automated insights using simple NLP-style analysis"""
    insights = []
    
    if len(filtered_df) == 0:
        return ["No data available for the selected filters."]
    
    # Total statistics
    total_victims = len(filtered_df)
    female_pct = (len(filtered_df[filtered_df['gender'] == 'Female']) / total_victims * 100)
    minor_pct = (len(filtered_df[filtered_df['age_group'] == 'Below 18']) / total_victims * 100)
    high_risk_pct = (len(filtered_df[filtered_df['risk_label'] == 'High']) / total_victims * 100)
    
    insights.append(f"📊 Total victims in dataset: {total_victims:,}")
    insights.append(f"👩 Female victims comprise {female_pct:.1f}% of all cases")
    insights.append(f"🧒 Minors (below 18) account for {minor_pct:.1f}% of victims")
    insights.append(f"⚠️ High-risk cases represent {high_risk_pct:.1f}% of total victims")
    
    # Top state analysis
    top_state = filtered_df['state_clean'].value_counts().index[0]
    top_state_count = filtered_df['state_clean'].value_counts().iloc[0]
    state_pct = (top_state_count / total_victims * 100)
    insights.append(f"🏆 {top_state} has the highest number of cases ({top_state_count:,} victims, {state_pct:.1f}%)")
    
    # Gender-region analysis
    if 'region_type' in filtered_df.columns:
        urban_female = len(filtered_df[(filtered_df['region_type'] == 'Urban') & (filtered_df['gender'] == 'Female')])
        rural_female = len(filtered_df[(filtered_df['region_type'] == 'Rural') & (filtered_df['gender'] == 'Female')])
        if urban_female > rural_female:
            insights.append(f"🏙️ Urban areas show higher female trafficking ({urban_female} vs {rural_female} in rural areas)")
        elif rural_female > urban_female:
            insights.append(f"🌾 Rural areas show higher female trafficking ({rural_female} vs {urban_female} in urban areas)")
    
    # Year-based insights (if year data available)
    if 'year' in filtered_df.columns and len(filtered_df['year'].unique()) > 1:
        yearly_counts = filtered_df['year'].value_counts().sort_index()
        if len(yearly_counts) >= 2:
            latest_year = yearly_counts.index[-1]
            prev_year = yearly_counts.index[-2]
            change = yearly_counts[latest_year] - yearly_counts[prev_year]
            change_pct = (change / yearly_counts[prev_year] * 100)
            if change > 0:
                insights.append(f"📈 Cases increased by {change:,} ({change_pct:+.1f}%) from {prev_year} to {latest_year}")
            elif change < 0:
                insights.append(f"📉 Cases decreased by {abs(change):,} ({change_pct:.1f}%) from {prev_year} to {latest_year}")
    
    return insights

def chatbot_response(query):
    """Simple rule-based chatbot for data queries"""
    query = query.lower().strip()
    
    # Define response patterns
    if any(word in query for word in ['highest', 'most', 'maximum']):
        if 'state' in query:
            top_state = df['state_clean'].value_counts().index[0]
            count = df['state_clean'].value_counts().iloc[0]
            if 'minor' in query or 'child' in query:
                minors_by_state = df[df['age_group'] == 'Below 18']['state_clean'].value_counts()
                top_state = minors_by_state.index[0]
                count = minors_by_state.iloc[0]
                return f"🧒 {top_state} has the highest number of minor victims: {count:,} cases"
            elif 'female' in query or 'women' in query:
                females_by_state = df[df['gender'] == 'Female']['state_clean'].value_counts()
                top_state = females_by_state.index[0]
                count = females_by_state.iloc[0]
                return f"👩 {top_state} has the highest number of female victims: {count:,} cases"
            else:
                return f"🏆 {top_state} has the highest number of trafficking victims: {count:,} cases"
    
    elif any(word in query for word in ['total', 'how many', 'count']):
        if 'minor' in query or 'child' in query:
            count = len(df[df['age_group'] == 'Below 18'])
            return f"🧒 Total minor victims (below 18): {count:,}"
        elif 'female' in query or 'women' in query:
            count = len(df[df['gender'] == 'Female'])
            return f"👩 Total female victims: {count:,}"
        elif 'male' in query or 'men' in query:
            count = len(df[df['gender'] == 'Male'])
            return f"👨 Total male victims: {count:,}"
        else:
            return f"📊 Total trafficking victims in dataset: {len(df):,}"
    
    elif any(word in query for word in ['year', '2018', '2019', '2020']):
        if 'year' in df.columns:
            year_match = None
            for year in ['2018', '2019', '2020']:
                if year in query:
                    year_match = year
                    break
            if year_match:
                year_data = df[df['year'] == year_match]
                count = len(year_data)
                return f"📅 In {year_match}, there were {count:,} trafficking victims recorded"
            else:
                yearly_counts = df['year'].value_counts().sort_index()
                return f"📅 Year-wise data: " + ", ".join([f"{year}: {count:,}" for year, count in yearly_counts.items()])
        else:
            return "⚠️ Year information is not available in the current dataset"
    
    elif any(word in query for word in ['risk', 'high risk', 'dangerous']):
        high_risk = len(df[df['risk_label'] == 'High'])
        low_risk = len(df[df['risk_label'] == 'Low'])
        high_risk_pct = (high_risk / len(df) * 100)
        return f"⚠️ High-risk cases: {high_risk:,} ({high_risk_pct:.1f}%), Low-risk cases: {low_risk:,}"
    
    elif any(word in query for word in ['help', 'what can', 'options']):
        return """🤖 I can help you with questions like:
        • "Which state had the highest number of victims?"
        • "How many minor victims were there?"
        • "What's the total count of female victims?"
        • "Show me high-risk cases"
        • "What happened in 2019?"
        """
    
    else:
        return "🤔 I didn't understand that. Try asking about states, victims, years, or type 'help' for options."

# ---------------------------
# Initialize app
# ---------------------------
app = dash.Dash(__name__, external_stylesheets=[
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    "/assets/style.css"
])
app.title = "Enhanced Human Trafficking Analysis Dashboard"

# ---------------------------
# Layout
# ---------------------------
app.layout = html.Div([
    # Modern Navbar
    html.Nav([
        html.Div([
            html.I(className="fas fa-chart-line", style={'marginRight': '10px'}),
            "Enhanced Human Trafficking Analytics"
        ], className="navbar-brand", style={
            'fontSize': '1.5rem', 'fontWeight': '600', 'color': '#00bcf2'
        }),
        html.Ul([
            html.Li(html.A([html.I(className="fas fa-home", style={'marginRight': '8px'}), "Dashboard"], 
                          href="#top", className="active")),
            html.Li(html.A([html.I(className="fas fa-map", style={'marginRight': '8px'}), "Geographic View"], 
                          href="#map-section")),
            html.Li(html.A([html.I(className="fas fa-balance-scale", style={'marginRight': '8px'}), "Compare"], 
                          href="#compare-section")),
            html.Li(html.A([html.I(className="fas fa-robot", style={'marginRight': '8px'}), "AI Insights"], 
                          href="#ai-section")),
            html.Li(html.A([html.I(className="fas fa-download", style={'marginRight': '8px'}), "Reports"], 
                          href="#reports-section"))
        ], className="navbar-nav")
    ], className="navbar"),

    html.Div(id="top"),
    
    # Page Header
    html.Div([
        html.H1([html.I(className="fas fa-brain", style={'marginRight': '15px'}), 
                 "Enhanced Human Trafficking Analysis Dashboard"]),
        html.P("Comprehensive analytics with AI insights, geospatial analysis, and comparative features")
    ], className="page-header"),

    # Enhanced KPI Cards with YoY changes
    html.Div([
        html.Div([
            html.Div(html.I(className="fas fa-users"), className="icon metric-icon"),
            html.H2(id='total-records', className="metric"),
            html.Div(id='total-change', className="metric-change"),
            html.Div("Total Victims Recorded", className="metric-label")
        ], className='kpi-card'),
        html.Div([
            html.Div(html.I(className="fas fa-female"), className="icon metric-icon"),
            html.H2(id='female-count', className="metric"),
            html.Div(id='female-change', className="metric-change"),
            html.Div("Female Victims", className="metric-label")
        ], className='kpi-card'),
        html.Div([
            html.Div(html.I(className="fas fa-child"), className="icon metric-icon"),
            html.H2(id='minor-count', className="metric"),
            html.Div(id='minor-change', className="metric-change"),
            html.Div("Minors (Under 18)", className="metric-label")
        ], className='kpi-card'),
        html.Div([
            html.Div(html.I(className="fas fa-exclamation-triangle"), className="icon metric-icon"),
            html.H2(id='high-risk-count', className="metric"),
            html.Div(id='high-risk-change', className="metric-change"),
            html.Div("High-Risk Cases", className="metric-label")
        ], className='kpi-card'),
    ], className="metrics"),

    # Enhanced Filter Section
    html.Div([
        html.Div([
            html.H3([html.I(className="fas fa-filter", style={'marginRight': '10px'}), "Advanced Filters"])
        ], className="filter-header"),
        html.Div([
            html.Div([
                html.Label([html.I(className="fas fa-map-marker-alt", style={'marginRight': '8px'}), "Select States/UTs"], className="filter-label"),
                dcc.Dropdown(
                    id='state-filter', 
                    options=[{'label': s, 'value': s} for s in sorted(df['state_clean'].unique()) if s is not None],
                    multi=True, 
                    placeholder="Select State(s)",
                    className="form-select",
                    style={'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white'}
                )
            ], className="filter-group"),
            html.Div([
                html.Label([html.I(className="fas fa-venus-mars", style={'marginRight': '8px'}), "Select Gender"], className="filter-label"),
                dcc.Dropdown(
                    id='gender-filter', 
                    options=[{'label': g, 'value': g} for g in sorted(df['gender'].unique())],
                    multi=True, 
                    placeholder="Select Gender(s)",
                    className="form-select",
                    style={'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white'}
                )
            ], className="filter-group"),
            html.Div([
                html.Label([html.I(className="fas fa-birthday-cake", style={'marginRight': '8px'}), "Select Age Groups"], className="filter-label"),
                dcc.Dropdown(
                    id='age-filter', 
                    options=[{'label': a, 'value': a} for a in sorted(df['age_group'].unique())],
                    multi=True, 
                    placeholder="Select Age Group(s)",
                    className="form-select",
                    style={'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white'}
                )
            ], className="filter-group"),
            html.Div([
                html.Label([html.I(className="fas fa-calendar", style={'marginRight': '8px'}), "Select Year"], className="filter-label"),
                dcc.Dropdown(
                    id='year-filter', 
                    options=[{'label': y, 'value': y} for y in sorted(df['year'].unique()) if y != 'Unknown'],
                    multi=True, 
                    placeholder="Select Year(s)",
                    className="form-select",
                    style={'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white'}
                )
            ], className="filter-group")
        ], className="filter-grid")
    ], className="filter-section"),

    # AI-Driven Insights Section
    html.Div(id="ai-section"),
    html.Div([
        html.Div([
            html.H3([html.I(className="fas fa-lightbulb", style={'marginRight': '10px'}), "AI-Generated Insights"], className="section-title")
        ], className="section-header"),
        html.Div(id="ai-insights-content", className="insights-container")
    ], className="insights-section"),

    # Chatbot Section
    html.Div([
        html.Div([
            html.H3([html.I(className="fas fa-robot", style={'marginRight': '10px'}), "Ask the Data Assistant"], className="section-title")
        ], className="section-header"),
        html.Div([
            html.Div([
                dcc.Input(
                    id="chatbot-input",
                    type="text",
                    placeholder="Ask me about the data... e.g., 'Which state has the most victims?'",
                    style={'width': '70%', 'padding': '10px', 'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white', 'borderRadius': '5px'},
                ),
                html.Button("Ask", id="chatbot-submit", n_clicks=0, 
                           style={'marginLeft': '10px', 'padding': '10px 20px', 'backgroundColor': '#00bcf2', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
            html.Div(id="chatbot-response", className="chatbot-response")
        ], className="chatbot-container")
    ], className="chatbot-section"),

    # Comparison Section
    html.Div(id="compare-section"),
    html.Div([
        html.Div([
            html.H3([html.I(className="fas fa-balance-scale", style={'marginRight': '10px'}), "Comparative Analytics"], className="section-title")
        ], className="section-header"),
        html.Div([
            html.Div([
                html.Label("Compare States:", className="filter-label"),
                dcc.Dropdown(
                    id='compare-state1',
                    options=[{'label': s, 'value': s} for s in sorted(df['state_clean'].unique()) if s is not None],
                    placeholder="Select first state",
                    style={'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white'}
                ),
                dcc.Dropdown(
                    id='compare-state2',
                    options=[{'label': s, 'value': s} for s in sorted(df['state_clean'].unique()) if s is not None],
                    placeholder="Select second state",
                    style={'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white', 'marginTop': '10px'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                html.Label("Compare Years:", className="filter-label"),
                dcc.Dropdown(
                    id='compare-year1',
                    options=[{'label': y, 'value': y} for y in sorted(df['year'].unique()) if y != 'Unknown'],
                    placeholder="Select first year",
                    style={'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white'}
                ),
                dcc.Dropdown(
                    id='compare-year2',
                    options=[{'label': y, 'value': y} for y in sorted(df['year'].unique()) if y != 'Unknown'],
                    placeholder="Select second year",
                    style={'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white', 'marginTop': '10px'}
                )
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ]),
        html.Div(id="comparison-charts", style={'marginTop': '20px'})
    ], className="comparison-section"),

    # Reports Section
    html.Div(id="reports-section"),
    html.Div([
        html.Div([
            html.H3([html.I(className="fas fa-file-download", style={'marginRight': '10px'}), "Download Reports"], className="section-title")
        ], className="section-header"),
        html.Div([
            html.Button([html.I(className="fas fa-file-csv", style={'marginRight': '8px'}), "Download CSV Report"], 
                       id="download-csv", n_clicks=0, className="download-btn"),
            html.Button([html.I(className="fas fa-chart-bar", style={'marginRight': '8px'}), "Download Summary Report"], 
                       id="download-summary", n_clicks=0, className="download-btn"),
            dcc.Download(id="download-dataframe-csv"),
            dcc.Download(id="download-summary-report")
        ], style={'display': 'flex', 'gap': '15px'})
    ], className="reports-section"),

    # Enhanced Charts Section
    html.Div([
        html.Div([
            html.Div(html.H3([html.I(className="fas fa-chart-bar", style={'marginRight': '10px'}), "Top 10 States by Victims"], className="chart-title"), className="chart-header"),
            dcc.Graph(id='top-states-chart', style={'height': '400px'}, className="chart-wrapper")
        ], className="chart-container"),
        html.Div([
            html.Div(html.H3([html.I(className="fas fa-chart-column", style={'marginRight': '10px'}), "Gender Distribution by Region"], className="chart-title"), className="chart-header"),
            dcc.Graph(id='gender-region-chart', style={'height': '400px'}, className="chart-wrapper")
        ], className="chart-container"),
        html.Div([
            html.Div(html.H3([html.I(className="fas fa-chart-area", style={'marginRight': '10px'}), "Age Group vs Risk Level"], className="chart-title"), className="chart-header"),
            dcc.Graph(id='age-risk-chart', style={'height': '400px'}, className="chart-wrapper")
        ], className="chart-container"),
        html.Div([
            html.Div(html.H3([html.I(className="fas fa-chart-pie", style={'marginRight': '10px'}), "Overall Risk Distribution"], className="chart-title"), className="chart-header"),
            dcc.Graph(id='risk-distribution-chart', style={'height': '400px'}, className="chart-wrapper")
        ], className="chart-container"),
        html.Div([
            html.Div(html.H3([html.I(className="fas fa-chart-line", style={'marginRight': '10px'}), "Year-over-Year Trends"], className="chart-title"), className="chart-header"),
            dcc.Graph(id='yearly-trends-chart', style={'height': '400px'}, className="chart-wrapper")
        ], className="chart-container"),
        html.Div([
            html.Div(html.H3([html.I(className="fas fa-th", style={'marginRight': '10px'}), "Correlation Heatmap"], className="chart-title"), className="chart-header"),
            dcc.Graph(id='correlation-heatmap', style={'height': '400px'}, className="chart-wrapper")
        ], className="chart-container"),
    ], className="chart-grid"),

    # Enhanced Map Section
    html.Div(id="map-section"),
    html.Div([
        html.Div([
            html.H3([html.I(className="fas fa-map-marked-alt", style={'marginRight': '10px'}), "Enhanced Geographic Analysis"], className="map-title")
        ], className="map-header"),
        html.Div([
            html.Div([
                html.Label("Map Display:", className="filter-label"),
                dcc.Dropdown(
                    id='map-metric-selector',
                    options=[
                        {'label': 'Total Victims', 'value': 'total'},
                        {'label': 'Female Victims', 'value': 'female'},
                        {'label': 'Minor Victims', 'value': 'minor'},
                        {'label': 'High-Risk Cases', 'value': 'high_risk'}
                    ],
                    value='total',
                    style={'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white', 'marginBottom': '10px'}
                )
            ], style={'width': '25%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='enhanced-india-map', style={'height': '600px'})
            ], style={'width': '70%', 'float': 'right'})
        ], className="map-wrapper")
    ], className="map-container"),

    # Clustering Analysis
    html.Div([
        html.Div([
            html.H3([html.I(className="fas fa-project-diagram", style={'marginRight': '10px'}), "Regional Clustering Analysis"], className="section-title")
        ], className="section-header"),
        html.Div([
            html.Div([
                html.Label("View:", className="filter-label", style={'marginRight': '8px'}),
                dcc.Dropdown(
                    id='cluster-view',
                    options=[
                        {'label': 'Minor% vs Risk%', 'value': 'minor_risk'},
                        {'label': 'Minor% vs Female%', 'value': 'minor_female'},
                        {'label': 'Female% vs Risk%', 'value': 'female_risk'},
                        {'label': 'PCA (2D)', 'value': 'pca_2d'},
                    ],
                    value='minor_risk',
                    clearable=False,
                    style={'minWidth': '240px', 'backgroundColor': '#2d2d2d', 'border': '1px solid #484848', 'color': 'white'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
            dcc.Graph(id='clustering-analysis', style={'height': '500px'})
        ], className="chart-wrapper")
    ], className="clustering-section")

], style={
    'backgroundColor': '#1e1e1e', 
    'fontFamily': 'Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif',
    'color': '#ffffff',
    'minHeight': '100vh'
})

# ---------------------------
# Callbacks
# ---------------------------

# Main metrics callback with YoY changes
@app.callback(
    [Output('total-records', 'children'),
     Output('female-count', 'children'),
     Output('minor-count', 'children'),
     Output('high-risk-count', 'children'),
     Output('total-change', 'children'),
     Output('female-change', 'children'),
     Output('minor-change', 'children'),
     Output('high-risk-change', 'children')],
    [Input('state-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('age-filter', 'value'),
     Input('year-filter', 'value')]
)
def update_metrics(states, genders, ages, years):
    # Filter data
    filtered_df = df.copy()
    
    if states:
        filtered_df = filtered_df[filtered_df['state_clean'].isin(states)]
    if genders:
        filtered_df = filtered_df[filtered_df['gender'].isin(genders)]
    if ages:
        filtered_df = filtered_df[filtered_df['age_group'].isin(ages)]
    if years:
        years = [str(y) for y in years]
        filtered_df = filtered_df[filtered_df['year'].isin(years)]
    
    # Current metrics
    total = len(filtered_df)
    female = len(filtered_df[filtered_df['gender'] == 'Female'])
    minor = len(filtered_df[filtered_df['age_group'] == 'Below 18'])
    high_risk = len(filtered_df[filtered_df['risk_label'] == 'High'])
    
    # Calculate year-over-year changes if possible
    def calc_yoy_change(metric_filter=None):
        if 'year' not in filtered_df.columns or len(filtered_df['year'].unique()) < 2:
            return ""
        
        years_available = sorted([y for y in filtered_df['year'].unique() if y != 'Unknown'])
        if len(years_available) < 2:
            return ""
        
        current_year = years_available[-1]
        prev_year = years_available[-2]
        
        current_data = filtered_df[filtered_df['year'] == current_year]
        prev_data = filtered_df[filtered_df['year'] == prev_year]
        
        if metric_filter:
            current_count = len(current_data[current_data[metric_filter[0]] == metric_filter[1]])
            prev_count = len(prev_data[prev_data[metric_filter[0]] == metric_filter[1]])
        else:
            current_count = len(current_data)
            prev_count = len(prev_data)
        
        if prev_count == 0:
            return ""
        
        change = current_count - prev_count
        change_pct = (change / prev_count) * 100
        
        if change > 0:
            return f"↗️ +{change_pct:.1f}% vs {prev_year}"
        elif change < 0:
            return f"↘️ {change_pct:.1f}% vs {prev_year}"
        else:
            return f"→ No change vs {prev_year}"
    
    total_change = calc_yoy_change()
    female_change = calc_yoy_change(('gender', 'Female'))
    minor_change = calc_yoy_change(('age_group', 'Below 18'))
    high_risk_change = calc_yoy_change(('risk_label', 'High'))
    
    return f"{total:,}", f"{female:,}", f"{minor:,}", f"{high_risk:,}", total_change, female_change, minor_change, high_risk_change

# AI Insights callback
@app.callback(
    Output('ai-insights-content', 'children'),
    [Input('state-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('age-filter', 'value'),
     Input('year-filter', 'value')]
)
def update_insights(states, genders, ages, years):
    # Filter data
    filtered_df = df.copy()
    
    if states:
        filtered_df = filtered_df[filtered_df['state_clean'].isin(states)]
    if genders:
        filtered_df = filtered_df[filtered_df['gender'].isin(genders)]
    if ages:
        filtered_df = filtered_df[filtered_df['age_group'].isin(ages)]
    if years:
        filtered_df = filtered_df[filtered_df['year'].isin(years)]
    
    insights = generate_insights(filtered_df)
    
    insight_elements = []
    for insight in insights:
        insight_elements.append(html.Div(insight, className="insight-item"))
    
    return insight_elements

# Chatbot callback
@app.callback(
    Output('chatbot-response', 'children'),
    [Input('chatbot-submit', 'n_clicks')],
    [State('chatbot-input', 'value')]
)
def update_chatbot(n_clicks, query):
    if n_clicks == 0 or not query:
        return "👋 Hello! I'm your data assistant. Ask me anything about the human trafficking data!"
    
    response = chatbot_response(query)
    return response

# Charts callbacks (keeping existing functionality)
@app.callback(
    [Output('top-states-chart', 'figure'),
     Output('gender-region-chart', 'figure'),
     Output('age-risk-chart', 'figure'),
     Output('risk-distribution-chart', 'figure'),
     Output('yearly-trends-chart', 'figure'),
     Output('correlation-heatmap', 'figure')],
    [Input('state-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('age-filter', 'value'),
     Input('year-filter', 'value')]
)
def update_charts(states, genders, ages, years):
    # Filter data
    filtered_df = df.copy()
    
    if states:
        filtered_df = filtered_df[filtered_df['state_clean'].isin(states)]
    if genders:
        filtered_df = filtered_df[filtered_df['gender'].isin(genders)]
    if ages:
        filtered_df = filtered_df[filtered_df['age_group'].isin(ages)]
    if years:
        # Ensure string comparison for years
        years = [str(y) for y in years]
        filtered_df = filtered_df[filtered_df['year'].isin(years)]
    
    # If no data matches filters, return placeholder figures to avoid callback errors
    if filtered_df.empty:
        placeholder = px.bar(title="No data available for selected filters")
        placeholder.update_layout(
            plot_bgcolor='rgba(36, 36, 36, 1)',
            paper_bgcolor='rgba(36, 36, 36, 1)',
            font=dict(color='white')
        )
        return placeholder, placeholder, placeholder, placeholder, placeholder, placeholder
    
    # Chart styling
    chart_style = {
        'plot_bgcolor': 'rgba(36, 36, 36, 1)',
        'paper_bgcolor': 'rgba(36, 36, 36, 1)',
        'font': dict(color='white', size=12),
        'title_font': dict(size=16, color='white'),
        'xaxis': dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
        'yaxis': dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
        'margin': dict(l=60, r=60, t=60, b=60)
    }
    
    # Top 10 States
    top_states = filtered_df['state_clean'].value_counts().head(10)
    fig1 = px.bar(
        top_states, x=top_states.index, y=top_states.values,
        color=top_states.values, color_continuous_scale='Viridis',
        title='Top 10 States by Victims'
    )
    fig1.update_layout(**chart_style)
    
    # Gender by Region
    if 'region_type' in filtered_df.columns:
        gender_region = filtered_df.groupby(['gender','region_type']).size().reset_index(name='count')
        fig2 = px.bar(
            gender_region, x='gender', y='count', color='region_type', barmode='group',
            title="Victims by Gender & Region Type"
        )
    else:
        fig2 = px.bar(title="Region Type data not available")
    fig2.update_layout(**chart_style)
    
    # Age vs Risk
    fig3 = px.bar(
        filtered_df, x='age_group', color='risk_label', barmode='stack',
        color_discrete_map={'Low':'#00b294','High':'#e81123'},
        title="Victim Age Group vs Risk Level"
    )
    fig3.update_layout(**chart_style)
    
    # Risk Distribution
    risk_counts = filtered_df['risk_label'].value_counts()
    fig4 = px.pie(
        values=risk_counts.values, names=risk_counts.index,
        color_discrete_map={'Low':'#00b294','High':'#e81123'},
        title="Overall Risk Distribution", hole=0.4
    )
    fig4.update_layout(**chart_style)
    
    # Yearly trends
    if 'year' in filtered_df.columns:
        yearly_data = filtered_df[filtered_df['year'] != 'Unknown']
        yearly_counts = yearly_data.groupby(['year', 'gender']).size().reset_index(name='count')
        fig5 = px.line(
            yearly_counts, x='year', y='count', color='gender',
            title="Year-over-Year Trends by Gender"
        )
    else:
        fig5 = px.line(title="Year data not available")
    fig5.update_layout(**chart_style)
    
    # Correlation heatmap
    numeric_data = pd.get_dummies(filtered_df[['gender', 'age_group', 'risk_label']])
    correlation_matrix = numeric_data.corr()
    fig6 = px.imshow(
        correlation_matrix, text_auto=True, aspect="auto",
        title="Feature Correlation Heatmap"
    )
    fig6.update_layout(**chart_style)
    
    return fig1, fig2, fig3, fig4, fig5, fig6

# Enhanced map callback
@app.callback(
    Output('enhanced-india-map', 'figure'),
    [Input('state-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('age-filter', 'value'),
     Input('year-filter', 'value'),
     Input('map-metric-selector', 'value')]
)
def update_enhanced_map(states, genders, ages, years, metric):
    # Filter data
    filtered_df = df.copy()
    
    if states:
        filtered_df = filtered_df[filtered_df['state_clean'].isin(states)]
    if genders:
        filtered_df = filtered_df[filtered_df['gender'].isin(genders)]
    if ages:
        filtered_df = filtered_df[filtered_df['age_group'].isin(ages)]
    if years:
        years = [str(y) for y in years]
        filtered_df = filtered_df[filtered_df['year'].isin(years)]
    
    # Calculate state-wise metrics
    state_stats = filtered_df.groupby('state_mapped').agg({
        'gender': 'count',  # total
        'age_group': lambda x: (x == 'Below 18').sum(),  # minors
        'risk_label': lambda x: (x == 'High').sum()  # high_risk
    }).rename(columns={'gender': 'total', 'age_group': 'minor', 'risk_label': 'high_risk'})
    
    # Calculate female count
    female_counts = filtered_df[filtered_df['gender'] == 'Female'].groupby('state_mapped').size()
    state_stats['female'] = female_counts.fillna(0).astype(int)
    
    state_stats = state_stats.reset_index()
    
    # Select metric for coloring
    color_column = metric
    if metric == 'total':
        color_values = state_stats['total']
        title = "Total Trafficking Victims by State"
    elif metric == 'female':
        color_values = state_stats['female']
        title = "Female Trafficking Victims by State"
    elif metric == 'minor':
        color_values = state_stats['minor']
        title = "Minor Trafficking Victims by State"
    elif metric == 'high_risk':
        color_values = state_stats['high_risk']
        title = "High-Risk Trafficking Cases by State"
    
    # Create choropleth map
    fig = px.choropleth(
        state_stats,
        geojson=india_geojson,
        locations='state_mapped',
        color=color_values,
        hover_name='state_mapped',
        hover_data={'total': True, 'female': True, 'minor': True, 'high_risk': True},
        color_continuous_scale="Viridis",
        featureidkey="properties.NAME_1",
        title=title
    )
    
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        plot_bgcolor='rgba(36, 36, 36, 1)',
        paper_bgcolor='rgba(36, 36, 36, 1)',
        font=dict(color='white', size=12),
        title_font=dict(size=16, color='white'),
        geo=dict(bgcolor='rgba(36, 36, 36, 1)')
    )
    
    return fig

# Comparison callback
@app.callback(
    Output('comparison-charts', 'children'),
    [Input('compare-state1', 'value'),
     Input('compare-state2', 'value'),
     Input('compare-year1', 'value'),
     Input('compare-year2', 'value')]
)
def update_comparison(state1, state2, year1, year2):
    if not any([state1, state2, year1, year2]):
        return html.Div("Select states or years to compare", className="comparison-placeholder")
    
    comparison_elements = []
    
    # State comparison
    if state1 and state2:
        state1_data = df[df['state_clean'] == state1]
        state2_data = df[df['state_clean'] == state2]
        
        # Comparison metrics
        metrics = {
            'Total Victims': [len(state1_data), len(state2_data)],
            'Female Victims': [len(state1_data[state1_data['gender'] == 'Female']), 
                             len(state2_data[state2_data['gender'] == 'Female'])],
            'Minor Victims': [len(state1_data[state1_data['age_group'] == 'Below 18']), 
                            len(state2_data[state2_data['age_group'] == 'Below 18'])],
            'High-Risk Cases': [len(state1_data[state1_data['risk_label'] == 'High']), 
                              len(state2_data[state2_data['risk_label'] == 'High'])]
        }
        
        # Create comparison chart
        comparison_df = pd.DataFrame({
            state1: [metrics[m][0] for m in metrics],
            state2: [metrics[m][1] for m in metrics]
        }, index=list(metrics.keys()))
        
        fig = px.bar(comparison_df, barmode='group', title=f"State Comparison: {state1} vs {state2}")
        fig.update_layout(
            plot_bgcolor='rgba(36, 36, 36, 1)',
            paper_bgcolor='rgba(36, 36, 36, 1)',
            font=dict(color='white', size=12),
            title_font=dict(size=16, color='white'),
            xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white')
        )
        
        comparison_elements.append(dcc.Graph(figure=fig, style={'height': '400px'}))
    
    # Year comparison
    if year1 and year2:
        year1_data = df[df['year'] == year1]
        year2_data = df[df['year'] == year2]
        
        metrics = {
            'Total Victims': [len(year1_data), len(year2_data)],
            'Female Victims': [len(year1_data[year1_data['gender'] == 'Female']), 
                             len(year2_data[year2_data['gender'] == 'Female'])],
            'Minor Victims': [len(year1_data[year1_data['age_group'] == 'Below 18']), 
                            len(year2_data[year2_data['age_group'] == 'Below 18'])],
            'High-Risk Cases': [len(year1_data[year1_data['risk_label'] == 'High']), 
                              len(year2_data[year2_data['risk_label'] == 'High'])]
        }
        
        comparison_df = pd.DataFrame({
            year1: [metrics[m][0] for m in metrics],
            year2: [metrics[m][1] for m in metrics]
        }, index=list(metrics.keys()))
        
        fig = px.bar(comparison_df, barmode='group', title=f"Year Comparison: {year1} vs {year2}")
        fig.update_layout(
            plot_bgcolor='rgba(36, 36, 36, 1)',
            paper_bgcolor='rgba(36, 36, 36, 1)',
            font=dict(color='white', size=12),
            title_font=dict(size=16, color='white'),
            xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white')
        )
        
        comparison_elements.append(dcc.Graph(figure=fig, style={'height': '400px', 'marginTop': '20px'}))
    
    return comparison_elements

# Clustering analysis callback
@app.callback(
    Output('clustering-analysis', 'figure'),
    [Input('state-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('age-filter', 'value'),
     Input('year-filter', 'value'),
     Input('cluster-view', 'value')]
)
def update_clustering(states, genders, ages, years, cluster_view):
    # Filter data
    filtered_df = df.copy()
    
    if states:
        filtered_df = filtered_df[filtered_df['state_clean'].isin(states)]
    if genders:
        filtered_df = filtered_df[filtered_df['gender'].isin(genders)]
    if ages:
        filtered_df = filtered_df[filtered_df['age_group'].isin(ages)]
    if years:
        years = [str(y) for y in years]
        filtered_df = filtered_df[filtered_df['year'].isin(years)]
    
    if filtered_df.empty:
        fig = px.scatter(title="No data available for clustering")
        fig.update_layout(
            plot_bgcolor='rgba(36, 36, 36, 1)',
            paper_bgcolor='rgba(36, 36, 36, 1)',
            font=dict(color='white')
        )
        return fig
    
    # Create state-level features for clustering
    state_features = filtered_df.groupby('state_mapped').agg({
        'gender': 'count',  # total victims
        'age_group': lambda x: (x == 'Below 18').sum() / len(x),  # minor percentage
        'risk_label': lambda x: (x == 'High').sum() / len(x)  # risk percentage
    }).rename(columns={'gender': 'total_victims', 'age_group': 'minor_pct', 'risk_label': 'risk_pct'})
    
    # Add female percentage
    female_pct = filtered_df[filtered_df['gender'] == 'Female'].groupby('state_mapped').size() / filtered_df.groupby('state_mapped').size()
    state_features['female_pct'] = female_pct.fillna(0)
    
    # Normalize features for clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_mat = state_features[['minor_pct', 'female_pct', 'risk_pct']]
    features_scaled = scaler.fit_transform(features_mat)

    # Apply DBSCAN clustering
    try:
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(features_scaled)
        state_features['cluster'] = clustering.labels_
    except Exception:
        state_features['cluster'] = -1

    # Prepare plotting dataframe and labels
    df_plot = state_features.reset_index()
    cluster_labels = df_plot['cluster'].replace({-1: 'Noise'}).astype(str)

    # Choose 2D view
    if not cluster_view:
        cluster_view = 'minor_risk'

    title = "Regional Clustering"
    hover_data = {'total_victims': True, 'minor_pct': ':.1%', 'female_pct': ':.1%', 'risk_pct': ':.1%'}

    if cluster_view == 'pca_2d':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        comps = pca.fit_transform(features_scaled)
        df_plot['PC1'] = comps[:, 0]
        df_plot['PC2'] = comps[:, 1]
        x_col, y_col = 'PC1', 'PC2'
        title = "Regional Clustering (PCA 2D)"
    elif cluster_view == 'minor_female':
        x_col, y_col = 'minor_pct', 'female_pct'
        title = "Minor% vs Female% by State"
    elif cluster_view == 'female_risk':
        x_col, y_col = 'female_pct', 'risk_pct'
        title = "Female% vs Risk% by State"
    else:  # 'minor_risk'
        x_col, y_col = 'minor_pct', 'risk_pct'
        title = "Minor% vs Risk% by State"

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color=cluster_labels,
        size='total_victims',
        hover_name='state_mapped',
        hover_data=hover_data,
        title=f"{title} (DBSCAN clusters)"
    )

    fig.update_layout(
        plot_bgcolor='rgba(36, 36, 36, 1)',
        paper_bgcolor='rgba(36, 36, 36, 1)',
        font=dict(color='white', size=12),
        title_font=dict(size=16, color='white'),
        xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
        yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
        legend_title_text='Cluster'
    )

    return fig

# Download callbacks
@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-csv", "n_clicks")],
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    return dcc.send_data_frame(df.to_csv, "trafficking_data.csv")

@app.callback(
    Output("download-summary-report", "data"),
    [Input("download-summary", "n_clicks")],
    prevent_initial_call=True,
)
def download_summary(n_clicks):
    # Create summary report
    summary_stats = {
        'Metric': ['Total Victims', 'Female Victims', 'Minor Victims', 'High-Risk Cases', 
                  'Top State', 'Most Affected Year'],
        'Value': [
            len(df),
            len(df[df['gender'] == 'Female']),
            len(df[df['age_group'] == 'Below 18']),
            len(df[df['risk_label'] == 'High']),
            df['state_clean'].value_counts().index[0],
            df['year'].value_counts().index[0] if 'year' in df.columns else 'N/A'
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    return dcc.send_data_frame(summary_df.to_csv, "trafficking_summary.csv")

if __name__ == '__main__':
    app.run(debug=True, port=8051)
