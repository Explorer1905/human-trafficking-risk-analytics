import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("final_combined_dataset.csv")

# Clean state names by stripping whitespace
df['state'] = df['state'].str.strip()

# Load GeoJSON for Indian states
geojson_path = r"C:\Users\Shravani\Desktop\DWM_Project\Human_trafficking\india-maps-data-main\geojson\india.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    india_geojson = json.load(f)

# Ensure state mapping consistency
# Create mapping for states that have different names in CSV vs GeoJSON
state_name_mapping = {
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'Delhi UT': 'Delhi',
    'DNH  and  Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'A & N Islands': 'Andaman and Nicobar Islands',
    'Unknown': None  # Exclude unknown states
}

# Clean and map state names
df['state_clean'] = df['state'].replace(state_name_mapping)
df = df[df['state_clean'].notna()]  # Remove rows with unknown states
df['state_mapped'] = df['state_clean']

# ---------------------------
# Initialize app
# ---------------------------
app = dash.Dash(__name__, external_stylesheets=[
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    "/assets/style.css"  # Will use our custom CSS
])
app.title = "Human Trafficking Analysis Dashboard"

# ---------------------------
# Layout
# ---------------------------
app.layout = html.Div([
    # Modern Navbar - Power BI Style
    html.Nav([
        html.Div([
            html.I(className="fas fa-chart-line", style={'marginRight': '10px'}),
            "Human Trafficking Analytics"
        ], className="navbar-brand", style={
            'fontSize': '1.5rem', 'fontWeight': '600', 'color': '#00bcf2'
        }),
        html.Ul([
            html.Li(html.A([html.I(className="fas fa-home", style={'marginRight': '8px'}), "Dashboard"], 
                          href="#top", className="active")),
            html.Li(html.A([html.I(className="fas fa-map", style={'marginRight': '8px'}), "Geographic View"], 
                          href="#map-section"))
        ], className="navbar-nav")
    ], className="navbar"),

    html.Div(id="top"),
    
    # Page Header - Power BI Style
    html.Div([
        html.H1([html.I(className="fas fa-brain", style={'marginRight': '15px'}), 
                 "Human Trafficking Analysis Dashboard"]),
        html.P("Comprehensive analytics and insights into human trafficking data patterns")
    ], className="page-header"),

    # Modern KPI Cards
    html.Div([
        html.Div([
            html.Div(html.I(className="fas fa-users"), className="icon metric-icon"),
            html.H2(id='total-records', className="metric"),
            html.Div("Total Victims Recorded", className="metric-label")
        ], className='kpi-card'),
        html.Div([
            html.Div(html.I(className="fas fa-female"), className="icon metric-icon"),
            html.H2(id='female-count', className="metric"),
            html.Div("Female Victims", className="metric-label")
        ], className='kpi-card'),
        html.Div([
            html.Div(html.I(className="fas fa-child"), className="icon metric-icon"),
            html.H2(id='minor-count', className="metric"),
            html.Div("Minors (Under 18)", className="metric-label")
        ], className='kpi-card'),
        html.Div([
            html.Div(html.I(className="fas fa-exclamation-triangle"), className="icon metric-icon"),
            html.H2(id='high-risk-count', className="metric"),
            html.Div("High-Risk Cases", className="metric-label")
        ], className='kpi-card'),
    ], className="metrics"),

    # Modern Filter Section - Power BI Style
    html.Div([
        html.Div([
            html.H3([html.I(className="fas fa-filter", style={'marginRight': '10px'}), "Data Filters"])
        ], className="filter-header"),
        html.Div([
            html.Div([
                html.Label([html.I(className="fas fa-map-marker-alt", style={'marginRight': '8px'}), "Select States/UTs"], className="filter-label"),
                dcc.Dropdown(
                    id='state-filter', 
                    options=[{'label': s, 'value': s} for s in sorted(df['state_clean'].unique()) if s is not None and s != 'Unknown'],
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
            ], className="filter-group")
        ], className="filter-grid")
    ], className="filter-section"),

    html.Br(),

    # Charts Section - Power BI Style
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
            html.Div(html.H3([html.I(className="fas fa-chart-pie", style={'marginRight': '10px'}), "Cases by Region Type"], className="chart-title"), className="chart-header"),
            dcc.Graph(id='region-pie-chart', style={'height': '400px'}, className="chart-wrapper")
        ], className="chart-container"),
        html.Div([
            html.Div(html.H3([html.I(className="fas fa-th", style={'marginRight': '10px'}), "Correlation Heatmap"], className="chart-title"), className="chart-header"),
            dcc.Graph(id='correlation-heatmap', style={'height': '400px'}, className="chart-wrapper")
        ], className="chart-container"),
    ], className="chart-grid"),

    html.Br(),

    # Map Section - Power BI Style
    html.Div(id="map-section"),
    html.Div([
        html.Div([
            html.H3([html.I(className="fas fa-map-marked-alt", style={'marginRight': '10px'}), "State-wise Trafficking Intensity"], className="map-title")
        ], className="map-header"),
        html.Div([
            dcc.Graph(id='india-map', style={'height': '600px'})
        ], className="map-wrapper")
    ], className="map-container")
],
    style={
        'backgroundColor': '#1e1e1e', 
        'fontFamily': 'Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif',
        'color': '#ffffff',
        'minHeight': '100vh'
    }
)

# ---------------------------
# Callbacks
# ---------------------------
@app.callback(
    Output('total-records', 'children'),
    Output('female-count', 'children'),
    Output('minor-count', 'children'),
    Output('high-risk-count', 'children'),
    Output('top-states-chart', 'figure'),
    Output('gender-region-chart', 'figure'),
    Output('age-risk-chart', 'figure'),
    Output('risk-distribution-chart', 'figure'),
    Output('region-pie-chart', 'figure'),
    Output('correlation-heatmap', 'figure'),
    Output('india-map', 'figure'),
    Input('state-filter', 'value'),
    Input('gender-filter', 'value'),
    Input('age-filter', 'value')
)
def update_dashboard(selected_states, selected_genders, selected_ages):
    dff = df.copy()
    if selected_states:
        dff = dff[dff['state_clean'].isin(selected_states)]
    if selected_genders:
        dff = dff[dff['gender'].isin(selected_genders)]
    if selected_ages:
        dff = dff[dff['age_group'].isin(selected_ages)]

    # KPIs
    total = len(dff)
    female = len(dff[dff['gender'] == 'Female'])
    minor = len(dff[dff['age_group'] == 'Below 18'])
    high_risk = len(dff[dff['risk_label'].str.lower() == 'high'])

    # Check if filtered data is empty
    if len(dff) == 0:
        # Return empty figures if no data matches the filters
        empty_fig = px.bar(title="No data available for selected filters")
        empty_fig.update_layout(plot_bgcolor='#0A192F', paper_bgcolor='#0A192F', font_color='white')

        return total, female, minor, high_risk, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # Charts
    try:
        # Top states chart - Enhanced Power BI styling
        state_counts = dff['state_clean'].value_counts().nlargest(10).reset_index()
        if len(state_counts) > 0:
            fig1 = px.bar(state_counts, x='index', y='state_clean', color='state_clean',
                         color_continuous_scale='Viridis', title="Top 10 States by Victims")
            fig1.update_traces(
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Victims: %{y}<extra></extra>'
            )
        else:
            fig1 = px.bar(title="No state data available")
        fig1.update_layout(
            plot_bgcolor='#242424', 
            paper_bgcolor='#242424', 
            font=dict(color='white', size=12),
            title_font=dict(size=14, color='white'),
            xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            margin=dict(l=60, r=60, t=60, b=60)
        )

        # Gender-Region chart - Enhanced styling
        gender_region = dff.groupby(['region_type', 'gender']).size().reset_index(name='Count')
        if len(gender_region) > 0:
            fig2 = px.bar(gender_region, x='region_type', y='Count', color='gender', barmode='group',
                         color_discrete_sequence=['#00bcf2', '#ff8c00', '#00b294', '#e81123'],
                         title="Gender Distribution by Region")
            fig2.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Region: %{x}<br>Count: %{y}<extra></extra>')
        else:
            fig2 = px.bar(title="No gender-region data available")
        fig2.update_layout(
            plot_bgcolor='#242424', 
            paper_bgcolor='#242424', 
            font=dict(color='white', size=12),
            title_font=dict(size=14, color='white'),
            xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            legend=dict(bgcolor='rgba(42, 42, 42, 0.8)', bordercolor='rgba(72, 72, 72, 0.5)', borderwidth=1),
            margin=dict(l=60, r=60, t=60, b=60)
        )

        # Age-Risk chart - Enhanced styling
        age_risk = dff.groupby(['age_group', 'risk_label']).size().reset_index(name='Count')
        if len(age_risk) > 0:
            fig3 = px.bar(age_risk, x='age_group', y='Count', color='risk_label', barmode='group',
                         color_discrete_map={'High': '#e81123', 'Low': '#00b294'},
                         title="Age Group vs Risk Level")
            fig3.update_traces(hovertemplate='<b>%{fullData.name} Risk</b><br>Age Group: %{x}<br>Count: %{y}<extra></extra>')
        else:
            fig3 = px.bar(title="No age-risk data available")
        fig3.update_layout(
            plot_bgcolor='#242424', 
            paper_bgcolor='#242424', 
            font=dict(color='white', size=12),
            title_font=dict(size=14, color='white'),
            xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            legend=dict(bgcolor='rgba(42, 42, 42, 0.8)', bordercolor='rgba(72, 72, 72, 0.5)', borderwidth=1),
            margin=dict(l=60, r=60, t=60, b=60)
        )

        # Risk distribution pie chart - Enhanced styling
        risk_dist = dff['risk_label'].value_counts().reset_index()
        if len(risk_dist) > 0:
            fig4 = px.pie(risk_dist, values='risk_label', names='index', title="Overall Risk Distribution",
                         color_discrete_map={'Low':'#00b294','High':'#e81123'},
                         hole=0.4)
            fig4.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label} Risk</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
        else:
            fig4 = px.pie(title="No risk data available")
        fig4.update_layout(
            paper_bgcolor='#242424', 
            font=dict(color='white', size=12),
            title_font=dict(size=14, color='white'),
            legend=dict(bgcolor='rgba(42, 42, 42, 0.8)', bordercolor='rgba(72, 72, 72, 0.5)', borderwidth=1),
            margin=dict(l=60, r=60, t=60, b=60)
        )

        # Region distribution pie chart - Enhanced styling
        region_dist = dff['region_type'].value_counts().reset_index()
        if len(region_dist) > 0:
            fig5 = px.pie(region_dist, values='region_type', names='index', title="Cases by Region Type",
                         color_discrete_sequence=['#00bcf2', '#ff8c00', '#00b294', '#e81123'],
                         hole=0.3)
            fig5.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
        else:
            fig5 = px.pie(title="No region data available")
        fig5.update_layout(
            paper_bgcolor='#242424', 
            font=dict(color='white', size=12),
            title_font=dict(size=14, color='white'),
            legend=dict(bgcolor='rgba(42, 42, 42, 0.8)', bordercolor='rgba(72, 72, 72, 0.5)', borderwidth=1),
            margin=dict(l=60, r=60, t=60, b=60)
        )

    except Exception as e:
        # If any chart fails, create empty charts
        empty_fig = px.bar(title=f"Chart error: {str(e)}")
        empty_fig.update_layout(plot_bgcolor='#0A192F', paper_bgcolor='#0A192F', font_color='white')
        fig1 = fig2 = fig3 = fig4 = fig5 = empty_fig

    # Create correlation heatmap with available data - Enhanced styling
    numeric_cols = dff.select_dtypes(include='number')
    if len(numeric_cols.columns) > 0:
        corr = numeric_cols.corr()
        fig6 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
        fig6.update_traces(
            textfont=dict(color='white', size=10),
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.2f}<extra></extra>'
        )
    else:
        # Create a placeholder heatmap if no numeric columns
        fig6 = px.imshow([[0, 0], [0, 0]], text_auto=True, title="Correlation Heatmap (No Numeric Data)")
    fig6.update_layout(
        plot_bgcolor='#242424', 
        paper_bgcolor='#242424', 
        font=dict(color='white', size=12),
        title_font=dict(size=14, color='white'),
        xaxis=dict(color='white'),
        yaxis=dict(color='white'),
        coloraxis_colorbar=dict(
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
        margin=dict(l=60, r=60, t=60, b=60)
    )

    # India Map - Enhanced with better boundaries and styling
    try:
        state_counts = dff['state_mapped'].value_counts().reset_index()
        state_counts.columns = ['State', 'Count']
        if len(state_counts) > 0:
            fig_map = px.choropleth(
                state_counts, 
                geojson=india_geojson, 
                locations='State',
                featureidkey="properties.st_nm", 
                color='Count',
                color_continuous_scale="Reds", 
                projection="natural earth",
                title="State-wise Human Trafficking Intensity"
            )
            fig_map.update_traces(
                marker_line_width=2, 
                marker_line_color="white",
                hovertemplate='<b>%{location}</b><br>Cases: %{z}<extra></extra>'
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
        else:
            fig_map = px.choropleth(
                [{'State': 'India', 'Count': 0}], 
                geojson=india_geojson,
                locations='State', 
                featureidkey="properties.st_nm", 
                color='Count',
                title="No map data available"
            )
            fig_map.update_geos(visible=False)
            
        fig_map.update_layout(
            plot_bgcolor='#242424',
            paper_bgcolor='#242424', 
            font=dict(color='white', size=12),
            title_font=dict(size=14, color='white'),
            geo=dict(
                bgcolor='#242424',
                showframe=True,
                framecolor='rgba(72, 72, 72, 0.8)',
                framewidth=2
            ),
            coloraxis_colorbar=dict(
                title_font=dict(color='white'),
                tickfont=dict(color='white'),
                bgcolor='rgba(42, 42, 42, 0.8)',
                bordercolor='rgba(72, 72, 72, 0.5)',
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )
    except Exception as e:
        # If map fails, create empty map
        fig_map = px.choropleth(
            [{'State': 'India', 'Count': 0}], 
            geojson=india_geojson,
            locations='State', 
            featureidkey="properties.st_nm", 
            color='Count',
            title=f"Map error: {str(e)}"
        )
        fig_map.update_geos(visible=False)
        fig_map.update_layout(
            paper_bgcolor='#242424', 
            font=dict(color='white', size=12),
            title_font=dict(size=14, color='white')
        )

    return total, female, minor, high_risk, fig1, fig2, fig3, fig4, fig5, fig6, fig_map


# ---------------------------
# Run Server
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
