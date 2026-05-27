from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime, timedelta
import warnings
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# Try to import advanced libraries (install if needed)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    FORECASTING_AVAILABLE = True
except ImportError:
    print("[WARNING] statsmodels not available. Installing...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'statsmodels'])
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.seasonal import seasonal_decompose
        FORECASTING_AVAILABLE = True
    except:
        print("[ERROR] Could not install statsmodels. Forecasting will be limited.")
        FORECASTING_AVAILABLE = False

app = Flask(__name__)

# ================================
# 📥 Load Dataset
# ================================
df = pd.read_csv("final_combined_dataset.csv")

# Clean state names - remove trailing/leading spaces
df['state'] = df['state'].str.strip()

categorical_cols = ['state', 'age_group', 'gender', 'region_type', 'risk_label']
for col in categorical_cols:
    df[col] = df[col].astype(str)

# ================================
# 🤖 Enhanced Analytics Functions
# ================================

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
    top_state = filtered_df['state'].value_counts().index[0]
    top_state_count = filtered_df['state'].value_counts().iloc[0]
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
    
    return insights

def chatbot_response(query):
    """Simple rule-based chatbot for data queries"""
    query = query.lower().strip()
    
    # Define response patterns
    if any(word in query for word in ['highest', 'most', 'maximum']):
        if 'state' in query:
            top_state = df['state'].value_counts().index[0]
            count = df['state'].value_counts().iloc[0]
            if 'minor' in query or 'child' in query:
                minors_by_state = df[df['age_group'] == 'Below 18']['state'].value_counts()
                top_state = minors_by_state.index[0]
                count = minors_by_state.iloc[0]
                return f"🧒 {top_state} has the highest number of minor victims: {count:,} cases"
            elif 'female' in query or 'women' in query:
                females_by_state = df[df['gender'] == 'Female']['state'].value_counts()
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
        """
    
    else:
        return "🤔 I didn't understand that. Try asking about states, victims, or type 'help' for options."

# ================================
# 🤖 PREDICTIVE MODELING FUNCTIONS
# ================================

def train_risk_prediction_model():
    """Train a model to predict risk levels based on features"""
    try:
        # Prepare features for modeling
        model_df = df.copy()
        
        # Encode categorical variables
        le_state = LabelEncoder()
        le_gender = LabelEncoder()
        le_age = LabelEncoder()
        le_region = LabelEncoder()
        
        model_df['state_encoded'] = le_state.fit_transform(model_df['state'])
        model_df['gender_encoded'] = le_gender.fit_transform(model_df['gender'])
        model_df['age_encoded'] = le_age.fit_transform(model_df['age_group'])
        model_df['region_encoded'] = le_region.fit_transform(model_df['region_type'])
        model_df['year_numeric'] = pd.to_numeric(model_df['year'], errors='coerce')
        
        # Features and target
        X = model_df[['state_encoded', 'gender_encoded', 'age_encoded', 'region_encoded', 'year_numeric']].dropna()
        y = model_df.loc[X.index, 'risk_label']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Logistic Regression with class balancing to handle imbalanced data
        model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced',  # This balances the classes automatically
            C=0.1  # Regularization to prevent overfitting
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        return {
            'model': model,
            'encoders': {'state': le_state, 'gender': le_gender, 'age': le_age, 'region': le_region},
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_names': ['state', 'gender', 'age_group', 'region_type', 'year']
        }
    except Exception as e:
        return {'error': str(e)}

def predict_risk(state, gender, age_group, region_type, year, model_info):
    """Make risk prediction for given inputs"""
    try:
        if 'error' in model_info:
            return {'error': model_info['error']}
        
        # Debug information
        print(f"\n[DEBUG] Input: {state}, {gender}, {age_group}, {region_type}, {year}")
        
        # Rule-based override for obvious cases based on data analysis
        if age_group == 'Below 18':
            print("[RULE] OVERRIDE: Below 18 = 100% High Risk in dataset")
            return {
                'prediction': 'High',
                'probability': {'High': 1.0, 'Low': 0.0},
                'confidence': 100.0,
                'method': 'Rule-based (Age < 18 = Always High Risk)'
            }
        elif age_group == 'Above 18':
            print("[RULE] OVERRIDE: Above 18 = 0% High Risk in dataset")
            return {
                'prediction': 'Low',
                'probability': {'High': 0.0, 'Low': 1.0},
                'confidence': 100.0,
                'method': 'Rule-based (Age > 18 = Always Low Risk)'
            }
        elif age_group == 'Unknown':
            print("[RULE] OVERRIDE: Unknown Age = 0% High Risk in dataset")
            return {
                'prediction': 'Low',
                'probability': {'High': 0.0, 'Low': 1.0},
                'confidence': 100.0,
                'method': 'Rule-based (Unknown Age = Always Low Risk)'
            }
        
        # Encode input features
        encoders = model_info['encoders']
        
        # Handle unseen categories with better fallbacks
        try:
            state_encoded = encoders['state'].transform([state])[0]
            print(f"[OK] State '{state}' encoded as: {state_encoded}")
        except ValueError:
            print(f"[WARNING] Unknown state '{state}', using fallback")
            state_encoded = 0  # Default to first category
            
        try:
            gender_encoded = encoders['gender'].transform([gender])[0]
            print(f"[OK] Gender '{gender}' encoded as: {gender_encoded}")
        except ValueError:
            print(f"[WARNING] Unknown gender '{gender}', using fallback")
            gender_encoded = 0
            
        try:
            age_encoded = encoders['age'].transform([age_group])[0]
            print(f"[OK] Age '{age_group}' encoded as: {age_encoded}")
        except ValueError:
            print(f"[WARNING] Unknown age '{age_group}', using fallback")
            age_encoded = 0
            
        try:
            region_encoded = encoders['region'].transform([region_type])[0]
            print(f"[OK] Region '{region_type}' encoded as: {region_encoded}")
        except ValueError:
            print(f"[WARNING] Unknown region '{region_type}', using fallback")
            region_encoded = 0
        
        # Create feature array
        features = np.array([[state_encoded, gender_encoded, age_encoded, region_encoded, int(year)]])
        print(f"[DATA] Feature vector: {features[0]}")
        
        # Make prediction
        prediction = model_info['model'].predict(features)[0]
        probability = model_info['model'].predict_proba(features)[0]
        
        print(f"[PREDICT] Raw prediction: {prediction}")
        print(f"[PREDICT] Raw probabilities: {probability}")
        print(f"[PREDICT] Model classes: {model_info['model'].classes_}")
        
        # Fix probability mapping
        class_names = model_info['model'].classes_
        prob_dict = {}
        for i, class_name in enumerate(class_names):
            prob_dict[class_name] = probability[i]
        
        result = {
            'prediction': prediction,
            'probability': prob_dict,
            'confidence': max(probability) * 100,
            'debug_info': {
                'encoded_features': features[0].tolist(),
                'class_names': class_names.tolist(),
                'raw_probabilities': probability.tolist()
            }
        }
        
        print(f"[RESULT] Final result: {result}")
        return result
        
    except Exception as e:
        print(f"[ERROR] Prediction error: {str(e)}")
        return {'error': str(e)}

# ================================
# 📈 ADVANCED MODEL 1: TIME-SERIES FORECASTING
# ================================

def create_forecasting_model(df_filtered):
    """
    Create time-series forecasting model to predict future trafficking trends
    """
    try:
        # Prepare time-series data
        ts_df = df_filtered[df_filtered['year'] != 'Unknown'].copy()
        ts_df['year'] = pd.to_numeric(ts_df['year'], errors='coerce')
        ts_df = ts_df.dropna(subset=['year'])
        
        if len(ts_df) == 0:
            return {'error': 'No valid time-series data available'}
        
        # Create monthly time series (simulate monthly data from yearly)
        yearly_counts = ts_df.groupby('year').size().reset_index(name='count')
        
        if len(yearly_counts) < 3:
            return {'error': 'Not enough historical data for forecasting (need at least 3 years)'}
        
        # Simple forecasting using linear trend
        years = yearly_counts['year'].values
        counts = yearly_counts['count'].values
        
        # Fit linear trend
        z = np.polyfit(years, counts, 1)
        trend_line = np.poly1d(z)
        
        # Generate forecasts for next 2 years
        max_year = years.max()
        forecast_years = [max_year + 1, max_year + 2]
        forecast_values = [max(0, int(trend_line(year))) for year in forecast_years]
        
        # Add some uncertainty bounds (simple approach)
        std_dev = np.std(counts)
        lower_bounds = [max(0, val - std_dev) for val in forecast_values]
        upper_bounds = [val + std_dev for val in forecast_values]
        
        # Advanced ARIMA forecasting if available
        if FORECASTING_AVAILABLE and len(yearly_counts) >= 5:
            try:
                # Fit ARIMA model
                model = ARIMA(counts, order=(1, 1, 1))
                fitted_model = model.fit()
                
                # Generate forecasts
                arima_forecast = fitted_model.forecast(steps=2)
                forecast_values = [max(0, int(val)) for val in arima_forecast]
                
                # Get confidence intervals
                forecast_ci = fitted_model.get_forecast(steps=2).conf_int()
                lower_bounds = [max(0, val) for val in forecast_ci.iloc[:, 0].values]
                upper_bounds = forecast_ci.iloc[:, 1].values
                
            except Exception as e:
                print(f"⚠️ ARIMA failed, using linear trend: {e}")
        
        return {
            'historical_years': years.tolist(),
            'historical_counts': counts.tolist(),
            'forecast_years': forecast_years,
            'forecast_values': forecast_values,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'trend_slope': z[0],
            'model_type': 'ARIMA' if FORECASTING_AVAILABLE else 'Linear Trend'
        }
        
    except Exception as e:
        print(f"❌ Forecasting error: {str(e)}")
        return {'error': str(e)}

def generate_forecast_chart(forecast_data):
    """Generate forecasting visualization"""
    if 'error' in forecast_data:
        return f"<div style='color:white; text-align:center; padding:50px;'>Forecasting not available: {forecast_data['error']}</div>"
    
    try:
        # Create forecast chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=forecast_data['historical_years'],
            y=forecast_data['historical_counts'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#00bcf2', width=3),
            marker=dict(size=8)
        ))
        
        # Forecast data
        all_years = forecast_data['historical_years'] + forecast_data['forecast_years']
        all_values = forecast_data['historical_counts'] + forecast_data['forecast_values']
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast_data['forecast_years'],
            y=forecast_data['forecast_values'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff6b35', width=3, dash='dash'),
            marker=dict(size=10, symbol='diamond')
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_data['forecast_years'] + forecast_data['forecast_years'][::-1],
            y=forecast_data['upper_bounds'] + forecast_data['lower_bounds'][::-1],
            fill='toself',
            fillcolor='rgba(255, 107, 53, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
        
        # Update layout
        trend_direction = "📈 Increasing" if forecast_data['trend_slope'] > 0 else "📉 Decreasing" if forecast_data['trend_slope'] < 0 else "➡️ Stable"
        
        fig.update_layout(
            title=f"Human Trafficking Forecast ({forecast_data['model_type']}) - {trend_direction} Trend",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            plot_bgcolor='rgba(36, 36, 36, 1)',
            paper_bgcolor='rgba(36, 36, 36, 1)',
            font=dict(color='white', size=12),
            title_font=dict(size=16, color='white'),
            height=500,
            xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            legend=dict(
                bgcolor='rgba(42, 42, 42, 0.8)',
                bordercolor='rgba(72, 72, 72, 0.5)',
                borderwidth=1
            )
        )
        
        return fig.to_html(full_html=False)
        
    except Exception as e:
        return f"<div style='color:white; text-align:center; padding:50px;'>Error generating forecast chart: {str(e)}</div>"

# ================================
# 🔍 ADVANCED MODEL 2: ANOMALY DETECTION
# ================================

def detect_anomalies(df_filtered):
    """
    Detect unusual patterns and outliers in trafficking data
    """
    try:
        # Prepare data for anomaly detection
        state_features = df_filtered.groupby('state').agg({
            'gender': 'count',  # total cases
            'age_group': lambda x: (x == 'Below 18').sum(),  # minor count
            'risk_label': lambda x: (x == 'High').sum(),  # high risk count
            'region_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'  # most common region type
        }).rename(columns={
            'gender': 'total_cases',
            'age_group': 'minor_cases',
            'risk_label': 'high_risk_cases'
        })
        
        # Add derived features
        state_features['minor_percentage'] = (state_features['minor_cases'] / state_features['total_cases'] * 100).fillna(0)
        state_features['high_risk_percentage'] = (state_features['high_risk_cases'] / state_features['total_cases'] * 100).fillna(0)
        
        # Female percentage
        female_by_state = df_filtered[df_filtered['gender'] == 'Female'].groupby('state').size()
        state_features['female_percentage'] = (female_by_state / state_features['total_cases'] * 100).fillna(0)
        
        if len(state_features) < 5:
            return {'error': 'Not enough states for meaningful anomaly detection'}
        
        # Features for anomaly detection
        features = ['total_cases', 'minor_percentage', 'high_risk_percentage', 'female_percentage']
        X = state_features[features].fillna(0)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.1,  # Expect ~10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        anomaly_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.score_samples(X)
        
        # Add results to dataframe
        state_features['is_anomaly'] = anomaly_labels == -1
        state_features['anomaly_score'] = anomaly_scores
        
        # Identify types of anomalies
        anomalies = state_features[state_features['is_anomaly']].copy()
        anomaly_details = []
        
        for state, row in anomalies.iterrows():
            reasons = []
            if row['total_cases'] > state_features['total_cases'].quantile(0.95):
                reasons.append(f"Unusually high case count ({row['total_cases']:.0f})")
            if row['minor_percentage'] > state_features['minor_percentage'].quantile(0.95):
                reasons.append(f"High minor victim rate ({row['minor_percentage']:.1f}%)")
            if row['high_risk_percentage'] > state_features['high_risk_percentage'].quantile(0.95):
                reasons.append(f"High risk case rate ({row['high_risk_percentage']:.1f}%)")
            if row['female_percentage'] < state_features['female_percentage'].quantile(0.05):
                reasons.append(f"Unusually low female victim rate ({row['female_percentage']:.1f}%)")
            
            anomaly_details.append({
                'state': state,
                'anomaly_score': row['anomaly_score'],
                'reasons': reasons,
                'total_cases': row['total_cases'],
                'severity': 'High' if row['anomaly_score'] < -0.5 else 'Medium'
            })
        
        # Sort by anomaly score (most anomalous first)
        anomaly_details.sort(key=lambda x: x['anomaly_score'])
        
        return {
            'state_features': state_features,
            'anomalies': anomaly_details,
            'total_anomalies': len(anomaly_details),
            'anomaly_rate': len(anomaly_details) / len(state_features) * 100,
            'features_used': features
        }
        
    except Exception as e:
        print(f"❌ Anomaly detection error: {str(e)}")
        return {'error': str(e)}

def generate_anomaly_chart(anomaly_data):
    """Generate anomaly detection visualization"""
    if 'error' in anomaly_data:
        return f"<div style='color:white; text-align:center; padding:50px;'>Anomaly detection not available: {anomaly_data['error']}</div>"
    
    try:
        state_features = anomaly_data['state_features']
        
        # Create scatter plot showing anomalies
        fig = go.Figure()
        
        # Normal states
        normal_states = state_features[~state_features['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_states['total_cases'],
            y=normal_states['high_risk_percentage'],
            mode='markers',
            name='Normal States',
            marker=dict(
                size=8,
                color='#00b294',
                opacity=0.7
            ),
            text=normal_states.index,
            hovertemplate='<b>%{text}</b><br>Total Cases: %{x}<br>High Risk %: %{y:.1f}%<extra></extra>'
        ))
        
        # Anomalous states
        anomalous_states = state_features[state_features['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomalous_states['total_cases'],
            y=anomalous_states['high_risk_percentage'],
            mode='markers',
            name='Anomalous States',
            marker=dict(
                size=12,
                color='#e81123',
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            text=anomalous_states.index,
            hovertemplate='<b>%{text}</b><br>Total Cases: %{x}<br>High Risk %: %{y:.1f}%<br><b>ANOMALY DETECTED</b><extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Anomaly Detection: {anomaly_data['total_anomalies']} Anomalies Detected ({anomaly_data['anomaly_rate']:.1f}%)",
            xaxis_title="Total Cases",
            yaxis_title="High Risk Percentage (%)",
            plot_bgcolor='rgba(36, 36, 36, 1)',
            paper_bgcolor='rgba(36, 36, 36, 1)',
            font=dict(color='white', size=12),
            title_font=dict(size=16, color='white'),
            height=500,
            xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            legend=dict(
                bgcolor='rgba(42, 42, 42, 0.8)',
                bordercolor='rgba(72, 72, 72, 0.5)',
                borderwidth=1
            )
        )
        
        return fig.to_html(full_html=False)
        
    except Exception as e:
        return f"<div style='color:white; text-align:center; padding:50px;'>Error generating anomaly chart: {str(e)}</div>"

# ================================
# 📊 COMPREHENSIVE ANALYSIS SUMMARY GENERATOR
# ================================

def generate_comprehensive_summary(df_filtered, anomaly_data=None, forecast_data=None):
    """
    Generate a comprehensive analysis summary of all charts, graphs, and insights
    """
    try:
        summary = {
            'overview': {},
            'demographic_insights': {},
            'geographic_insights': {},
            'risk_insights': {},
            'temporal_insights': {},
            'advanced_insights': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # ================================
        # OVERVIEW ANALYSIS
        # ================================
        total_cases = len(df_filtered)
        female_cases = len(df_filtered[df_filtered['gender'] == 'Female'])
        minor_cases = len(df_filtered[df_filtered['age_group'] == 'Below 18'])
        high_risk_cases = len(df_filtered[df_filtered['risk_label'] == 'High'])
        
        summary['overview'] = {
            'total_cases': total_cases,
            'female_percentage': round((female_cases / total_cases) * 100, 1),
            'minor_percentage': round((minor_cases / total_cases) * 100, 1),
            'high_risk_percentage': round((high_risk_cases / total_cases) * 100, 1),
            'data_completeness': 'High' if total_cases > 1000 else 'Medium' if total_cases > 100 else 'Low'
        }
        
        # ================================
        # DEMOGRAPHIC INSIGHTS
        # ================================
        gender_distribution = df_filtered['gender'].value_counts()
        age_distribution = df_filtered['age_group'].value_counts()
        
        # Gender insights
        dominant_gender = gender_distribution.index[0]
        gender_ratio = round(gender_distribution.iloc[0] / gender_distribution.iloc[1], 2) if len(gender_distribution) > 1 else 'N/A'
        
        # Age insights
        most_affected_age = age_distribution.index[0]
        age_risk_analysis = df_filtered.groupby('age_group')['risk_label'].apply(lambda x: (x == 'High').mean() * 100)
        highest_risk_age = age_risk_analysis.idxmax() if len(age_risk_analysis) > 0 else 'Unknown'
        
        summary['demographic_insights'] = {
            'dominant_gender': dominant_gender,
            'gender_ratio': f"{gender_ratio}:1" if gender_ratio != 'N/A' else 'Equal',
            'most_affected_age_group': most_affected_age,
            'highest_risk_age_group': highest_risk_age,
            'minor_vulnerability': 'CRITICAL' if minor_cases > 0 else 'Low'
        }
        
        # ================================
        # GEOGRAPHIC INSIGHTS
        # ================================
        state_distribution = df_filtered['state'].value_counts()
        top_states = state_distribution.head(5)
        
        # Regional analysis
        region_distribution = df_filtered['region_type'].value_counts()
        dominant_region = region_distribution.index[0] if len(region_distribution) > 0 else 'Unknown'
        
        # Geographic concentration
        top_5_concentration = (top_states.sum() / total_cases) * 100
        geographic_spread = 'Highly Concentrated' if top_5_concentration > 70 else 'Moderately Concentrated' if top_5_concentration > 40 else 'Well Distributed'
        
        summary['geographic_insights'] = {
            'most_affected_state': state_distribution.index[0],
            'most_affected_state_cases': int(state_distribution.iloc[0]),
            'top_5_states': list(top_states.index),
            'top_5_concentration_percentage': round(top_5_concentration, 1),
            'geographic_spread': geographic_spread,
            'dominant_region_type': dominant_region
        }
        
        # ================================
        # RISK INSIGHTS
        # ================================
        risk_distribution = df_filtered['risk_label'].value_counts()
        
        # Risk by demographics
        female_high_risk = len(df_filtered[(df_filtered['gender'] == 'Female') & (df_filtered['risk_label'] == 'High')])
        male_high_risk = len(df_filtered[(df_filtered['gender'] == 'Male') & (df_filtered['risk_label'] == 'High')])
        
        # Risk by age
        minor_high_risk = len(df_filtered[(df_filtered['age_group'] == 'Below 18') & (df_filtered['risk_label'] == 'High')])
        adult_high_risk = len(df_filtered[(df_filtered['age_group'] == 'Above 18') & (df_filtered['risk_label'] == 'High')])
        
        summary['risk_insights'] = {
            'overall_risk_level': 'HIGH' if (high_risk_cases / total_cases) > 0.3 else 'MEDIUM' if (high_risk_cases / total_cases) > 0.15 else 'LOW',
            'female_high_risk_cases': female_high_risk,
            'male_high_risk_cases': male_high_risk,
            'minor_high_risk_cases': minor_high_risk,
            'adult_high_risk_cases': adult_high_risk,
            'primary_risk_demographic': 'Females' if female_high_risk > male_high_risk else 'Males',
            'age_risk_pattern': 'Minors at highest risk' if minor_high_risk > adult_high_risk else 'Adults at highest risk'
        }
        
        # ================================
        # TEMPORAL INSIGHTS
        # ================================
        if 'year' in df_filtered.columns:
            yearly_data = df_filtered[df_filtered['year'] != 'Unknown']
            if len(yearly_data) > 0:
                yearly_counts = yearly_data['year'].value_counts().sort_index()
                
                if len(yearly_counts) > 1:
                    trend = 'Increasing' if yearly_counts.iloc[-1] > yearly_counts.iloc[0] else 'Decreasing' if yearly_counts.iloc[-1] < yearly_counts.iloc[0] else 'Stable'
                    peak_year = yearly_counts.idxmax()
                    
                    summary['temporal_insights'] = {
                        'trend': trend,
                        'peak_year': str(peak_year),
                        'peak_cases': int(yearly_counts.max()),
                        'years_covered': f"{yearly_counts.index.min()}-{yearly_counts.index.max()}",
                        'data_span': len(yearly_counts)
                    }
        
        # ================================
        # ADVANCED INSIGHTS
        # ================================
        advanced_insights = {}
        
        # Anomaly insights
        if anomaly_data and 'anomalies' in anomaly_data:
            anomalies = anomaly_data['anomalies']
            if anomalies:
                most_severe_anomaly = max(anomalies, key=lambda x: x.get('total_cases', 0))
                advanced_insights['anomaly_detection'] = {
                    'anomalies_detected': len(anomalies),
                    'most_concerning_state': most_severe_anomaly['state'],
                    'anomaly_rate': round(anomaly_data.get('anomaly_rate', 0), 1)
                }
        
        # Forecasting insights
        if forecast_data and 'trend_slope' in forecast_data:
            trend_direction = 'increasing' if forecast_data['trend_slope'] > 0 else 'decreasing' if forecast_data['trend_slope'] < 0 else 'stable'
            advanced_insights['forecasting'] = {
                'predicted_trend': trend_direction,
                'model_type': forecast_data.get('model_type', 'Unknown'),
                'next_year_forecast': forecast_data.get('forecast_values', [0])[0] if forecast_data.get('forecast_values') else 'N/A'
            }
        
        summary['advanced_insights'] = advanced_insights
        
        # ================================
        # KEY FINDINGS
        # ================================
        findings = []
        
        # Demographic findings
        if summary['overview']['female_percentage'] > 60:
            findings.append(f"🚨 **Gender Disparity**: {summary['overview']['female_percentage']:.1f}% of victims are female, indicating significant gender-based targeting")
        
        if summary['overview']['minor_percentage'] > 20:
            findings.append(f"👶 **Child Vulnerability**: {summary['overview']['minor_percentage']:.1f}% of victims are minors, highlighting child protection concerns")
        
        # Geographic findings
        if summary['geographic_insights']['top_5_concentration_percentage'] > 60:
            findings.append(f"🗺️ **Geographic Concentration**: Top 5 states account for {summary['geographic_insights']['top_5_concentration_percentage']:.1f}% of all cases")
        
        # Risk findings
        if summary['risk_insights']['overall_risk_level'] == 'HIGH':
            findings.append(f"⚠️ **High Risk Environment**: {summary['overview']['high_risk_percentage']:.1f}% of cases are classified as high-risk")
        
        # Anomaly findings
        if 'anomaly_detection' in advanced_insights:
            findings.append(f"🔍 **Anomalies Detected**: {advanced_insights['anomaly_detection']['anomalies_detected']} states show unusual trafficking patterns")
        
        summary['key_findings'] = findings
        
        # ================================
        # RECOMMENDATIONS
        # ================================
        recommendations = []
        
        # Based on demographics
        if summary['demographic_insights']['minor_vulnerability'] == 'CRITICAL':
            recommendations.append("🛡️ **Child Protection**: Implement enhanced child protection measures and awareness programs")
        
        if summary['overview']['female_percentage'] > 70:
            recommendations.append("👩 **Gender-Focused Programs**: Develop targeted intervention programs for women and girls")
        
        # Based on geography
        if summary['geographic_insights']['geographic_spread'] == 'Highly Concentrated':
            recommendations.append(f"🎯 **Targeted Intervention**: Focus resources on {summary['geographic_insights']['most_affected_state']} and other high-impact states")
        
        # Based on risk
        if summary['risk_insights']['overall_risk_level'] == 'HIGH':
            recommendations.append("🚨 **Immediate Action**: High risk levels require immediate intervention and resource allocation")
        
        # Based on anomalies
        if 'anomaly_detection' in advanced_insights and advanced_insights['anomaly_detection']['anomalies_detected'] > 0:
            recommendations.append(f"🔍 **Investigate Anomalies**: Conduct detailed investigation of {advanced_insights['anomaly_detection']['most_concerning_state']} and other anomalous states")
        
        summary['recommendations'] = recommendations
        
        # ================================
        # VISUAL KEY FINDINGS (structured for UI cards)
        # ================================
        findings_visual = []
        try:
            findings_visual.append({
                'title': 'Female Victims',
                'value': round(summary['overview']['female_percentage'], 1),
                'unit': '%',
                'icon': 'fa-female',
                'color': '#e81123',
                'progress': round(summary['overview']['female_percentage'], 1),
                'subtitle': 'Share of victims who are female'
            })
            findings_visual.append({
                'title': 'Minor Victims',
                'value': round(summary['overview']['minor_percentage'], 1),
                'unit': '%',
                'icon': 'fa-child',
                'color': '#ff8c00',
                'progress': round(summary['overview']['minor_percentage'], 1),
                'subtitle': 'Share of victims who are below 18'
            })
            findings_visual.append({
                'title': 'Top-5 Concentration',
                'value': round(summary['geographic_insights']['top_5_concentration_percentage'], 1),
                'unit': '%',
                'icon': 'fa-map-marked-alt',
                'color': '#ff6b35',
                'progress': round(summary['geographic_insights']['top_5_concentration_percentage'], 1),
                'subtitle': 'Cases concentrated in top 5 states'
            })
            findings_visual.append({
                'title': 'High-Risk Cases',
                'value': round(summary['overview']['high_risk_percentage'], 1),
                'unit': '%',
                'icon': 'fa-exclamation-triangle',
                'color': '#e81123',
                'progress': round(summary['overview']['high_risk_percentage'], 1),
                'subtitle': 'Cases labeled as High risk'
            })
            if summary.get('advanced_insights') and summary['advanced_insights'].get('anomaly_detection'):
                anom = summary['advanced_insights']['anomaly_detection']
                findings_visual.append({
                    'title': 'Anomalies Detected',
                    'value': int(anom.get('anomalies_detected', 0)),
                    'unit': '',
                    'icon': 'fa-bell',
                    'color': '#00bcf2',
                    'progress': round(anom.get('anomaly_rate', 0), 1),
                    'subtitle': f"States with unusual patterns ({anom.get('anomaly_rate', 0):.1f}% of states)"
                })
        except Exception:
            pass
        
        summary['key_findings_visual'] = findings_visual
        
        return summary
        
    except Exception as e:
        print(f"❌ Error generating summary: {str(e)}")
        return {'error': str(e)}

# ================================
# 🎯 ADVANCED MODEL 3: ADVANCED RISK CLASSIFICATION
# ================================

def train_advanced_risk_model():
    """
    Train advanced risk classification model using Gradient Boosting
    """
    try:
        # Prepare features for modeling
        model_df = df.copy()
        
        # Encode categorical variables
        le_state = LabelEncoder()
        le_gender = LabelEncoder()
        le_age = LabelEncoder()
        le_region = LabelEncoder()
        
        model_df['state_encoded'] = le_state.fit_transform(model_df['state'])
        model_df['gender_encoded'] = le_gender.fit_transform(model_df['gender'])
        model_df['age_encoded'] = le_age.fit_transform(model_df['age_group'])
        model_df['region_encoded'] = le_region.fit_transform(model_df['region_type'])
        model_df['year_numeric'] = pd.to_numeric(model_df['year'], errors='coerce')
        
        # Create additional engineered features
        model_df['is_female'] = (model_df['gender'] == 'Female').astype(int)
        model_df['is_minor'] = (model_df['age_group'] == 'Below 18').astype(int)
        
        # Features and target
        feature_cols = ['state_encoded', 'gender_encoded', 'age_encoded', 'region_encoded', 'year_numeric', 'is_female', 'is_minor']
        X = model_df[feature_cols].dropna()
        y = model_df.loc[X.index, 'risk_label']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Gradient Boosting Classifier
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            subsample=0.8
        )
        gb_model.fit(X_train, y_train)
        
        # Train Decision Tree as backup
        dt_model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        dt_model.fit(X_train, y_train)
        
        # Evaluate models
        gb_train_acc = gb_model.score(X_train, y_train)
        gb_test_acc = gb_model.score(X_test, y_test)
        gb_cv_scores = cross_val_score(gb_model, X, y, cv=5)
        
        dt_train_acc = dt_model.score(X_train, y_train)
        dt_test_acc = dt_model.score(X_test, y_test)
        dt_cv_scores = cross_val_score(dt_model, X, y, cv=5)
        
        # Choose best model
        best_model = gb_model if gb_cv_scores.mean() > dt_cv_scores.mean() else dt_model
        best_model_name = 'Gradient Boosting' if gb_cv_scores.mean() > dt_cv_scores.mean() else 'Decision Tree'
        best_cv_scores = gb_cv_scores if gb_cv_scores.mean() > dt_cv_scores.mean() else dt_cv_scores
        
        # Feature importance
        feature_importance = best_model.feature_importances_
        feature_names = ['State', 'Gender', 'Age Group', 'Region Type', 'Year', 'Is Female', 'Is Minor']
        importance_dict = dict(zip(feature_names, feature_importance))
        
        return {
            'model': best_model,
            'model_name': best_model_name,
            'encoders': {'state': le_state, 'gender': le_gender, 'age': le_age, 'region': le_region},
            'cv_mean': best_cv_scores.mean(),
            'cv_std': best_cv_scores.std(),
            'feature_importance': importance_dict,
            'feature_names': feature_names,
            'gb_performance': {'train': gb_train_acc, 'test': gb_test_acc, 'cv': gb_cv_scores.mean()},
            'dt_performance': {'train': dt_train_acc, 'test': dt_test_acc, 'cv': dt_cv_scores.mean()}
        }
        
    except Exception as e:
        return {'error': str(e)}

def predict_advanced_risk(state, gender, age_group, region_type, year, advanced_model_info):
    """
    Make advanced risk prediction with feature importance
    """
    try:
        if 'error' in advanced_model_info:
            return {'error': advanced_model_info['error']}
        
        print(f"\n🎯 Advanced Model - Input: {state}, {gender}, {age_group}, {region_type}, {year}")
        
        # Apply rule-based logic first (since it's 100% accurate for this dataset)
        if age_group == 'Below 18':
            print("🎯 RULE OVERRIDE: Below 18 = High Risk")
            return {
                'prediction': 'High',
                'probability': {'High': 1.0, 'Low': 0.0},
                'confidence': 100.0,
                'method': 'Rule-based (Age < 18)',
                'model_used': 'Rule-based Override'
            }
        elif age_group in ['Above 18', 'Unknown']:
            print("🎯 RULE OVERRIDE: Above 18/Unknown = Low Risk")
            return {
                'prediction': 'Low',
                'probability': {'High': 0.0, 'Low': 1.0},
                'confidence': 100.0,
                'method': 'Rule-based (Age >= 18)',
                'model_used': 'Rule-based Override'
            }
        
        # If somehow we get here (shouldn't happen with current data), use ML model
        encoders = advanced_model_info['encoders']
        
        # Encode features
        try:
            state_encoded = encoders['state'].transform([state])[0]
        except ValueError:
            state_encoded = 0
            
        try:
            gender_encoded = encoders['gender'].transform([gender])[0]
        except ValueError:
            gender_encoded = 0
            
        try:
            age_encoded = encoders['age'].transform([age_group])[0]
        except ValueError:
            age_encoded = 0
            
        try:
            region_encoded = encoders['region'].transform([region_type])[0]
        except ValueError:
            region_encoded = 0
        
        # Create feature array
        is_female = 1 if gender == 'Female' else 0
        is_minor = 1 if age_group == 'Below 18' else 0
        
        features = np.array([[state_encoded, gender_encoded, age_encoded, region_encoded, int(year), is_female, is_minor]])
        
        # Make prediction
        prediction = advanced_model_info['model'].predict(features)[0]
        probability = advanced_model_info['model'].predict_proba(features)[0]
        
        # Map probabilities
        class_names = advanced_model_info['model'].classes_
        prob_dict = {class_names[i]: probability[i] for i in range(len(class_names))}
        
        return {
            'prediction': prediction,
            'probability': prob_dict,
            'confidence': max(probability) * 100,
            'method': f'ML Model ({advanced_model_info["model_name"]})',
            'model_used': advanced_model_info['model_name'],
            'feature_importance': advanced_model_info['feature_importance']
        }
        
    except Exception as e:
        print(f"❌ Advanced prediction error: {str(e)}")
        return {'error': str(e)}

# ================================
# 📈 TIME-SERIES ANALYSIS FUNCTIONS 
# ================================

def create_time_series_charts(filtered_df):
    """Create comprehensive time-series analysis charts"""
    try:
        # Filter for valid years
        ts_df = filtered_df[filtered_df['year'] != 'Unknown'].copy()
        ts_df['year'] = pd.to_numeric(ts_df['year'], errors='coerce')
        ts_df = ts_df.dropna(subset=['year'])
        
        if len(ts_df) == 0:
            return {'error': 'No valid year data available'}
        
        # 1. Overall trend by year
        yearly_counts = ts_df.groupby('year').size().reset_index(name='count')
        
        # 2. Gender trends
        gender_yearly = ts_df.groupby(['year', 'gender']).size().reset_index(name='count')
        
        # 3. Age group trends  
        age_yearly = ts_df.groupby(['year', 'age_group']).size().reset_index(name='count')
        
        # 4. Risk level trends
        risk_yearly = ts_df.groupby(['year', 'risk_label']).size().reset_index(name='count')
        
        # 5. Region type trends
        region_yearly = ts_df.groupby(['year', 'region_type']).size().reset_index(name='count')
        
        return {
            'overall': yearly_counts,
            'gender': gender_yearly,
            'age': age_yearly,
            'risk': risk_yearly,
            'region': region_yearly,
            'year_range': f"{ts_df['year'].min():.0f} - {ts_df['year'].max():.0f}"
        }
    except Exception as e:
        return {'error': str(e)}

def generate_time_series_charts(ts_data):
    """Generate plotly charts for time-series data"""
    if 'error' in ts_data:
        return f"<div style='color:white; text-align:center; padding:50px;'>Time-series data not available: {ts_data['error']}</div>"
    
    try:
        # Create subplots for multiple trends
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Overall Trend Over Years',
                'Gender Distribution Trends', 
                'Risk Level Trends',
                'Age Group Trends'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Overall trend
        fig.add_trace(
            go.Scatter(
                x=ts_data['overall']['year'],
                y=ts_data['overall']['count'],
                mode='lines+markers',
                name='Total Cases',
                line=dict(color='#00bcf2', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Gender trends
        colors = {'Male': '#ff8c00', 'Female': '#e81123'}
        for gender in ts_data['gender']['gender'].unique():
            gender_data = ts_data['gender'][ts_data['gender']['gender'] == gender]
            fig.add_trace(
                go.Scatter(
                    x=gender_data['year'],
                    y=gender_data['count'],
                    mode='lines+markers',
                    name=f'{gender} Victims',
                    line=dict(color=colors.get(gender, '#00b294'), width=2),
                    marker=dict(size=6)
                ),
                row=1, col=2
            )
        
        # Risk level trends
        risk_colors = {'High': '#e81123', 'Low': '#00b294'}
        for risk in ts_data['risk']['risk_label'].unique():
            risk_data = ts_data['risk'][ts_data['risk']['risk_label'] == risk]
            fig.add_trace(
                go.Scatter(
                    x=risk_data['year'],
                    y=risk_data['count'],
                    mode='lines+markers',
                    name=f'{risk} Risk',
                    line=dict(color=risk_colors.get(risk, '#00b294'), width=2),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )
        
        # Age group trends
        age_colors = {'Below 18': '#ff6b35', '18-30': '#00d4ff', '31-45': '#6b46c1', 'Above 45': '#10b981', 'Unknown': '#64748b'}
        for age in ts_data['age']['age_group'].unique():
            age_data = ts_data['age'][ts_data['age']['age_group'] == age]
            fig.add_trace(
                go.Scatter(
                    x=age_data['year'],
                    y=age_data['count'],
                    mode='lines+markers',
                    name=age,
                    line=dict(color=age_colors.get(age, '#64748b'), width=2),
                    marker=dict(size=5)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Human Trafficking Trends Analysis ({ts_data['year_range']})",
            plot_bgcolor='rgba(36, 36, 36, 1)',
            paper_bgcolor='rgba(36, 36, 36, 1)',
            font=dict(color='white', size=11),
            title_font=dict(size=16, color='white'),
            height=600,
            showlegend=True,
            legend=dict(
                bgcolor='rgba(42, 42, 42, 0.8)',
                bordercolor='rgba(72, 72, 72, 0.5)',
                borderwidth=1,
                font=dict(size=10)
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Year", gridcolor='rgba(72, 72, 72, 0.5)', color='white')
        fig.update_yaxes(title_text="Number of Cases", gridcolor='rgba(72, 72, 72, 0.5)', color='white')
        
        return fig.to_html(full_html=False)
        
    except Exception as e:
        return f"<div style='color:white; text-align:center; padding:50px;'>Error generating time-series charts: {str(e)}</div>"

# ================================
# 💾 MODEL PERSISTENCE FUNCTIONS
# ================================

def save_model(model_info, filename):
    """Save model to disk"""
    try:
        models_dir = Path('saved_models')
        models_dir.mkdir(exist_ok=True)
        
        filepath = models_dir / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"[SUCCESS] Model saved: {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Error saving model {filename}: {e}")
        return False

def load_model(filename):
    """Load model from disk"""
    try:
        filepath = Path('saved_models') / f"{filename}.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                model_info = pickle.load(f)
            print(f"[SUCCESS] Model loaded from cache: {filename}")
            return model_info
        else:
            return None
    except Exception as e:
        print(f"[ERROR] Error loading model {filename}: {e}")
        return None

def is_model_outdated(filename, max_age_hours=24):
    """Check if model is older than specified hours"""
    try:
        filepath = Path('saved_models') / f"{filename}.pkl"
        if not filepath.exists():
            return True
        
        from datetime import datetime, timedelta
        file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        max_age = timedelta(hours=max_age_hours)
        
        return datetime.now() - file_time > max_age
    except:
        return True

# ================================
# 🤖 INTELLIGENT MODEL LOADING
# ================================

print("[INFO] Initializing ML models (with intelligent caching)...")

# Basic risk model
print("1/3 Loading/Training basic risk model...")
risk_model_info = load_model('basic_risk_model')
if risk_model_info is None or is_model_outdated('basic_risk_model', 24):
    print("[INFO] Training new basic risk model...")
    risk_model_info = train_risk_prediction_model()
    if 'error' not in risk_model_info:
        save_model(risk_model_info, 'basic_risk_model')
        print(f"[SUCCESS] Basic model trained & saved! CV Accuracy: {risk_model_info['cv_mean']:.3f} +/- {risk_model_info['cv_std']:.3f}")
    else:
        print(f"[ERROR] Basic model training failed: {risk_model_info['error']}")
else:
    print(f"[SUCCESS] Basic model loaded from cache! CV Accuracy: {risk_model_info['cv_mean']:.3f}")

# Advanced risk model
print("2/3 Loading/Training advanced risk model...")
advanced_model_info = load_model('advanced_risk_model')
if advanced_model_info is None or is_model_outdated('advanced_risk_model', 24):
    print("[INFO] Training new advanced risk model...")
    advanced_model_info = train_advanced_risk_model()
    if 'error' not in advanced_model_info:
        save_model(advanced_model_info, 'advanced_risk_model')
        print(f"[SUCCESS] Advanced model trained & saved! Model: {advanced_model_info['model_name']}, CV Accuracy: {advanced_model_info['cv_mean']:.3f}")
    else:
        print(f"[ERROR] Advanced model training failed: {advanced_model_info['error']}")
else:
    print(f"[SUCCESS] Advanced model loaded from cache! Model: {advanced_model_info.get('model_name', 'Unknown')}")

print("[SUCCESS] All models ready! (Training only happens once per 24 hours)")

# ================================
# ⚡ Dashboard Route
# ================================
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    # Filters
    selected_states = request.form.getlist('state') or df['state'].unique()
    selected_genders = request.form.getlist('gender') or df['gender'].unique()
    selected_age = request.form.getlist('age_group') or df['age_group'].unique()

    df_filtered = df[
        (df['state'].isin(selected_states)) &
        (df['gender'].isin(selected_genders)) &
        (df['age_group'].isin(selected_age))
    ]

    # ================================
    # 📊 Metrics
    # ================================
    total_victims = len(df_filtered)
    female_victims = len(df_filtered[df_filtered['gender']=='Female'])
    minor_victims = len(df_filtered[df_filtered['age_group']=='Below 18'])
    high_risk = len(df_filtered[df_filtered['risk_label']=='High'])

    # ================================
    # 🔹 Charts
    # ================================

    # Top 10 States - Enhanced styling
    top_states = df_filtered['state'].value_counts().head(10)
    fig1 = px.bar(
        top_states,
        x=top_states.index, y=top_states.values,
        color=top_states.values,
        color_continuous_scale='Viridis',
        title='Top 10 States by Victims',
        text=top_states.values
    )
    fig1.update_layout(
        yaxis_title="Number of Victims", 
        xaxis_title="State",
        plot_bgcolor='rgba(36, 36, 36, 1)',
        paper_bgcolor='rgba(36, 36, 36, 1)',
        font=dict(color='white', size=12),
        title_font=dict(size=16, color='white'),
        xaxis=dict(
            gridcolor='rgba(72, 72, 72, 0.5)', 
            color='white',
            tickangle=-45,  # Rotate state names for visibility
            tickfont=dict(size=10),  # Smaller font for state names
            automargin=True  # Automatically adjust margins for labels
        ),
        yaxis=dict(
            gridcolor='rgba(72, 72, 72, 0.5)', 
            color='white',
            autorange=True  # Ensure all bars are visible
        ),
        margin=dict(l=60, r=60, t=60, b=200),  # Much larger bottom margin
        height=600,  # Larger height to accommodate everything
        autosize=True,  # Auto-resize to fit container
        showlegend=False  # Remove legend to save space
    )
    fig1.update_traces(
        textposition='outside',
        textfont=dict(color='white', size=10),
        hovertemplate='<b>%{x}</b><br>Victims: %{y}<extra></extra>'
    )
    chart1 = fig1.to_html(full_html=False)

    # Victims by Gender & Region Type - Enhanced styling
    gender_region = df_filtered.groupby(['gender','region_type']).size().reset_index(name='count')
    fig2 = px.bar(
        gender_region, x='gender', y='count', color='region_type', barmode='group',
        color_discrete_sequence=['#00bcf2', '#ff8c00', '#00b294', '#e81123'],
        title="Victims by Gender & Region Type",
        text='count'
    )
    fig2.update_layout(
        yaxis_title="Number of Victims", 
        xaxis_title="Gender",
        plot_bgcolor='rgba(36, 36, 36, 1)',
        paper_bgcolor='rgba(36, 36, 36, 1)',
        font=dict(color='white', size=12),
        title_font=dict(size=16, color='white'),
        xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
        yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
        legend=dict(bgcolor='rgba(42, 42, 42, 0.8)', bordercolor='rgba(72, 72, 72, 0.5)', borderwidth=1),
        margin=dict(l=60, r=60, t=60, b=60),
        height=400
    )
    fig2.update_traces(
        textposition='outside',
        textfont=dict(color='white', size=10),
        hovertemplate='<b>%{fullData.name}</b><br>Gender: %{x}<br>Count: %{y}<extra></extra>'
    )
    chart2 = fig2.to_html(full_html=False)

    # Age Group vs Risk Level - Enhanced styling
    fig3 = px.bar(
        df_filtered, x='age_group', color='risk_label',
        barmode='stack',
        color_discrete_map={'Low':'#00b294','High':'#e81123'},
        title="Victim Age Group vs Risk Level",
        text_auto=True
    )
    fig3.update_layout(
        yaxis_title="Number of Victims", 
        xaxis_title="Age Group",
        plot_bgcolor='rgba(36, 36, 36, 1)',
        paper_bgcolor='rgba(36, 36, 36, 1)',
        font=dict(color='white', size=12),
        title_font=dict(size=16, color='white'),
        xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
        yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
        legend=dict(bgcolor='rgba(42, 42, 42, 0.8)', bordercolor='rgba(72, 72, 72, 0.5)', borderwidth=1, title_font=dict(color='white')),
        margin=dict(l=60, r=60, t=60, b=60),
        height=400
    )
    fig3.update_traces(
        textposition='inside',
        textfont=dict(color='white', size=10),
        hovertemplate='<b>%{fullData.name} Risk</b><br>Age Group: %{x}<br>Count: %{y}<extra></extra>'
    )
    chart3 = fig3.to_html(full_html=False)

    # Overall Risk Distribution Pie - Enhanced styling
    risk_counts = df_filtered['risk_label'].value_counts()
    fig4 = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color_discrete_map={'Low':'#00b294','High':'#e81123'},
        title="Overall Risk Distribution",
        hole=0.4  # Creates a donut chart
    )
    fig4.update_layout(
        plot_bgcolor='rgba(36, 36, 36, 1)',
        paper_bgcolor='rgba(36, 36, 36, 1)',
        font=dict(color='white', size=12),
        title_font=dict(size=16, color='white'),
        legend=dict(bgcolor='rgba(42, 42, 42, 0.8)', bordercolor='rgba(72, 72, 72, 0.5)', borderwidth=1),
        margin=dict(l=60, r=60, t=60, b=60),
        height=400
    )
    fig4.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont=dict(color='white', size=12),
        hovertemplate='<b>%{label} Risk</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        marker=dict(line=dict(color='rgba(72, 72, 72, 0.8)', width=2))
    )
    chart4 = fig4.to_html(full_html=False)

    # ================================
    # 🆕 NEW ENHANCED CHARTS
    # ================================
    
    # Enhanced Correlation Heatmap
    try:
        # Create a simpler, more robust correlation matrix
        features = ['gender', 'age_group', 'region_type', 'risk_label']
        
        # Convert categorical to numeric for correlation
        corr_df = df_filtered[features].copy()
        for col in features:
            corr_df[col] = pd.Categorical(corr_df[col]).codes
        
        # Calculate correlation matrix
        corr_matrix = corr_df.corr()
        
        # Create clean feature names
        clean_names = [f.replace('_', ' ').title() for f in features]
        corr_matrix.index = clean_names
        corr_matrix.columns = clean_names
        
        # Create heatmap
        fig5 = px.imshow(
            corr_matrix,
            text_auto='.3f',
            aspect='auto',
            color_continuous_scale=[
                [0, '#1e3a8a'],    # Dark blue for low correlation
                [0.5, '#64748b'],   # Gray for medium 
                [1.0, '#dc2626']    # Red for high correlation
            ],
            title="Feature Correlation Analysis",
            labels={'color': 'Correlation Strength'}
        )
        
        fig5.update_layout(
            plot_bgcolor='rgba(36, 36, 36, 1)',
            paper_bgcolor='rgba(36, 36, 36, 1)',
            font=dict(color='white', size=12),
            title_font=dict(size=16, color='white'),
            height=400,
            width=None,  # Auto-width
            autosize=True,
            margin=dict(l=60, r=60, t=80, b=60),
            xaxis=dict(title='Features', color='white', gridcolor='rgba(72, 72, 72, 0.5)'),
            yaxis=dict(title='Features', color='white', gridcolor='rgba(72, 72, 72, 0.5)')
        )
        chart5 = fig5.to_html(full_html=False)
    except Exception as e:
        chart5 = f"<div style='color:white; text-align:center; padding:50px;'>Correlation analysis not available: {str(e)}</div>"
    
    # Clustering Analysis
    try:
        # Create state-level features for clustering
        state_features = df_filtered.groupby('state').agg({
            'gender': 'count',  # total victims
            'age_group': lambda x: (x == 'Below 18').sum() / len(x),  # minor percentage
            'risk_label': lambda x: (x == 'High').sum() / len(x)  # risk percentage
        }).rename(columns={'gender': 'total_victims', 'age_group': 'minor_pct', 'risk_label': 'risk_pct'})
        
        # Add female percentage
        female_pct = df_filtered[df_filtered['gender'] == 'Female'].groupby('state').size() / df_filtered.groupby('state').size()
        state_features['female_pct'] = female_pct.fillna(0)
        
        # Filter states with enough data
        state_features = state_features[state_features['total_victims'] >= 10]
        
        # Normalize & cluster (DBSCAN)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(state_features[['minor_pct', 'female_pct', 'risk_pct']])
        try:
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(features_scaled)
            state_features['cluster'] = clustering.labels_
        except Exception:
            state_features['cluster'] = -1
        
        # 2D scatter: Minor% vs Risk%, size by total victims, color by cluster
        plot_df = state_features.reset_index()
        plot_df['cluster_label'] = plot_df['cluster'].replace({-1: 'Noise'}).astype(str)
        fig6 = px.scatter(
            plot_df,
            x='minor_pct', y='risk_pct',
            color='cluster_label',
            size='total_victims',
            hover_name='state',
            hover_data={'minor_pct': ':.1%', 'female_pct': ':.1%', 'risk_pct': ':.1%', 'total_victims': True},
            title="Regional Clustering: Minor% vs Risk% (DBSCAN)",
            labels={'minor_pct': 'Minor Victim %', 'risk_pct': 'High Risk %'}
        )
        
        fig6.update_layout(
            plot_bgcolor='rgba(36, 36, 36, 1)',
            paper_bgcolor='rgba(36, 36, 36, 1)',
            font=dict(color='white', size=12),
            title_font=dict(size=16, color='white'),
            height=400,
            width=None,
            autosize=True,
            margin=dict(l=60, r=60, t=80, b=60),
            xaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            yaxis=dict(gridcolor='rgba(72, 72, 72, 0.5)', color='white'),
            legend_title_text='Cluster'
        )
        chart6 = fig6.to_html(full_html=False)
    except Exception as e:
        chart6 = f"<div style='color:white; text-align:center; padding:50px;'>Clustering analysis not available: {str(e)}</div>"
    
    # ================================
    # 🤖 AI Insights Generation
    # ================================
    ai_insights = generate_insights(df_filtered)
    
    # ================================
    # 📈 TIME-SERIES ANALYSIS
    # ================================
    ts_data = create_time_series_charts(df_filtered)
    chart7 = generate_time_series_charts(ts_data)
    
    # ================================
    # 📈 FORECASTING
    # ================================
    forecast_data = create_forecasting_model(df_filtered)
    chart8 = generate_forecast_chart(forecast_data)
    
    # ================================
    # 🔍 ANOMALY DETECTION
    # ================================
    anomaly_data = detect_anomalies(df_filtered)
    chart9 = generate_anomaly_chart(anomaly_data)
    
    # ================================
    # 📈 COMPREHENSIVE ANALYSIS SUMMARY
    # ================================
    comprehensive_summary = generate_comprehensive_summary(df_filtered, anomaly_data, forecast_data)

    # ================================
    # Render Template
    # ================================
    return render_template(
        'index.html',
        total_victims=total_victims,
        female_victims=female_victims,
        minor_victims=minor_victims,
        high_risk=high_risk,
        states=df['state'].unique(),
        genders=df['gender'].unique(),
        age_groups=df['age_group'].unique(),
        selected_states=selected_states,
        selected_genders=selected_genders,
        selected_age=selected_age,
        chart1=chart1,
        chart2=chart2,
        chart3=chart3,
        chart4=chart4,
        chart5=chart5,
        chart6=chart6,
        chart7=chart7,
        chart8=chart8,
        chart9=chart9,
        ai_insights=ai_insights,
        anomaly_data=anomaly_data,
        comprehensive_summary=comprehensive_summary
    )

# ================================
# 🤖 Chatbot API Endpoint
# ================================
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    query = data.get('query', '')
    response = chatbot_response(query)
    return jsonify({'response': response})

# ================================
# 🔮 Prediction API Endpoint
# ================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract prediction parameters
        state = data.get('state', '')
        gender = data.get('gender', '')
        age_group = data.get('age_group', '')
        region_type = data.get('region_type', '')
        year = data.get('year', 2023)
        
        # Make prediction
        result = predict_risk(state, gender, age_group, region_type, year, risk_model_info)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

# ================================
# 🎯 Advanced Prediction API Endpoint
# ================================
@app.route('/predict-advanced', methods=['POST'])
def predict_advanced():
    try:
        data = request.get_json()
        
        # Extract prediction parameters
        state = data.get('state', '')
        gender = data.get('gender', '')
        age_group = data.get('age_group', '')
        region_type = data.get('region_type', '')
        year = data.get('year', 2023)
        
        # Make advanced prediction
        result = predict_advanced_risk(state, gender, age_group, region_type, year, advanced_model_info)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

# ================================
# 🔄 Manual Model Retraining Route
# ================================
@app.route('/retrain-models')
def retrain_models():
    """Manually retrain all models (for testing/updates)"""
    try:
        # Delete saved models to force retraining
        models_dir = Path('saved_models')
        if models_dir.exists():
            for model_file in models_dir.glob('*.pkl'):
                model_file.unlink()
        
        return "<h2>Models will be retrained on next app restart!</h2><p>Restart your Flask app to retrain all models.</p>"
    except Exception as e:
        return f"<h2>Error:</h2><p>{str(e)}</p>"

# ================================
# 📄 Comprehensive Report Download
# ================================
@app.route('/download-report')
def download_report():
    """Generate and download comprehensive analysis report"""
    try:
        # Use full dataset for report
        df_for_report = df.copy()
        
        # Generate comprehensive summary
        anomaly_data = detect_anomalies(df_for_report)
        forecast_data = create_forecasting_model(df_for_report)
        summary = generate_comprehensive_summary(df_for_report, anomaly_data, forecast_data)
        
        if 'error' in summary:
            return f"<h2>Error generating report:</h2><p>{summary['error']}</p>"
        
        # Create text report
        report_lines = []
        report_lines.append("📊 COMPREHENSIVE HUMAN TRAFFICKING ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("📈 EXECUTIVE SUMMARY")
        report_lines.append("-" * 30)
        report_lines.append(f"Total Cases Analyzed: {summary['overview']['total_cases']:,}")
        report_lines.append(f"Female Victims: {summary['overview']['female_percentage']}%")
        report_lines.append(f"Minor Victims: {summary['overview']['minor_percentage']}%")
        report_lines.append(f"High-Risk Cases: {summary['overview']['high_risk_percentage']}%")
        report_lines.append(f"Most Affected State: {summary['geographic_insights']['most_affected_state']}")
        report_lines.append("")
        
        # Key Findings
        report_lines.append("🔍 KEY FINDINGS")
        report_lines.append("-" * 30)
        for finding in summary['key_findings']:
            report_lines.append(f"• {finding}")
        report_lines.append("")
        
        # Detailed Insights
        report_lines.append("📊 DETAILED ANALYSIS")
        report_lines.append("-" * 30)
        
        report_lines.append("👥 Demographics:")
        report_lines.append(f"  - Gender Ratio: {summary['demographic_insights']['gender_ratio']}")
        report_lines.append(f"  - Dominant Gender: {summary['demographic_insights']['dominant_gender']}")
        report_lines.append(f"  - Vulnerability Level: {summary['demographic_insights']['minor_vulnerability']}")
        report_lines.append("")
        
        report_lines.append("🗺️ Geography:")
        report_lines.append(f"  - Geographic Spread: {summary['geographic_insights']['geographic_spread']}")
        report_lines.append(f"  - Top 5 Concentration: {summary['geographic_insights']['top_5_concentration_percentage']}%")
        report_lines.append(f"  - Dominant Region: {summary['geographic_insights']['dominant_region_type']}")
        report_lines.append("")
        
        report_lines.append("⚠️ Risk Analysis:")
        report_lines.append(f"  - Overall Risk Level: {summary['risk_insights']['overall_risk_level']}")
        report_lines.append(f"  - Primary Risk Group: {summary['risk_insights']['primary_risk_demographic']}")
        report_lines.append(f"  - Age Risk Pattern: {summary['risk_insights']['age_risk_pattern']}")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("💡 STRATEGIC RECOMMENDATIONS")
        report_lines.append("-" * 30)
        for recommendation in summary['recommendations']:
            report_lines.append(f"• {recommendation}")
        report_lines.append("")
        
        # Advanced Analytics
        if summary['advanced_insights']:
            report_lines.append("🧠 ADVANCED ANALYTICS")
            report_lines.append("-" * 30)
            if 'anomaly_detection' in summary['advanced_insights']:
                anom = summary['advanced_insights']['anomaly_detection']
                report_lines.append(f"Anomalies Detected: {anom['anomalies_detected']} states")
                report_lines.append(f"Most Concerning State: {anom['most_concerning_state']}")
            if 'forecasting' in summary['advanced_insights']:
                forecast = summary['advanced_insights']['forecasting']
                report_lines.append(f"Predicted Trend: {forecast['predicted_trend'].title()}")
                report_lines.append(f"Model Type: {forecast['model_type']}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create response
        report_text = "\n".join(report_lines)
        
        from flask import Response
        return Response(
            report_text,
            mimetype='text/plain',
            headers={'Content-Disposition': 'attachment;filename=trafficking_analysis_report.txt'}
        )
        
    except Exception as e:
        return f"<h2>Error:</h2><p>{str(e)}</p>"

# ================================
# Map Page
# ================================
@app.route('/map')
def map_page():
    # Load GeoJSON file
    geojson_path = os.path.join("india-maps-data-main", "geojson", "india.geojson")
    with open(geojson_path) as f:
        india_geojson = json.load(f)
    
    # High-Risk Data Processing
    state_risk = df.groupby('state')['risk_label'].apply(lambda x: (x=='High').sum()).reset_index()
    state_risk.columns = ['state', 'high_risk_count']
    
    # Since the GeoJSON has district-level data, we'll use it with state matching
    # The approach: match our state data to all districts of that state in GeoJSON
    
    # Create a mapping from each district to its state's risk count
    district_risk_data = []
    for feature in india_geojson['features']:
        properties = feature.get('properties', {})
        district = properties.get('district', 'Unknown')
        state_name = properties.get('st_nm', 'Unknown')
        
        # Skip if we don't have required data
        if district == 'Unknown' or state_name == 'Unknown':
            continue
            
        # Find matching state in our data
        state_match = state_risk[state_risk['state'] == state_name]
        if not state_match.empty:
            risk_count = state_match['high_risk_count'].iloc[0]
        else:
            # Try alternative state names
            alt_matches = state_risk[state_risk['state'].str.contains(state_name.split()[0], case=False, na=False)]
            if not alt_matches.empty:
                risk_count = alt_matches['high_risk_count'].iloc[0]
            else:
                risk_count = 0
        
        district_risk_data.append({
            'district': district,
            'state': state_name,
            'high_risk_count': risk_count
        })
    
    district_df = pd.DataFrame(district_risk_data)
    
    # Create choropleth using the district GeoJSON but with state-level data
    choropleth = px.choropleth(
        district_df,
        geojson=india_geojson,
        featureidkey="properties.district",
        locations='district',
        color='high_risk_count',
        color_continuous_scale=[
            [0, '#0a0e1a'],      # Very dark blue for low values
            [0.2, '#1e3a8a'],    # Dark blue
            [0.4, '#3b82f6'],    # Blue
            [0.6, '#f59e0b'],    # Amber
            [0.8, '#ef4444'],    # Red
            [1.0, '#dc2626']     # Dark red for high values
        ],
        title="High-Risk Areas in India",
        labels={'high_risk_count': 'High-Risk Cases'}
    )
    # Enhanced styling for choropleth map
    choropleth.update_traces(
        marker_line_width=0.5,
        marker_line_color="#ffffff",
        hovertemplate='<b>%{customdata[1]}</b><br>District: %{location}<br>High-Risk Cases: %{z}<br><extra></extra>'
    )
    
    # Add state names to hover data
    choropleth.update_traces(
        customdata=district_df[['district', 'state']].values
    )
    
    choropleth.update_layout(
        plot_bgcolor='rgba(15, 20, 25, 1)',
        paper_bgcolor='rgba(15, 20, 25, 1)',
        font=dict(color='white', size=12),
        title_font=dict(size=18, color='white'),
        geo=dict(
            bgcolor='rgba(15, 20, 25, 1)',
            showframe=False,
            showlakes=False,
            showocean=False,
            showland=False,
            showcoastlines=False,
            showcountries=False,
            showsubunits=False
        ),
        coloraxis_colorbar=dict(
            title="High-Risk Cases",
            title_font=dict(color='white', size=14),
            tickfont=dict(color='white', size=12),
            bgcolor='rgba(26, 35, 50, 0.9)',
            bordercolor='#ffffff',
            borderwidth=2,
            thickness=20,
            len=0.8
        ),
        margin=dict(l=20, r=20, t=80, b=20),
        height=650
    )
    
    # Fit bounds to show only India
    choropleth.update_geos(
        fitbounds="locations",
        visible=False
    )
    choropleth_html = choropleth.to_html(full_html=False)

    # Trafficking Routes
    state_coords = {
        'Maharashtra': [19.7515, 75.7139],
        'Delhi': [28.7041, 77.1025],
        'Tamil Nadu': [11.1271, 78.6569],
        'Uttar Pradesh': [26.8467, 80.9462],
        'West Bengal': [22.9868, 87.8550],
        'Karnataka': [15.3173, 75.7139],
        'Gujarat': [22.2587, 71.1924],
    }

    routes = [
        ('Maharashtra', 'Delhi'),
        ('Uttar Pradesh', 'West Bengal'),
        ('Tamil Nadu', 'Karnataka'),
        ('Gujarat', 'Maharashtra'),
    ]

    routes_map = go.Figure()
    # Add bounds to restrict view to India only
    routes_map.update_layout(
        mapbox=dict(
            style="carto-darkmatter",  # Dark theme for consistency
            zoom=4.8,  # Zoom in more on India
            center=dict(lat=22.9734, lon=78.6569),  # Center on India
        ),
        margin=dict(r=20, t=80, l=20, b=20),
        title=dict(
            text="Simulated Human Trafficking Routes in India",
            x=0.5,
            font=dict(size=18, color='white')
        ),
        plot_bgcolor='rgba(15, 20, 25, 1)',
        paper_bgcolor='rgba(15, 20, 25, 1)',
        font=dict(color='white', size=12),
        height=650,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26, 35, 50, 0.9)',
            bordercolor='#ffffff',
            borderwidth=1,
            font=dict(color='white')
        )
    )

    # Color scheme for different routes
    route_colors = ['#00d4ff', '#ff6b35', '#00c896', '#6b46c1']
    
    for i, (origin, dest) in enumerate(routes):
        o_lat, o_lon = state_coords.get(origin, [0,0])
        d_lat, d_lon = state_coords.get(dest, [0,0])

        # Create curved route path
        lats = np.linspace(o_lat, d_lat, 50)
        lons = np.linspace(o_lon, d_lon, 50) + np.sin(np.linspace(0, np.pi, 50)) * 1.5
        
        color = route_colors[i % len(route_colors)]
        
        # Add route line
        routes_map.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='lines',
            line=dict(width=5, color=color),
            hoverinfo='text',
            text=f"<b>Route:</b> {origin} → {dest}",
            hovertemplate='<b>Trafficking Route</b><br>From: <b>%{customdata[0]}</b><br>To: <b>%{customdata[1]}</b><extra></extra>',
            customdata=[[origin, dest]] * len(lats),
            name=f"{origin} → {dest}",
            showlegend=True
        ))
        
        # Add origin marker (larger, with symbol)
        routes_map.add_trace(go.Scattermapbox(
            lat=[o_lat],
            lon=[o_lon],
            mode='markers+text',
            marker=dict(size=20, color=color, opacity=1.0, symbol='circle'),
            text=[f"FROM: {origin}"],
            textposition="top center",
            textfont=dict(size=12, color='white'),
            hoverinfo='text',
            hovertext=f"<b>Source:</b> {origin}<br><b>Route:</b> {origin} → {dest}",
            hovertemplate='%{hovertext}<extra></extra>',
            name=f"Source: {origin}",
            showlegend=False
        ))
        
        # Add destination marker (larger, with different symbol)
        routes_map.add_trace(go.Scattermapbox(
            lat=[d_lat],
            lon=[d_lon],
            mode='markers+text',
            marker=dict(size=20, color=color, opacity=1.0, symbol='square'),
            text=[f"TO: {dest}"],
            textposition="bottom center",
            textfont=dict(size=12, color='white'),
            hoverinfo='text',
            hovertext=f"<b>Destination:</b> {dest}<br><b>Route:</b> {origin} → {dest}",
            hovertemplate='%{hovertext}<extra></extra>',
            name=f"Destination: {dest}",
            showlegend=False
        ))
        
        # Add directional arrow at midpoint
        mid_idx = len(lats) // 2
        routes_map.add_trace(go.Scattermapbox(
            lat=[lats[mid_idx]],
            lon=[lons[mid_idx]],
            mode='markers',
            marker=dict(size=15, color='white', symbol='arrow', angle=45),
            hoverinfo='text',
            hovertext=f"<b>Direction:</b> {origin} → {dest}",
            hovertemplate='%{hovertext}<extra></extra>',
            showlegend=False
        ))

    # Restrict map bounds to India only
    routes_map.update_layout(
        mapbox_bounds={
            "west": 68,
            "east": 97, 
            "south": 6,
            "north": 37
        }
    )
    
    routes_html = routes_map.to_html(full_html=False)

    return render_template('map.html', choropleth_html=choropleth_html, routes_html=routes_html)


if __name__ == '__main__':
    app.run(debug=True)
