# Human Trafficking Risk Analytics Dashboard

> An ML-powered analytics platform for assessing human trafficking risk patterns across India, built for law enforcement, researchers, and policymakers.

**Research-backed** · Research paper under minor revision at *Expert Systems with Applications*, Elsevier (Impact Factor 7.5)

---

## Overview

This project is an open-source, interactive analytics dashboard built on 43,671 victim records spanning multiple years across all Indian states and union territories. It combines machine learning, time-series forecasting, anomaly detection, and geospatial visualization into a unified platform to surface actionable insights on human trafficking patterns in India.

---

## Features

- **Risk Classification** — Hybrid XGBoost + LightGBM ensemble achieving 93.4% accuracy and 0.97 AUC
- **Time-Series Forecasting** — ARIMA(1,1,1) model for year-on-year trend prediction
- **Anomaly Detection** — Isolation Forest for identifying statistical outliers in trafficking data
- **Clustering** — DBSCAN-based regional pattern identification and corridor analysis
- **Geospatial Visualization** — Interactive choropleth maps with state-level filtering
- **Automated Insights Engine** — AI-generated trend summaries and key findings
- **NLP Chatbot** — Natural language interface for querying the dataset
- **Downloadable Reports** — One-click CSV export for research use

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python, Flask |
| ML / Analytics | XGBoost, LightGBM, Scikit-learn, Statsmodels |
| Visualization | Plotly, Dash, Dash Bootstrap Components |
| Data Processing | Pandas, NumPy |
| Geospatial | GeoJSON (all Indian states), DBSCAN clustering |
| Frontend | HTML, CSS, Bootstrap, JavaScript |

---

## Project Structure

```
human-trafficking-risk-analytics/
├── app.py                        # Main Flask application
├── dash_app.py                   # Dash-based dashboard
├── enhanced_dash_app.py          # Enhanced Dash version with full features
├── load_and_clean.py             # Data loading and preprocessing pipeline
├── gui_app.py                    # GUI interface
├── saved_models/                 # Pre-trained ML model files
│   ├── advanced_risk_model.pkl
│   └── basic_risk_model.pkl
├── templates/                    # HTML templates
│   ├── index.html
│   └── map.html
├── static/css/                   # Stylesheets
├── assets/                       # Dash assets
├── india-maps-data-main/         # GeoJSON & TopoJSON for all Indian states
├── requirements_enhanced.txt     # Python dependencies
├── ARIMA_EXPLANATION.md
├── DASHBOARD_COMPONENTS_DETAILED.md
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Explorer1905/human-trafficking-risk-analytics.git
cd human-trafficking-risk-analytics

# Install dependencies
pip install -r requirements_enhanced.txt
```

### Running the App

```bash
# Flask app
python app.py

# Or the enhanced Dash app
python enhanced_dash_app.py
```

Then open `http://localhost:5000` in your browser.

### Dataset

The dataset contains 43,671 records with the following key columns:

| Column | Description |
|--------|-------------|
| `state` | Indian state / union territory |
| `age_group` | Victim age category (Below 18, 18–30, Above 30) |
| `gender` | Male / Female / Unknown |
| `region_type` | State or union territory |
| `risk_label` | Target variable — High / Low risk |
| `year` | Year of record |

> **Note:** The raw dataset is not included in this repository as it is linked to an ongoing research submission. The dataset is available on request for research purposes — please open an issue or email the maintainer.

---

## Model Performance

| Model | Accuracy | AUC |
|-------|----------|-----|
| XGBoost + LightGBM Ensemble | 93.4% | 0.97 |
| Isolation Forest (Anomaly Detection) | — | — |
| ARIMA(1,1,1) (Forecasting) | — | — |

---

## Contributing

Contributions are welcome! Here are some ways to get involved:

**Good first issues (beginner-friendly)**
- Improve README documentation
- Add unit tests for data loading functions
- Fix UI/UX issues in the dashboard
- Add new chart types to the visualization layer

**Intermediate issues**
- Add support for newer NCRB datasets (2021–2023)
- Improve the NLP chatbot with better intent parsing
- Add REST API endpoints for external data access
- Mobile-responsive improvements

**Advanced issues**
- Integrate newer ML models (CatBoost, TabNet)
- Build a data ingestion pipeline for live NCRB updates
- Extend geospatial analysis to district-level mapping

To contribute:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes and commit (`git commit -m 'Add: your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

---

## Research

This project is backed by an ongoing research paper:

> **Shravani Chavan et al.** — *A Multi-Model Analytics Framework for Human Trafficking Risk Assessment and Trend Prediction in India*  
> *Expert Systems with Applications*, Elsevier (Impact Factor 7.5) — Under Minor Revision (2026)

---

## Author

**Shravani Chavan**  
Computer Engineering Student, Vidyalankar Institute of Technology, Mumbai  
[GitHub](https://github.com/Explorer1905) · [LinkedIn](https://linkedin.com/in/shravani-chavan)

---

*Built with the goal of using data and AI to support anti-trafficking efforts in India.*
