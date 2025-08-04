import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="Crop Price Prediction Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the CSV data"""
    try:
        df = pd.read_csv("AP_Crop_Wholesale_Prices_2020_2024.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load all trained ARIMA models"""
    models = {}
    model_dir = "crop_price_models"
    
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith("_arima_model.pkl"):
                crop_name = filename.replace("_arima_model.pkl", "").title()
                try:
                    with open(os.path.join(model_dir, filename), "rb") as f:
                        models[crop_name] = pickle.load(f)
                except Exception as e:
                    st.warning(f"Error loading model for {crop_name}: {e}")
    
    return models

def get_available_crops(df):
    """Get list of available crops"""
    if df is not None:
        return sorted(df["Crop"].unique())
    return []

def create_price_chart(df, selected_crop):
    """Create historical price chart"""
    if df is None or selected_crop is None:
        return None
    
    crop_data = df[df["Crop"] == selected_crop].copy()
    crop_data["Date"] = pd.to_datetime(crop_data["Month"] + " " + crop_data["Year"].astype(str))
    crop_data = crop_data.sort_values("Date")
    
    fig = px.line(
        crop_data, 
        x="Date", 
        y="Average_Wholesale_Price (Rs/quintal)",
        title=f"Historical Price Trend for {selected_crop}",
        labels={"Average_Wholesale_Price (Rs/quintal)": "Price (Rs/quintal)", "Date": "Date"}
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (Rs/quintal)",
        hovermode="x unified"
    )
    return fig

def create_forecast_chart(historical_data, forecast_data, selected_crop):
    """Create forecast chart with historical and predicted data"""
    if historical_data is None or forecast_data is None:
        return None
    
    # Prepare historical data
    hist_df = historical_data.copy()
    hist_df["Date"] = pd.to_datetime(hist_df["Month"] + " " + hist_df["Year"].astype(str))
    hist_df = hist_df.sort_values("Date")
    
    # Prepare forecast data
    last_date = hist_df["Date"].max()
    forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, 7)]
    
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Price": forecast_data,
        "Type": "Forecast"
    })
    
    # Combine historical and forecast data
    hist_df["Type"] = "Historical"
    combined_df = pd.concat([
        hist_df[["Date", "Average_Wholesale_Price (Rs/quintal)", "Type"]].rename(
            columns={"Average_Wholesale_Price (Rs/quintal)": "Price"}
        ),
        forecast_df
    ])
    
    # Create the chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=hist_df["Date"],
        y=hist_df["Average_Wholesale_Price (Rs/quintal)"],
        mode="lines+markers",
        name="Historical",
        line=dict(color="blue", width=2),
        marker=dict(size=6)
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Price"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="red", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond")
    ))
    
    fig.update_layout(
        title=f"Price Forecast for {selected_crop}",
        xaxis_title="Date",
        yaxis_title="Price (Rs/quintal)",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Crop Price Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        df = load_data()
        models = load_models()
    
    if df is None:
        st.error("Failed to load data. Please check if the CSV file exists.")
        return
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Crop selection
    available_crops = get_available_crops(df)
    selected_crop = st.sidebar.selectbox(
        "Select Crop for Prediction:",
        available_crops,
        index=0 if available_crops else None
    )
    
    # Main content
    if selected_crop:
        st.header(f"📈 Analysis for {selected_crop}")
        
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Historical price chart
            st.subheader("Historical Price Trend")
            hist_chart = create_price_chart(df, selected_crop)
            if hist_chart:
                st.plotly_chart(hist_chart, use_container_width=True)
        
        with col2:
            # Statistics
            st.subheader("📊 Statistics")
            crop_data = df[df["Crop"] == selected_crop]
            
            if not crop_data.empty:
                current_price = crop_data["Average_Wholesale_Price (Rs/quintal)"].iloc[-1]
                avg_price = crop_data["Average_Wholesale_Price (Rs/quintal)"].mean()
                min_price = crop_data["Average_Wholesale_Price (Rs/quintal)"].min()
                max_price = crop_data["Average_Wholesale_Price (Rs/quintal)"].max()
                
                st.metric("Current Price", f"₹{current_price:.2f}")
                st.metric("Average Price", f"₹{avg_price:.2f}")
                st.metric("Min Price", f"₹{min_price:.2f}")
                st.metric("Max Price", f"₹{max_price:.2f}")
        
        # Prediction section
        st.header("🔮 Price Prediction")
        
        if selected_crop.title() in models:
            try:
                # Get forecast
                model = models[selected_crop.title()]
                forecast = model.forecast(steps=6)
                forecast_values = forecast.round(2).tolist()
                
                # Create forecast chart
                forecast_chart = create_forecast_chart(
                    df[df["Crop"] == selected_crop], 
                    forecast_values, 
                    selected_crop
                )
                
                if forecast_chart:
                    st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Forecast details
                st.subheader("📅 6-Month Forecast")
                
                # Create forecast table
                forecast_dates = []
                current_date = datetime.now()
                for i in range(1, 7):
                    forecast_date = current_date + timedelta(days=30*i)
                    forecast_dates.append(forecast_date.strftime("%B %Y"))
                
                forecast_df = pd.DataFrame({
                    "Month": forecast_dates,
                    "Predicted Price (₹/quintal)": forecast_values
                })
                
                st.dataframe(forecast_df, use_container_width=True)
                
                # Key insights
                st.subheader("💡 Key Insights")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    price_change = forecast_values[-1] - current_price
                    change_percent = (price_change / current_price) * 100
                    st.metric(
                        "6-Month Change", 
                        f"₹{price_change:.2f}",
                        f"{change_percent:.1f}%"
                    )
                
                with col2:
                    st.metric("Highest Forecast", f"₹{max(forecast_values):.2f}")
                
                with col3:
                    st.metric("Lowest Forecast", f"₹{min(forecast_values):.2f}")
                
            except Exception as e:
                st.error(f"Error generating prediction for {selected_crop}: {e}")
        else:
            st.warning(f"No trained model available for {selected_crop}. Please ensure the model has been trained.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🌾 Crop Price Prediction Dashboard | Powered by ARIMA Time Series Analysis</p>
            <p>Data Source: AP Crop Wholesale Prices 2020-2024</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 