import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pydeck as pdk
from streamlit_extras.stylable_container import stylable_container
from utils import data_loader
import time

# === Page Config ===
st.set_page_config(
    page_title="ETF Sector Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Custom CSS for cyberpunk theme ===
st.markdown("""
<style>
    /* Global theme */
    .stApp {
        background-color: #0A0E17;
        color: #E0E7FF;
    }
    
    /* Neon text effects */
    .neon-header {
        font-family: 'Orbitron', sans-serif;
        color: #00F5D4;
        text-shadow: 0 0 5px #00F5D4, 0 0 15px #00F5D4, 0 0 30px #00F5D4;
        letter-spacing: 1px;
    }
    
    .neon-subhead {
        font-family: 'Rajdhani', sans-serif;
        color: #00CFFF;
        text-shadow: 0 0 5px #00CFFF;
        letter-spacing: 0.5px;
    }
    
    /* Neon boxes */
    .neon-box {
        background-color: rgba(10, 14, 23, 0.7);
        border: 1px solid #00F5D4;
        border-radius: 8px;
        box-shadow: 0 0 8px #00F5D4;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .neon-box-blue {
        background-color: rgba(10, 14, 23, 0.7);
        border: 1px solid #00CFFF;
        border-radius: 8px;
        box-shadow: 0 0 8px #00CFFF;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .neon-box-purple {
        background-color: rgba(10, 14, 23, 0.7);
        border: 1px solid #BF00FF;
        border-radius: 8px;
        box-shadow: 0 0 8px #BF00FF;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(45deg, #00F5D4, #00CFFF);
        color: #0A0E17;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background: linear-gradient(45deg, #00CFFF, #BF00FF);
        box-shadow: 0 0 10px #00CFFF;
        transform: translateY(-2px);
    }
    
    /* Metric animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    
    .metric-container {
        animation: pulse 2s infinite;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #0E131F;
    }
    
    /* Table styling */
    .dataframe {
        background-color: #0E131F;
        border: 1px solid #00CFFF;
    }
    
    .dataframe th {
        background-color: #1A1F35;
        color: #00F5D4;
        text-align: left;
        padding: 8px;
    }
    
    .dataframe td {
        padding: 8px;
        border-bottom: 1px solid #1A1F35;
    }
    
    /* Card containers */
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: space-between;
    }
    
    .card {
        background-color: #0E131F;
        border-radius: 8px;
        padding: 15px;
        flex: 1;
        min-width: 200px;
        border-left: 3px solid #00F5D4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 245, 212, 0.2);
    }
    
</style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown("""
    <h1 class="neon-header">🚀 ETF SECTOR DASHBOARD</h1>
    <p style='color:#B0BEC5;'>Welcome to the financial matrix. Enjoy the analytics.</p>
""", unsafe_allow_html=True)

# === Sidebar ETF Reference ===
with st.sidebar:
    st.markdown('<h2 class="neon-subhead">🔍 ETF SECTOR CODEX</h2>', unsafe_allow_html=True)
    
    # Create a visually appealing table for the ETF reference
    etf_data = {
        "Ticker": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLB", "XLU", "XLRE", "XLC"],
        "Sector": ["Technology", "Financials", "Energy", "Healthcare", "Consumer Discr.", 
                  "Industrials", "Materials", "Utilities", "Real Estate", "Communication"]
    }
    
    etf_df = pd.DataFrame(etf_data)
    
    # Add a color column for visual distinction
    colors = ['#00F5D4', '#00E5FF', '#00CFFF', '#00BFFF', '#00AAFF', 
             '#0099FF', '#0088FF', '#0077FF', '#0066FF', '#0055FF']
    
    etf_df['Color'] = colors
    
    # Display styled table
    st.dataframe(
        etf_df[['Ticker', 'Sector']],
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker",
                help="ETF ticker symbol",
                width="small",
            ),
            "Sector": st.column_config.TextColumn(
                "Sector",
                help="Market sector",
                width="medium",
            ),
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown('<h3 class="neon-subhead">🔄 DATA REFRESH</h3>', unsafe_allow_html=True)
    if st.button("Force Data Reload"):
        st.session_state['data_timestamp'] = time.time()
        st.toast("Data cache cleared!", icon="🔄")

# Initialize session state for region selection
if 'selected_region' not in st.session_state:
    st.session_state['selected_region'] = 'USA'

# === Main Layout Structure ===
# Create two columns for the top section
col1, col2 = st.columns([2, 1])

# === Region Map in Column 1 ===
with col1:
    with stylable_container(
        key="map_container",
        css_styles="""
            {
                border: 1px solid #00CFFF;
                border-radius: 8px;
                padding: 10px;
                background-color: rgba(10, 14, 23, 0.7);
                box-shadow: 0 0 8px #00CFFF;
            }
            """,
    ):
        st.markdown('<h2 class="neon-subhead">🌐 SELECT MARKET REGION</h2>', unsafe_allow_html=True)
        
        # Region coordinates and initial settings
        regions = {
            'USA': {'lat': 37.0902, 'lon': -95.7129, 'color': [0, 245, 212]},
            'Europe': {'lat': 54.5260, 'lon': 15.2551, 'color': [0, 207, 255]},
            'Asia': {'lat': 34.0479, 'lon': 100.6197, 'color': [191, 0, 255]},
            'Global': {'lat': 10.0, 'lon': 0.0, 'color': [255, 255, 255]}
        }
        
        # Create deck data
        deck_data = []
        for region, details in regions.items():
            deck_data.append({
                'name': region,
                'lat': details['lat'],
                'lon': details['lon'],
                'color': details['color'],
                'radius': 300000 if region == st.session_state['selected_region'] else 200000,
                'elevation': 100000 if region == st.session_state['selected_region'] else 50000
            })
            
        # Create the deck
        deck = pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v10',
            initial_view_state=pdk.ViewState(
                latitude=20,
                longitude=0,
                zoom=1,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=deck_data,
                    get_position='[lon, lat]',
                    get_color='color',
                    get_radius='radius',
                    pickable=True,
                    auto_highlight=True,
                ),
                pdk.Layer(
                    'TextLayer',
                    data=deck_data,
                    get_position='[lon, lat]',
                    get_text='name',
                    get_size=18,
                    get_color=[255, 255, 255, 200],
                    get_angle=0,
                    get_text_anchor='"middle"',
                    get_alignment_baseline='"center"',
                )
            ],
            tooltip={
                'html': '<div style="background: #0A0E17; padding: 10px; border: 1px solid #00F5D4; border-radius: 4px;">'
                        '<span style="color: #00F5D4; font-weight: bold;">{name}</span>'
                        '</div>',
                'style': {
                    'color': 'white'
                }
            }
        )
        
        # Handle clicks on the map
        deck_chart = st.pydeck_chart(deck)
        
        # Below the map, show buttons for quick region selection
        region_cols = st.columns(4)
        for idx, region in enumerate(['USA', 'Europe', 'Asia', 'Global']):
            if region_cols[idx].button(f"{region}", key=f"btn_{region}"):
                st.session_state['selected_region'] = region
                st.rerun()

        st.caption(f"Currently viewing: **{st.session_state['selected_region']}** market data")

# === ETF Selection in Column 2 ===
with col2:
    with stylable_container(
        key="etf_selector",
        css_styles="""
            {
                border: 1px solid #00F5D4;
                border-radius: 8px;
                padding: 20px;
                background-color: rgba(10, 14, 23, 0.7);
                box-shadow: 0 0 8px #00F5D4;
                height: 100%;
            }
            """,
    ):
        st.markdown('<h2 class="neon-subhead">📊 ETF SELECTION</h2>', unsafe_allow_html=True)
        
        # ETF Ticker Selection
        ticker = st.text_input("Enter Sector ETF Ticker:", 
                              value="XLK", 
                              help="Enter a valid ETF ticker like XLK, XLF, etc.")
        
        # Time Range Selection
        period_options = {
            "1mo": "Last Month",
            "3mo": "Last Quarter",
            "6mo": "Last 6 Months",
            "1y": "Last Year",
            "2y": "Last 2 Years",
            "5y": "Last 5 Years"
        }
        
        period = st.select_slider(
            "Time Horizon:",
            options=list(period_options.keys()),
            format_func=lambda x: period_options[x],
            value="1y"
        )
        
        # Fetch Button
        fetch_pressed = st.button("⚡ FETCH DATA", use_container_width=True)
        
        if fetch_pressed:
            # Show loading animation
            with st.spinner("Accessing financial matrix..."):
                time.sleep(0.5)  # Small delay for visual effect
                st.toast(f"Retrieving {ticker} data from the network", icon="🔌")

#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

# === ETF Data Display ===
# Only process data if fetch button was pressed or there's data in the session state
should_load_data = fetch_pressed or ('etf_data' in st.session_state and st.session_state['etf_data'] is not None)

if should_load_data:
    try:
        # Load data (use cached data if available and not forcing refresh)
        if fetch_pressed or 'etf_data' not in st.session_state:
            df = data_loader.load_yahoo_data(ticker, period=period)

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # === Side-by-side: Raw Data + Data Types ===
            preview_col1, preview_col2 = st.columns(2)

            with preview_col1:
                with stylable_container(
                    key="raw_data_preview",
                    css_styles="""
                        {
                            border: 1px solid #00CFFF;
                            border-radius: 10px;
                            background-color: #0A0E17;
                            box-shadow: 0 0 15px #00CFFF;
                            padding: 20px;
                        }
                        .dataframe {
                            background-color: #0E131F;
                            border: 1px solid #00CFFF;
                        }
                        .dataframe th {
                            background-color: #1A1F35;
                            color: #00F5D4;
                            padding: 8px;
                        }
                        .dataframe td {
                            padding: 8px;
                            border-bottom: 1px solid #1A1F35;
                        }
                    """
                ):
                    st.markdown("""
                        <div class="neon-subhead" style="font-size:18px; margin-bottom:10px;">
                            🔍 <span style='color:#00CFFF;'>Raw Data Preview</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(df.head(10), use_container_width=True)

            with preview_col2:
                with stylable_container(
                    key="data_types_display",
                    css_styles="""
                        {
                            border: 1px solid #BF00FF;
                            border-radius: 10px;
                            background-color: #0A0E17;
                            box-shadow: 0 0 15px #BF00FF;
                            padding: 20px;
                        }
                    """
                ):
                    st.markdown("""
                        <div class="neon-subhead" style="font-size:18px; margin-bottom:10px;">
                            ✅ <span style='color:#BF00FF;'>Data Types</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(
                        df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}),
                        use_container_width=True
                    )

            if df is not None and not df.empty:
                st.session_state['etf_data'] = df
                st.session_state['etf_ticker'] = ticker
                st.session_state['etf_period'] = period
                st.toast(f"Successfully loaded {len(df)} data points", icon="✅")
            else:
                st.error("Failed to retrieve data. Please check ticker or try again.")
                st.stop()
        else:
            df = st.session_state['etf_data']
            ticker = st.session_state['etf_ticker']
            period = st.session_state['etf_period']

        # Re-validate column format
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        df = df.dropna(subset=['Close'])

        # === Summary Metrics ===
        summary = {
            "Mean Return (%)": float(df['Close'].pct_change().mean() * 100),
            "Volatility (%)": float(df['Close'].pct_change().std() * 100),
            "Max Drawdown (%)": float((1 - df['Close'] / df['Close'].cummax()).max() * 100)
        }

        metric_cols = st.columns(3)
        metrics = [
            ("📊 Average Return", "Mean Return (%)", "#00F5D4"),
            ("📉 Volatility", "Volatility (%)", "#00CFFF"),
            ("📉 Max Drawdown", "Max Drawdown (%)", "#BF00FF")
        ]

        for col, (label, key, color) in zip(metric_cols, metrics):
            with col:
                with stylable_container(
                    key=f"metric_{key}",
                    css_styles=f"""
                        {{
                            border-left: 3px solid {color};
                            border-radius: 3px;
                            padding: 10px 15px;
                            background: linear-gradient(90deg, rgba(0,245,212,0.1) 0%, rgba(10,14,23,0) 100%);
                        }}
                    """
                ):
                    st.metric(label, value=f"{summary.get(key, 0):.2f}%", delta=None)

        # === Price Evolution Chart ===
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Price',
                                 line=dict(color='#00F5D4', width=2)))

        if len(df) >= 20:
            df['MA20'] = df['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='20-Day MA',
                                     line=dict(color='#00CFFF', width=1.5, dash='dot')))
        if len(df) >= 50:
            df['MA50'] = df['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], mode='lines', name='50-Day MA',
                                     line=dict(color='#BF00FF', width=1.5, dash='dot')))

        if len(df) >= 20:
            df['stddev'] = df['Close'].rolling(window=20).std()
            df['upper_band'] = df['MA20'] + (df['stddev'] * 2)
            df['lower_band'] = df['MA20'] - (df['stddev'] * 2)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['upper_band'], mode='lines',
                                     line=dict(color='rgba(0,207,255,0.3)', width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['lower_band'], mode='lines',
                                     line=dict(color='rgba(0,207,255,0.3)', width=0),
                                     fill='tonexty', fillcolor='rgba(0,207,255,0.05)', name='Bollinger Bands',
                                     showlegend=True))

        fig.update_layout(
            title="📈 Price Evolution",
            template="plotly_dark",
            plot_bgcolor="#0A0E17",
            paper_bgcolor="#0A0E17",
            font=dict(family="Rajdhani", color="white"),
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
        with st.expander("Details"):
            st.code(str(e))

