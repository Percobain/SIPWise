import numpy as np
import pandas as pd
import gradio as gr
import joblib
import plotly.graph_objects as go
import os
from huggingface_hub import hf_hub_download

# Define model file paths
MODEL_DATA_PATH = 'investment_predictor_data.joblib'
MODEL_PATH = 'investment_predictor_model.joblib'

# Check if model files exist locally, if not download them from Hugging Face Hub
def download_model_files():
    try:
        # Try to load the model data file
        if not os.path.exists(MODEL_DATA_PATH):
            print(f"Downloading {MODEL_DATA_PATH}...")
            hf_hub_download(repo_id=os.environ.get('SPACE_ID', 'percobain/SIPWise'), 
                           filename=MODEL_DATA_PATH, 
                           local_dir='.')
        
        if not os.path.exists(MODEL_PATH):
            print(f"Downloading {MODEL_PATH}...")
            hf_hub_download(repo_id=os.environ.get('SPACE_ID', 'percobain/SIPWise'), 
                           filename=MODEL_PATH, 
                           local_dir='.')
            
        print("Model files loaded successfully!")
    except Exception as e:
        print(f"Error downloading model files: {e}")
        raise

# Download model files if needed
download_model_files()

# Load the model and data
model_data = joblib.load(MODEL_DATA_PATH)
model = model_data['model']
risk_profiles = model_data['risk_profiles']
risk_profiles_list = model_data['risk_profiles_list']
profile_returns = model_data['profile_returns']
profile_volatility = model_data['profile_volatility']

def predict_monthly_sip(goal_amount, duration_years, risk_profile):
    """
    Predict monthly SIP required to reach a goal amount
    """
    # Encode risk profile
    risk_profile_encoded = risk_profiles_list.index(risk_profile)
    
    # Create input features
    X = np.array([[goal_amount, duration_years, risk_profile_encoded]])
    
    # Predict monthly SIP
    monthly_sip = model.predict(X)[0]
    
    return monthly_sip

def simulate_growth(monthly_sip, duration_years, risk_profile):
    """
    Simulate investment growth for visualization
    """
    # Get allocation for the risk profile
    allocation = risk_profiles[risk_profile]
    
    # Get expected annual return for the risk profile
    expected_return = profile_returns[risk_profile]
    
    # Number of months
    num_months = int(duration_years * 12)
    
    # Initialize arrays
    months = np.arange(num_months + 1)
    portfolio_values = np.zeros(num_months + 1)
    
    # Simulate month-by-month growth
    for month in range(1, num_months + 1):
        # Add monthly SIP
        portfolio_values[month] = portfolio_values[month - 1] + monthly_sip
        
        # Apply monthly return
        portfolio_values[month] *= (1 + expected_return / 12)
    
    return months, portfolio_values

def create_growth_chart(months, portfolio_values):
    """Create growth chart using Plotly"""
    fig = go.Figure()
    
    # Add investment amount line
    investment_amount = np.arange(len(months)) * portfolio_values[1]
    fig.add_trace(go.Scatter(
        x=months / 12,  # Convert to years
        y=investment_amount,
        mode='lines',
        name='Investment Amount',
        line=dict(color='blue', dash='dash')
    ))
    
    # Add portfolio value line
    fig.add_trace(go.Scatter(
        x=months / 12,  # Convert to years
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='green')
    ))
    
    # Update layout to be more compact
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        title='Growth Projection',
        xaxis_title='Years',
        yaxis_title='Amount (₹)',
        legend=dict(x=0.01, y=0.99, orientation="h"),
        template='plotly_white'
    )
    
    return fig

def create_allocation_chart(risk_profile):
    """Create allocation pie chart using Plotly"""
    # Get allocation for the risk profile
    allocation = risk_profiles[risk_profile]
    
    # Create labels and values
    labels = list(allocation.keys())
    values = list(allocation.values())
    
    # Filter out zero allocations
    non_zero_indices = [i for i, v in enumerate(values) if v > 0]
    labels = [labels[i] for i in non_zero_indices]
    values = [values[i] for i in non_zero_indices]
    
    # Create colors
    colors = ['#FFA500', '#FFD700', '#4682B4', '#6495ED']
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=colors)
    )])
    
    # Update layout to be more compact
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        title=f'Asset Allocation - {risk_profile}',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    return fig

def gradio_interface(goal_amount, duration_years, risk_profile):
    """
    Gradio interface function
    """
    # Predict monthly SIP
    monthly_sip = predict_monthly_sip(goal_amount, duration_years, risk_profile)
    
    # Round up to nearest 100
    monthly_sip = np.ceil(monthly_sip / 100) * 100
    
    # Simulate growth
    months, portfolio_values = simulate_growth(monthly_sip, duration_years, risk_profile)
    
    # Create growth chart
    growth_chart = create_growth_chart(months, portfolio_values)
    
    # Create allocation chart
    allocation_chart = create_allocation_chart(risk_profile)
    
    # Calculate total investment and expected returns
    total_investment = monthly_sip * duration_years * 12
    expected_final_value = portfolio_values[-1]
    expected_returns = expected_final_value - total_investment
    
    # Create compact summary
    summary_html = f"""
    <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: space-between; font-size: 0.9em;">
        <div style="flex: 1; min-width: 150px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
            <h4 style="margin-top: 0;">Monthly SIP</h4>
            <p style="font-size: 1.2em; font-weight: bold;">₹{monthly_sip:,.0f}</p>
        </div>
        <div style="flex: 1; min-width: 150px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
            <h4 style="margin-top: 0;">Total Investment</h4>
            <p>₹{total_investment:,.0f}</p>
        </div>
        <div style="flex: 1; min-width: 150px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
            <h4 style="margin-top: 0;">Expected Value</h4>
            <p>₹{expected_final_value:,.0f}</p>
        </div>
        <div style="flex: 1; min-width: 150px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
            <h4 style="margin-top: 0;">Returns</h4>
            <p>₹{expected_returns:,.0f}</p>
        </div>
        <div style="flex: 1; min-width: 150px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
            <h4 style="margin-top: 0;">CAGR</h4>
            <p>{profile_returns[risk_profile]:.2%}</p>
        </div>
        <div style="flex: 1; min-width: 150px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
            <h4 style="margin-top: 0;">Volatility</h4>
            <p>{profile_volatility[risk_profile]:.2%}</p>
        </div>
    </div>
    """
    
    return monthly_sip, growth_chart, allocation_chart, summary_html

# Set custom theme for better appearance
theme = gr.themes.Default().set(
    body_background_fill="#f0f0f0",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e0e0e0",
    block_radius="8px",
    block_shadow="0px 4px 6px rgba(0, 0, 0, 0.1)",
    button_primary_background_fill="#4682B4",
    button_primary_background_fill_hover="#5c9bd1",
    button_primary_text_color="#ffffff",
    input_background_fill="#f9f9f9",
    input_border_color="#d0d0d0",
    input_radius="4px"
    # Removed spacing_md and spacing_lg which were causing the error
)

# Create Gradio interface with a more compact layout
with gr.Blocks(theme=theme, css="""
    .gradio-container {max-width: 900px; margin: 0 auto;}
    .contain {display: flex; flex-direction: column; gap: 10px;}
    .top-row {display: flex; gap: 20px;}
    .input-panel {flex: 1; padding: 15px;}
    .output-panel {flex: 3;}
    .charts-row {display: flex; gap: 20px;}
    .chart-container {flex: 1;}
    footer {display: none !important;}
    .gradio-container {min-height: 0px !important;}
    """) as demo:
    
    gr.Markdown("# 💰 Goal-Based SIP Investment Predictor")
    gr.Markdown("Enter your financial goal, timeframe, and risk profile to get a recommended monthly SIP amount.")
    
    with gr.Row():
        with gr.Column(scale=1):
            goal_amount = gr.Number(label="Goal Amount (₹)", value=1000000, minimum=10000, maximum=10000000)
            duration_years = gr.Slider(minimum=1, maximum=30, step=1, label="Duration (years)", value=10)
            risk_profile = gr.Radio(choices=risk_profiles_list, label="Risk Profile", value="Balanced")
            submit_btn = gr.Button("Calculate SIP", variant="primary")
        
        with gr.Column(scale=2):
            monthly_sip_output = gr.Number(label="Recommended Monthly SIP (₹)")
            summary_output = gr.HTML()
    
    with gr.Row():
        growth_chart_output = gr.Plot(label="Growth Projection")
        allocation_chart_output = gr.Plot(label="Asset Allocation")
    
    submit_btn.click(
        fn=gradio_interface,
        inputs=[goal_amount, duration_years, risk_profile],
        outputs=[monthly_sip_output, growth_chart_output, allocation_chart_output, summary_output]
    )
    
    # Set examples for quick testing
    gr.Examples(
        examples=[
            [1000000, 10, "Conservative"],
            [2000000, 15, "Balanced"],
            [5000000, 20, "Aggressive"],
        ],
        inputs=[goal_amount, duration_years, risk_profile],
        outputs=[monthly_sip_output, growth_chart_output, allocation_chart_output, summary_output],
        fn=gradio_interface,
        cache_examples=True,
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()