import dash
import dash.dcc as dcc
import dash.html as html
import requests
import pandas as pd
import plotly.express as px
import threading

# ✅ Initialize Dash App
app = dash.Dash(__name__)

# ✅ Fetch Fraud Data from Flask API
def get_fraud_summary():
    try:
        response = requests.get("http://127.0.0.1:5000/fraud_summary")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching fraud data: {e}")
    return {}

# ✅ Layout for Dashboard
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={"textAlign": "center"}),

    # Fraud Summary
    html.Div(id="fraud-summary", style={"textAlign": "center", "fontSize": 20, "marginBottom": 30}),

    # Fraud Trend Over Time
    dcc.Graph(id="fraud-trend"),
])

# ✅ Update Dashboard Content
@app.callback(
    [dash.dependencies.Output("fraud-summary", "children"),
     dash.dependencies.Output("fraud-trend", "figure")],
    [dash.dependencies.Input("fraud-summary", "id")]
)
def update_dashboard(_):
    fraud_data = get_fraud_summary()
    if not fraud_data:
        return "Error loading data", {}

    # Summary Text
    summary_text = f"Total Transactions: {fraud_data['total_transactions']} | Fraud Cases: {fraud_data['fraud_cases']} | Fraud %: {fraud_data['fraud_percentage']}%"

    # Load fraud dataset for trend analysis
    try:
        df = pd.read_csv("C:\\Users\\Hasan\\Desktop\\EDA\\Week_8-9_challenge_document-2\\scripts\\Fraud_Data_Final.csv")
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        fraud_trend = df.groupby(df["purchase_time"].dt.date)["class"].sum().reset_index()
        fraud_trend_fig = px.line(fraud_trend, x="purchase_time", y="class", title="Fraud Cases Over Time")
    except Exception as e:
        print(f"Error loading fraud dataset: {e}")
        return summary_text, {}

    return summary_text, fraud_trend_fig

# ✅ Function to run Dash without blocking Jupyter
def run_dash():
    app.run_server(debug=True, port=5001, use_reloader=False)

# ✅ Run Dash in a separate thread so Jupyter remains interactive
dash_thread = threading.Thread(target=run_dash)
dash_thread.start()
