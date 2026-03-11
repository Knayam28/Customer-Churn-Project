import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. Page Setup
st.set_page_config(page_title="Churn Predictor Pro", page_icon="📈", layout="wide")

# Custom CSS to make the header look modern
st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 30px;'>
        <h1 style='color: #FFFFFF; margin:0;'>🛒 E-Commerce Churn Dashboard</h1>
        <p style='color: #AAAAAA; font-size: 18px; margin:0;'>Predict customer retention and generate targeted rescue strategies.</p>
    </div>
""", unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('xgb_churn_pipeline.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'xgb_churn_pipeline.pkl' not found.")
    st.stop()

# 3. Main Body UI
st.subheader("👤 Demographics & Profile")
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
with col2:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=60, value=1)
    warehouse_to_home = st.number_input("Warehouse To Home Distance", min_value=5, max_value=127, value=15)
with col3:
    complain = st.selectbox("Has the customer complained?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

st.subheader("📱 App & Engagement Metrics")
col4, col5, col6 = st.columns(3)
with col4:
    preferred_login_device = st.selectbox("Preferred Login Device", options=["Mobile Phone", "Computer"])
with col5:
    hour_spend_on_app = st.number_input("Hours Spent On App", min_value=0, max_value=10, value=2)
with col6:
    coupon_used = st.number_input("Coupons Used", min_value=0, max_value=20, value=1)

st.divider()

st.subheader("🛍️ Order & Payment History")
col7, col8, col9 = st.columns(3)
with col7:
    order_count = st.number_input("Total Order Count", min_value=1, max_value=50, value=2)
    prefered_order_cat = st.selectbox("Preferred Order Category", options=["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
with col8:
    day_since_last_order = st.number_input("Day Since Last Order", min_value=0, max_value=30, value=2)
    order_amount_hike = st.number_input("Order Amount Hike From Last Year (%)", min_value=11, max_value=26, value=15)
with col9:
    cashback_amount = st.number_input("Cashback Amount ($)", min_value=0, max_value=500, value=120)
    preferred_payment_mode = st.selectbox("Preferred Payment Mode", options=["Debit Card", "Credit Card", "E-Wallet", "UPI", "Cash on Delivery"])

st.markdown("<br>", unsafe_allow_html=True)

# 4. Data Formatting
input_data = pd.DataFrame({
    'Tenure': [tenure], 'DaySinceLastOrder': [day_since_last_order], 'CashbackAmount': [cashback_amount],
    'OrderCount': [order_count], 'Complain': [complain], 'WarehouseToHome': [warehouse_to_home],
    'CouponUsed': [coupon_used], 'HourSpendOnApp': [hour_spend_on_app], 'OrderAmountHikeFromlastYear': [order_amount_hike],
    'Gender': [gender], 'MaritalStatus': [marital_status], 'PreferredLoginDevice': [preferred_login_device],
    'PreferedOrderCat': [prefered_order_cat], 'PreferredPaymentMode': [preferred_payment_mode]
})

# 5. Prediction with styled button
_, center_button, _ = st.columns([1, 1, 1])
with center_button:
    # Adding type="primary" makes the button solid and stand out
    predict_clicked = st.button("🔍 Generate Churn Report", type="primary", use_container_width=True)

if predict_clicked:
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100
        
        st.divider()
        st.subheader("📊 Customer Risk Analysis")
        
        with st.container(border=True):
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                # The Gauge Chart 
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability,
                    number = {'suffix': "%", 'font': {'size': 40}},
                    title = {'text': "Probability of Churning", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "rgba(0,0,0,0)"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': "#00CC96"},
                            {'range': [40, 70], 'color': "#FFA15A"},
                            {'range': [70, 100], 'color': "#EF553B"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': probability
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with res_col2:
                st.markdown("<br><br><br>", unsafe_allow_html=True) # Aligns text nicely with the gauge
                if prediction == 1:
                    st.error("### 🚨 HIGH FLIGHT RISK")
                    st.write("The XGBoost model has flagged this user's behavioral metrics as highly indicative of churn.")
                else:
                    st.success("### ✅ LOYALTY SECURE")
                    st.write("This customer is currently exhibiting stable behavior with a low probability of abandonment.")
                    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")