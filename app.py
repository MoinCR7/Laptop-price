import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import traceback

# Load Model & Preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = tf.keras.models.load_model("laptop_price_model.h5")
        preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)

model, preprocessor, load_error = load_model_and_preprocessor()

# Modified predict function
def predict_price(input_data):
    try:
        # Convert the dictionary to a DataFrame with a single row
        input_df = pd.DataFrame([input_data])
        
        # Try standard transform
        try:
            processed_input = preprocessor.transform(input_df)
        except AttributeError:
            # Some preprocessors use transform_one
            if hasattr(preprocessor, 'transform_one'):
                processed_input = preprocessor.transform_one(input_data)
            else:
                raise
        
        # Make prediction
        return model.predict(processed_input)[0][0], None
    
    except Exception as e:
        error_msg = f"Preprocessing error: {str(e)}"
        
        # Fallback approach: Try with numeric features only
        try:
            numeric_input = np.array([[
                input_data['Inches'], 
                input_data['Ram'], 
                input_data['Memory_Space'],
                input_data['Weight'],
                input_data['Res_1'],
                input_data['Res_2'],
                input_data['Screen_Weights'],
                input_data['Clock_Speed'],
                input_data['SSD_Status'],
                input_data['Cpu_gen']
            ]], dtype=float)
            
            return model.predict(numeric_input)[0][0], "Using numeric features only"
        except Exception as e2:
            return None, f"{error_msg}\nSecond error: {str(e2)}"

# Streamlit UI
st.title("Laptop Price Prediction 💻")

if load_error:
    st.error(f"Error loading model or preprocessor: {load_error}")
    st.stop()

# Define Inputs
companies = ['Apple', 'HP', 'Dell', 'Lenovo', 'Acer', 'Asus']
typenames = ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible']
cpus = ['Intel', 'AMD']
gpus = ['Intel HD Graphics', 'Nvidia GTX', 'AMD Radeon']
os_options = ['Windows 10', 'macOS', 'Linux']
ssd_status = [0, 1]
cpu_gen = [5, 7, 9]

# Create two columns for better UI layout
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", companies)
    typename = st.selectbox("Type Name", typenames)
    cpu = st.selectbox("CPU", cpus)
    gpu = st.selectbox("GPU", gpus)
    os_choice = st.selectbox("Operating System", os_options)
    ssd = st.selectbox("SSD (1 for Yes, 0 for No)", ssd_status)
    cpu_gen_input = st.selectbox("CPU Generation", cpu_gen)

with col2:
    inches = st.slider("Screen Size (Inches)", 10.0, 20.0, 15.0, step=0.1)
    ram = st.slider("RAM (GB)", 2, 64, 8, step=2)
    memory_space = st.slider("Storage Space (GB)", 128, 2048, 512, step=128)
    weight = st.slider("Weight (kg)", 0.5, 5.0, 2.0, step=0.1)
    res_1 = st.selectbox("Resolution Width", [1366, 1440, 1920, 2560, 2880])
    res_2 = st.selectbox("Resolution Height", [768, 900, 1080, 1600, 1800])
    screen_weight = st.slider("Screen Weight", 0.5, 2.0, 1.0, step=0.1)
    clock_speed = st.slider("Clock Speed (GHz)", 1.0, 4.0, 2.5, step=0.1)

# Create a button to trigger prediction
if st.button("Predict Price", type="primary"):
    # Prepare input data as a dictionary
    input_data = {
        'Company': company,
        'TypeName': typename, 
        'Cpu': cpu,
        'Gpu': gpu,
        'OpSys': os_choice,
        'SSD_Status': ssd,
        'Cpu_gen': cpu_gen_input,
        'Inches': inches,
        'Ram': ram,
        'Memory_Space': memory_space,
        'Weight': weight, 
        'Res_1': res_1,
        'Res_2': res_2,
        'Screen_Weights': screen_weight,
        'Clock_Speed': clock_speed
    }
    
    # Debug section
    with st.expander("Input Data (Debug)"):
        st.write(input_data)
    
    # Make prediction
    predicted_price, warning = predict_price(input_data)
    
    if predicted_price is not None:
        # Show prediction with nice formatting
        st.success(f"### Predicted Laptop Price: ${predicted_price:.2f}")
        
        if warning:
            st.warning(f"Note: {warning}")
    else:
        st.error(f"Failed to make prediction: {warning}")
        
        # Show detailed instructions for fixing the error
        st.info("""
        ### How to fix "string indices must be integers" error:
        
        This likely means your preprocessor is trying to access string data incorrectly. Try:
        
        1. Re-examine how your preprocessor was created. It should be compatible with pandas DataFrames.
        2. Check if you're using ColumnTransformer or Pipeline from sklearn properly.
        3. Make sure categorical features are being handled with the right transformer.
        
        You may need to recreate your preprocessor with the correct configuration.
        """)