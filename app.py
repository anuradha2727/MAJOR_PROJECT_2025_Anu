import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'random_forest_model_optimized.pkl'  # Path to your saved model
    return joblib.load(model_path)

rf_model = load_model()

# Define the prediction function
def predict(csv_file):
    # Load the uploaded CSV file
    real_time_data = pd.read_csv(csv_file)

    # Ensure the columns match the training data
    required_columns = rf_model.feature_names_in_  # Features used during training

    # Add missing columns with default values
    for col in required_columns:
        if col not in real_time_data.columns:
            real_time_data[col] = 0  # Add missing columns with default value

    # Drop extra columns
    real_time_data = real_time_data[required_columns]

    # Handle missing values
    real_time_data.fillna(0, inplace=True)

    # Normalize the data
    scaler = MinMaxScaler()
    real_time_data = pd.DataFrame(scaler.fit_transform(real_time_data), columns=required_columns)

    # Make predictions
    predictions = rf_model.predict(real_time_data)

    # Map predictions to labels
    real_time_data['Predictions'] = predictions
    real_time_data['Prediction_Label'] = real_time_data['Predictions'].map({0: 'Safe', 1: 'Intrusion Detected'})

    return real_time_data

# Streamlit UI
st.title("Intrusion Detection System")
st.write("Upload a CSV file to predict whether there is a threat or not.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Display the uploaded file
        st.write("Uploaded File:")
        st.write(f"File name: {uploaded_file.name}")
        real_time_data = pd.read_csv(uploaded_file)
        st.write(real_time_data.head())
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    # Predict and display results
    with st.spinner("Processing..."):
        try:
            results = predict(uploaded_file)
            st.success("Prediction completed!")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

    # Display prediction summary
    st.subheader("Prediction Summary")
    prediction_summary = results['Prediction_Label'].value_counts()
    st.write(prediction_summary)

    # Display detailed results
    st.subheader("Detailed Results")
    st.write(results)

    # Download results as CSV
    st.download_button(
        label="Download Predictions as CSV",
        data=results.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )