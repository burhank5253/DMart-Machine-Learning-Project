
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="DMart ML Predictor",
    page_icon="🛒",
    layout="wide"
)


# Global model directory
MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Models")
)



st.sidebar.title("🛒 DMart Navigation")

selected_page = st.sidebar.radio(
    "Select a section",
    [
        "Project Overview",
        "Order Value Prediction",
        "Cancellation Prediction",
        "Model Performance"
    ]
)

st.sidebar.divider()
st.sidebar.caption("DMart E-commerce ML Project")


if selected_page == "Project Overview":
    st.title("🛒 DMart E-commerce Analytics & ML")
    st.subheader("Machine Learning Prediction Dashboard")

    st.markdown("""
    Welcome to the DMart E-commerce Machine Learning Project.

    This project uses customer, product and shopping behaviour data
    to demonstrate machine learning applications in e-commerce.

    ### Project Highlights
    - Dataset: 25,000 e-commerce records
    - Regression: Predict customer order value
    - Classification: Experimental order cancellation prediction
    - Algorithms: Linear Regression, Ridge, Lasso, Random Forest,
      Decision Tree, SVM and XGBoost
    - Tools: Python, Pandas, NumPy, Scikit-learn, XGBoost and Streamlit

    ### Project Objective
    To explore how machine learning can support e-commerce analytics
    through order-value estimation, cancellation analysis and model evaluation.
    """)

    st.divider()

elif selected_page == "Order Value Prediction":
    st.title("🛒 DMart E-commerce Machine Learning App")
    st.write("Predict customer order value using machine learning.")

    model = joblib.load(
        os.path.join(MODEL_DIR, "tuned_random_forest_regression.pkl")
    )

    feature_columns = joblib.load(
        os.path.join(MODEL_DIR, "regression_feature_columns.pkl")
    )

    st.success("Regression model loaded successfully!")

    st.subheader("Enter Customer and Order Details")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            customer_age = st.number_input(
                "Customer Age", min_value=18, max_value=100, value=30
            )
            gender = st.selectbox("Gender", ["Male", "Female"])
            mrp = st.number_input("Product MRP", min_value=0.0, value=1000.0)
            category = st.selectbox(
                "Category", ["Local", "Branded", "Imported"]
            )
            subscription = st.selectbox(
                "Subscription", ["Yes", "No", "Premium", "Premium Plus"]
            )

        with col2:
            time_spent = st.number_input(
                "Time Spent on Website", min_value=0.0, value=10.0
            )
            rating = st.number_input(
                "Product Rating", min_value=0.0, max_value=5.0, value=4.0
            )
            marketing = st.selectbox(
                "Marketing Advertisement",
                ["Instagram", "Facebook", "Friends", "TV", "Other"]
            )
            ship_mode = st.selectbox(
                "Shipping Mode", ["Free", "Priority", "Express"]
            )
            clicks = st.number_input(
                "Number of Clicks", min_value=0, value=5
            )

        submitted = st.form_submit_button("Predict Order Value")

    if submitted:
        input_data = pd.DataFrame([{
            "Customer_Age": customer_age,
            "Gender": gender,
            "MRP": mrp,
            "Category": category,
            "Subscription": subscription,
            "Time_Spent_on_Website": time_spent,
            "Rating": rating,
            "Marketing_Advertisement": marketing,
            "Ship_Mode": ship_mode,
            "No_of_Clicks": clicks
        }])

        encoded_data = pd.get_dummies(
            input_data,
            columns=[
                "Gender", "Category", "Subscription",
                "Marketing_Advertisement", "Ship_Mode"
            ],
            drop_first=True,
            dtype=int
        )

        encoded_data = encoded_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = model.predict(encoded_data)[0]

        st.subheader("Prediction Result")
        st.metric("Predicted Order Value", f"₹{prediction:,.2f}")

        st.caption(
            "This is an experimental estimate based on the trained model. "
            "Actual order value may differ."
        )


    st.divider()

elif selected_page == "Cancellation Prediction":
    st.header("🚦 Order Cancellation Prediction")

    st.warning(
        "This is an experimental prediction. "
        "The cancellation model showed weak predictive performance "
        "during testing and should not be used for actual business decisions."
    )

    classification_model = joblib.load(
        os.path.join(MODEL_DIR, "logistic_regression.pkl")
    )

    classification_scaler = joblib.load(
        os.path.join(MODEL_DIR, "classification_scaler.pkl")
    )

    classification_columns = joblib.load(
        os.path.join(MODEL_DIR, "classification_feature_columns.pkl")
    )

    st.success("Cancellation model loaded successfully!")

    with st.form("cancellation_form"):
        st.write("Enter customer and product details")

        cancel_age = st.number_input(
            "Customer Age for Cancellation",
            min_value=18, max_value=100, value=30
        )
        cancel_gender = st.selectbox(
            "Gender for Cancellation", ["Male", "Female"]
        )
        cancel_mrp = st.number_input(
            "Product MRP for Cancellation",
            min_value=0.0, value=1000.0
        )
        cancel_category = st.selectbox(
            "Category for Cancellation", ["Local", "Branded", "Imported"]
        )
        cancel_subscription = st.selectbox(
            "Subscription for Cancellation",
            ["Yes", "No", "Premium", "Premium Plus"]
        )
        cancel_time = st.number_input(
            "Website Time for Cancellation", min_value=0.0, value=10.0
        )
        cancel_rating = st.number_input(
            "Product Rating for Cancellation",
            min_value=0.0, max_value=5.0, value=4.0
        )
        cancel_marketing = st.selectbox(
            "Marketing for Cancellation",
            ["Instagram", "Facebook", "Friends", "TV", "Other"]
        )
        cancel_ship = st.selectbox(
            "Shipping Mode for Cancellation",
            ["Free", "Priority", "Express"]
        )
        cancel_clicks = st.number_input(
            "Number of Clicks for Cancellation", min_value=0, value=5
        )

        cancel_submitted = st.form_submit_button(
            "Predict Cancellation"
        )

    if cancel_submitted:
        cancel_input = pd.DataFrame([{
            "Customer_Age": cancel_age,
            "Gender": cancel_gender,
            "MRP": cancel_mrp,
            "Category": cancel_category,
            "Subscription": cancel_subscription,
            "Time_Spent_on_Website": cancel_time,
            "Rating": cancel_rating,
            "Marketing_Advertisement": cancel_marketing,
            "Ship_Mode": cancel_ship,
            "No_of_Clicks": cancel_clicks
        }])

        cancel_encoded = pd.get_dummies(
            cancel_input,
            columns=[
                "Gender", "Category", "Subscription",
                "Marketing_Advertisement", "Ship_Mode"
            ],
            drop_first=True,
            dtype=int
        )

        cancel_encoded = cancel_encoded.reindex(
            columns=classification_columns,
            fill_value=0
        )

        cancel_scaled = classification_scaler.transform(cancel_encoded)
        cancel_prediction = classification_model.predict(cancel_scaled)[0]

        st.subheader("Cancellation Prediction Result")

        if cancel_prediction == 1:
            st.error("The model predicts: Possible Cancellation")
        else:
            st.success("The model predicts: Not Cancelled")

        st.caption(
            "This is an experimental model output, not a reliable "
            "confirmation of whether an order will be cancelled."
        )


    st.divider()

elif selected_page == "Model Performance":
    st.header("📊 Machine Learning Model Performance")

    st.subheader("Regression Model Comparison")

    regression_results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Polynomial Regression",
            "XGBoost",
            "Random Forest Tuned",
            "Random Forest Original"
        ],
        "MAE": [203.72, 203.72, 203.72, 204.32, 205.76, 207.10, 218.71],
        "RMSE": [324.11, 324.11, 324.11, 324.16, 327.01, 327.84, 336.45],
        "R2 Score": [0.5603, 0.5603, 0.5603, 0.5602, 0.5524, 0.5502, 0.5262]
    })

    st.dataframe(regression_results, use_container_width=True)

    st.info(
        "Best regression performance among tested models: "
        "Linear, Ridge and Lasso Regression, with R² approximately 0.5603."
    )

    st.subheader("Classification Model Comparison")

    classification_results = pd.DataFrame({
        "Model": [
            "SVM",
            "Decision Tree",
            "Logistic Regression",
            "XGBoost",
            "Random Forest"
        ],
        "Accuracy": [0.5792, 0.4076, 0.4956, 0.6672, 0.8378],
        "Precision": [0.0745, 0.0700, 0.0709, 0.0732, 0.0858],
        "Recall": [0.4286, 0.5938, 0.5014, 0.3137, 0.1317],
        "F1 Score": [0.1270, 0.1252, 0.1243, 0.1186, 0.1039],
        "ROC-AUC": [0.4986, 0.5028, 0.4861, 0.4936, 0.4926]
    })

    st.dataframe(classification_results, use_container_width=True)

    st.warning(
        "Classification models showed weak predictive performance. "
        "ROC-AUC values are close to 0.50, so cancellation predictions "
        "should be treated as experimental."
    )



    st.subheader("Regression Model R² Comparison")

    regression_chart = regression_results.set_index("Model")[["R2 Score"]]
    st.bar_chart(regression_chart)

    st.subheader("Classification Model F1 Score Comparison")

    classification_chart = classification_results.set_index("Model")[["F1 Score"]]
    st.bar_chart(classification_chart)
