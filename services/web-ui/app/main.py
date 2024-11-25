import streamlit as st
import requests
from helpers import build_prediction_request_body, save_inference_results_to_db, \
    list_parquet_files, get_registered_models_info, load_model, load_batch_df
import logging
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
import scikitplot as skplt
import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.metrics import RocCurveDisplay
from sklearn.calibration import CalibrationDisplay
from matplotlib.gridspec import GridSpec

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

MLFLOW_HEALTH_ENDPOINT_URL = "http://prediction-service:4242/health"
PREDICTION_ENDPOINT_URL = "http://prediction-service:4242/prediction"
# Define the data directory
DATA_DIR = "data"
# Batch path for inference prediction POST
batch_path="./data/features_store/batch.parquet"
### Dashboard ###
# Tile
st.title("Project: :red[Churn Prediction]")
# Md
st.markdown("## Model Performance")
# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Test API", "Data", "Global Performance", "Local Performance"])
with tab1:
    # Note! This tab for testing API purpose.
    ### Get MLflow health ###
    st.write("Get Mlflow health from http://localhost:4242/health")
    if st.button("Get MLflow health"):
        with st.status("Getting..", expanded=True) as status:
            st.write("Get MLflow health...")
            res = requests.get(MLFLOW_HEALTH_ENDPOINT_URL)
            res = res.json()
            st.write("The response:", res)

    st.divider()

    ### Post prediction ###
    if 'predition_response' not in st.session_state:
        st.session_state.predition_response = []

    st.write("Post prediction using a production model")
    if st.button("Predict"):
        with st.status("Predicting..", expanded=True) as status:
            req_body = build_prediction_request_body(batch_path=batch_path)
            st.write("Posting to the prediction service...")
            
            resp = requests.post(PREDICTION_ENDPOINT_URL, json=req_body)

            st.success("Success posting to the prediction service")
            st.json(resp.json())
            status.update(
                label="Predicting completed!", state="complete", expanded=False
            )
            st.session_state.predition_response = resp.json()
        st.write("The response:", resp.json())

    st.divider()

    ### Save prediction results to the database ###
    st.write("Save prediction results to the database")
    if st.button("Save the results"):
        st.write(f"Save from {st.session_state.predition_response}")
        with st.status("Saving..", expanded=True) as status:
            ret = save_inference_results_to_db(st.session_state.predition_response)
            st.write(f"Return from save function: {ret}")
            status.update(
                label="Save completed!", state="complete", expanded=False
            )

with tab2:
    ### Load data ###
    # Get list of parquet files
    parquet_files = list_parquet_files(DATA_DIR)
    
    # Sidebar for file selection
    st.sidebar.header("Select Dataset")
    dataset_choice = st.sidebar.selectbox("Choose a dataset to load:", list(parquet_files.keys()))

    # Load selected dataset
    if dataset_choice:
        file_path = parquet_files[dataset_choice]
        try:
            df = pd.read_parquet(file_path)
            st.write(f"### Dataset: {dataset_choice}")
            st.write(df)
        except Exception as e:
            st.error(f"Error loading {dataset_choice}: {e}")

with tab3:
    # Display model metadata in Streamlit
    st.title("MLflow Model Information")

    # Fetch model metadata
    models_info, models = get_registered_models_info()

    # Display the metadata
    if models_info:
        for model_info in models_info:
            st.subheader(f"Model: {model_info['Model Name']} (Version {model_info['Version']})")
            st.write(f"**Tag**: {model_info['Tags']}")
            st.write(f"**Run ID**: {model_info['Run ID']}")
            col1, col2 = st.columns(2)
            with col1:
                # Display metrics
                st.write("### Metrics")
                st.write(pd.DataFrame(model_info["Metrics"].items(), columns=["Metric", "Value"]))
            with col2:
                # Display parameters
                st.write("### Parameters")
                st.write(pd.DataFrame(model_info["Parameters"].items(), columns=["Parameter", "Value"]))
    else:
        st.write("No registered models found.")

    st.divider()
    ### Model Prediction ###
    # Target feature
    target = "Attrition_Flag"
    # Load train, test data
    train_df = load_batch_df(batch_path="./data/features_store/train.parquet")
    test_df = load_batch_df(batch_path="./data/features_store/test.parquet")
    feature_names = test_df.drop(target, axis=1).columns
    X_train = train_df.drop(target, axis=1)
    X_test = test_df.drop(target, axis=1)
    # y_pred = np.array([])
    y_test = test_df[target]
        
    # Load model
    model, tag_value = load_model()

    df_prediction_result = {}
    try:
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Create prediction df
        predictions = {"actual": y_test, "prediction": y_pred}
        df_prediction_result = pd.DataFrame(predictions)
        logging.info(f"Dataframe prediction: {df.head(10)}")
        y_test = df_prediction_result["actual"]
        y_pred = df_prediction_result["prediction"]

        # Can process y_test, y_pred by getting from db

        col1, col2 = st.columns(2)
        with col1:
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            plt.figure(figsize=(10, 6))
            skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
            st.pyplot(plt)
        with col2:
            # Feature Importances
            logging.info(f"tag_value: {tag_value}")
            logging.info(f"model {model}")            
            logging.info(f"feature names: {np.array(feature_names)}")
            
            if tag_value == "gradient-boosting":            
                st.subheader("Feature Importances")
                plt.figure(figsize=(10, 6))
                skplt.estimators.plot_feature_importances(model, feature_names=feature_names, x_tick_rotation=90)
                st.pyplot(plt)
            elif tag_value == "logistic-regression":
                st.subheader("Feature Importances via Coefficients")
                importance = np.abs(model.coef_)[0] 
                logging.info(f"coef: {model.coef_[0]}")
                # Sort importance and corresponding feature names in descending order
                sorted_indices = np.argsort(importance)
                sorted_importance = importance[sorted_indices]
                sorted_features = np.array(feature_names)[sorted_indices]
                logging.info(f"quicksort: {np.argsort(importance)}")
             
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(y=sorted_features, width=sorted_importance, color='skyblue') 
                ax.set_title("Feature Importances via Coefficients")
                ax.set_xlabel("Features")
                ax.set_ylabel("Importance")
                ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability
                # Display the plot in Streamlit
                st.pyplot(fig)
            st.write(f"Tag value: {tag_value}")
        st.divider()

        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        # Display the DataFrame with Streamlit's dataframe widget
        st.dataframe(
            report_df,
            use_container_width=True  # Ensures the table fits the Streamlit container width
        )

        st.divider()
        # ROC plot
        st.title("ROC Curve Viewer")
        st.subheader("Compare ROC Curves of Models")
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis for plotting
        for model in models:
            logging.info(f"model_info roc_auc {model_info['Metrics']}")
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, alpha=0.8, pos_label=1)

        ax.set_title("ROC Curve Comparison")
        st.pyplot(fig)  # Display the figure in Streamlit

        st.divider()
        # Calibration plot

        st.title("Calibration Viewer")
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(4, 2)
        colors = plt.get_cmap("Dark2")
        ax_calibration_curve = fig.add_subplot(gs[:2, :2])
        calibration_displays = {}
        for i, model in enumerate(models):
            name = "Tag-" + models_info[i]['Tags']['model'] + "-Verion-" + models_info[i]['Version']
            logging.info(f"index i = : {i}")
            logging.info(f"Calibration chart name: {name}")
            logging.info(f"Model: {model}")
            display = CalibrationDisplay.from_estimator(
                model,
                X_test,
                y_test,
                n_bins=10,
                name=name,
                ax=ax_calibration_curve,
                color=colors(i),
            )
            calibration_displays[name] = display
        ax_calibration_curve.grid()
        ax_calibration_curve.set_title("Calibration plots")
        
        # Add histogram
        grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
        for i, model in enumerate(models):
            name = "Tag-" + models_info[i]['Tags']['model'] + "-Verion-" + models_info[i]['Version']
            row, col = grid_positions[i]
            ax = fig.add_subplot(gs[row, col])

            ax.hist(
                calibration_displays[name].y_prob,
                range=(0, 1),
                bins=10,
                label=name,
                color=colors(i),
            )
            ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"{e}")
with tab4:
    sliders = []
    col1, col2 = st.columns(2)
    with col1:
        for name in feature_names:
            input_slider = st.slider(label=name, min_value=float(test_df[name].min()), max_value=float(test_df[name].max()))
            sliders.append(input_slider)

    with col2:
        col1, col2 = st.columns(2, gap="medium")
        
        prediction = model.predict([sliders])
        with col1:
            st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(target), unsafe_allow_html=True)

        probs = model.predict_proba([sliders])
        probability = probs[0][prediction[0]]

        with col2:
            st.metric(label="Model Confidence", value="{:.2f} %".format(probability*100), delta="{:.2f} %".format((probability-0.5)*100))

        X_train = train_df.drop(target, axis=1)
        explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, mode="classification", feature_names=feature_names, random_state=500, discretize_continuous=False, class_names=[0, 1])
        explanation = explainer.explain_instance(np.array(sliders), model.predict_proba, num_features=len(feature_names))
        interpretation_fig = explanation.as_pyplot_figure()
        st.pyplot(interpretation_fig, use_container_width=True)
