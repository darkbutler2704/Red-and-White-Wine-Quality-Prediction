import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

#Streamlit UI
st.title("🍷 Wine Quality Prediction")
wine_type = st.selectbox("Select Wine Type", ["Red", "White"])

#Load dataset based on user selection
if wine_type == "Red":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    st.subheader("🔴 Red Wine Selected")
else:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    st.subheader("⚪ White Wine Selected")

df = pd.read_csv(url, sep=';')
df = df.drop_duplicates()
assert df.isnull().sum().sum() == 0, "There are missing values!"

df["quality_label"] = df["quality"].apply(lambda q: "low" if q <= 5 else "medium" if q == 6 else "high")
features_to_scale = df.drop(columns=["quality", "quality_label"])
scaler = MinMaxScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(features_to_scale), columns=features_to_scale.columns)
scaled_df = pd.concat([scaled_features.reset_index(drop=True), df[["quality", "quality_label"]].reset_index(drop=True)], axis=1)
X = scaled_df.drop(columns=["quality", "quality_label"])
y = scaled_df["quality_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Unique model filenames for red and white wines
MODEL_FILE = f"wine_quality_model_{wine_type.lower()}.pkl"
SCALER_FILE = f"scaler_{wine_type.lower()}.pkl"
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
except FileNotFoundError:
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

if "model_evaluated" not in st.session_state or st.session_state.get("last_type") != wine_type:
    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report_dict["accuracy"]
    report_dict["accuracy"] = {
        "precision": accuracy,
        "recall": accuracy,
        "f1-score": accuracy,
        "support": sum(report_dict[label]["support"] for label in ["high", "medium", "low"])
    }
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    st.write("📋 Classification Report:")
    st.code(tabulate(report_df, headers="keys", tablefmt="github"))

    st.write("📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=["low", "medium", "high"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low", "medium", "high"])
    disp.plot(cmap='Blues')
    st.pyplot(plt)
    st.session_state.model_evaluated = True
    st.session_state.last_type = wine_type

# Input sliders for prediction
if wine_type == "Red":
    st.write("Enter red wine's physicochemical properties to predict its quality level (Low, Medium, High).")
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.slider("Residual Sugar", 0.9, 15.5, 2.5)
    chlorides = st.slider("Chlorides", 0.01, 0.2, 0.05)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 15)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 46)
    density = st.slider("Density", 0.9900, 1.0040, 0.9968)
    pH = st.slider("pH", 2.5, 4.5, 3.3)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6)
    alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)
else:
    st.write("Enter white wine's physicochemical properties to predict its quality level (Low, Medium, High).")
    fixed_acidity = st.slider("Fixed Acidity", 3.0, 15.0, 6.8)               
    volatile_acidity = st.slider("Volatile Acidity", 0.08, 1.1, 0.27)        
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.33)                  
    residual_sugar = st.slider("Residual Sugar", 0.6, 65.0, 5.7)           
    chlorides = st.slider("Chlorides", 0.009, 0.35, 0.043)                  
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 289, 30)      
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 440, 115)   
    density = st.slider("Density", 0.9870, 1.0040, 0.9940)                 
    pH = st.slider("pH", 2.8, 4.2, 3.18)                                     
    sulphates = st.slider("Sulphates", 0.2, 1.1, 0.49)                       
    alcohol = st.slider("Alcohol", 8.0, 14.9, 10.5)                         

if st.button("Predict Quality"):
    user_input = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    classes = model.classes_

    st.success(f"Predicted Wine Quality Category: **{prediction.upper()}**")
    st.write("🔍 Prediction Confidence:")
    prob_df = pd.DataFrame({
        "Quality": classes,
        "Probability": [f"{p*100:.2f}%" for p in probabilities]
    })
    st.table(prob_df)

    filtered_df = scaled_df[scaled_df["quality_label"] == prediction]

    st.markdown(f"### 📊 Characteristics of **{prediction.upper()}** Quality Wines")
    st.subheader("🔍 Distribution of Features")
    features_only = filtered_df.drop(columns=["quality", "quality_label"])
    num_features = features_only.shape[1]

    fig, axes = plt.subplots(nrows=(num_features + 2) // 3, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(features_only.columns):
        axes[i].hist(features_only[col], bins=15, color='skyblue', edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Feature Distributions for {prediction.upper()} Quality Wines", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("🍷 Alcohol vs. Wine Quality")
    fig, ax = plt.subplots()
    sns.scatterplot(data=scaled_df, x="alcohol", y="quality", hue="quality_label", palette="Set2", ax=ax)
    ax.axhline(y=filtered_df["quality"].mean(), color='red', linestyle='--', label=f'{prediction.upper()} Avg Quality')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = filtered_df.drop(columns=["quality_label"]).corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.title(f"{prediction.upper()} Quality Wine Correlation Heatmap")
    st.pyplot(fig)

    st.markdown("### 🧠 Reason Behind the Prediction of Wine Quality")
    if wine_type == "Red":
        if prediction == "high":
            st.info("""
            - **Low Volatile Acidity** and **High Alcohol** often boost flavor quality.
            - **Balanced Acidity and Sulphates** support structure and microbial stability.
            - Overall, a **clean and well-balanced profile**.
            """)
        elif prediction == "low":
            st.warning("""
            - **High Volatile Acidity** and **Low Alcohol** suggest poor aroma and body.
            - **Unbalanced mineral content** may lead to flat or undesirable taste.
            - Profile often seen in **lower quality wines**.
            """)
        elif prediction == "medium":
            st.info("""
            - **Moderate levels across most features**, suggesting drinkable but unremarkable wine.
            - No extremes in acidity, sugar, or alcohol.
            - Typical of **average quality wines**.
            """)
    else:  # White wine
        if prediction == "high":
            st.info("""
            - **High Citric Acid** and **Low Volatile Acidity** enhance freshness and complexity.
            - **Balanced Sugar and Acidity** improves mouthfeel and aging potential.
            - **Higher Alcohol** often correlates with better ripeness and flavor integration.
            - Indicates a **crisp, clean, and aromatic profile**.
            """)
        elif prediction == "low":
            st.warning("""
            - **High Volatile Acidity** and **Low Citric Acid** reduce freshness and introduce off-notes.
            - **Imbalanced Acidity and Residual Sugar** can make the wine taste flat or cloying.
            - **Low Alcohol** often results in a thin body and underdeveloped flavors.
            - Suggests a **poorly balanced, less refined white wine**.
            """)
        elif prediction == "medium":
            st.info("""
            - **Moderate Citric Acid** and **Alcohol** levels create a decent but not exceptional wine.
            - **Acceptable sugar-acid balance**, though without distinct standout features.
            - Often lacks the aromatic complexity or texture of high-quality whites.
            - Reflects a **standard, easy-to-drink wine** with no major flaws.
            """)

