import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Student Placement Dashboard",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================
# DATASET LOADING
# =====================================

DATASET_PATHS = [
    os.path.join(BASE_DIR, "Dataset", "cleaned_placement.csv"),
    os.path.join(BASE_DIR, "dataset", "placement.csv"),
    os.path.join(BASE_DIR, "dataset", "cleaned_placement.csv"),
    os.path.join(BASE_DIR, "Dataset", "placement.csv")
]

@st.cache_data
def load_dataset():

    for path in DATASET_PATHS:
        if os.path.exists(path):
            return pd.read_csv(path)

    return pd.DataFrame()


df = load_dataset()

# =====================================
# MODEL LOADING
# =====================================

def load_model(filename):

    path = os.path.join(BASE_DIR, filename)

    if os.path.exists(path):
        return joblib.load(path)

    return None


# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp {
    background-color: #FFFFF;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
}

.main-title {
    color: #FFFFFF;
    font-size: 35px;
    font-weight: 800;
}

.subtitle {
    color: #102A56;
    font-size: 17px;
    font-weight: 700;
    opacity: 1
}

.metric-card {
    padding: 24px 12px;
    border-radius: 17px;
    text-align: center;
    min-height: 155px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.07);
    border: 1px solid #DCE5F3;
}

.metric-title {
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 12px;
}

.metric-value {
    font-size: 36px;
    font-weight: 800;
}

.blue-card {
    background: #E1EEFF;
    color: #1553A5;
}

.green-card {
    background: #E1F7E9;
    color: #087443;
}

.orange-card {
    background: #FFF0DF;
    color: #D84315;
}

.purple-card {
    background: #EEE8FF;
    color: #5526B5;
}

.section-card {
    background: white;
    padding: 22px;
    border-radius: 17px;
    border: 1px solid #DCE5F3;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
}

.footer {
    background: #DCEBFF;
    padding: 22px;
    text-align: center;
    border-radius: 15px;
    color: #102A56;
    font-size: 21px;
    font-weight: 700;
}

.analytics-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #DFE7F1;
    text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.analytics-card h2 {
    color: #1456A0;
    font-size: 30px;
}

.analytics-card p {
    color: #64748B;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background-color: #062653;
}

[data-testid="stSidebar"] * {
    color: white;
}

div.stButton > button {
    background: #174A91;
    color: white;
    border-radius: 8px;
    width: 100%;
    height: 45px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: #0D2D5C;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# SIDEBAR NAVIGATION
# =====================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:20px 0;">
        <div style="font-size:65px;">🎓</div>
        <h2>STUDENT PLACEMENT PREDICTION SYSTEM </h2>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("### MAIN MENU")

    page = st.radio(
        "Navigation",
        [
            "🎯 Dashboard",
            "📊 Data Analysis",
            "🔮 Prediction",
            "📈 Model Comparison",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("🧠 Machine Learning Project | 2026")


# =====================================
# DASHBOARD PAGE
# =====================================

if page == "🎯 Dashboard":

    st.title("Dashboard")

    st.markdown("---")
    if df.empty:
        st.error(
            "Dataset not found. Please check your dataset folder."
        )

    else:

        total_students = len(df)

        if "PlacementStatus" in df.columns:

            status = (
                df["PlacementStatus"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            placed_students = status.isin(
                ["placed", "1", "true"]
            ).sum()

            not_placed = total_students - placed_students

        else:
            placed_students = 0
            not_placed = 0

        placement_rate = (
            placed_students / total_students * 100
            if total_students > 0 else 0
        )

        # Metric Cards

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card blue-card">
                <div style="font-size:36px;">👥</div>
                <div class="metric-title">Total Students</div>
                <div class="metric-value">{total_students:,}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card green-card">
                <div style="font-size:36px;">✅</div>
                <div class="metric-title">Placed Students</div>
                <div class="metric-value">{placed_students:,}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card orange-card">
                <div style="font-size:36px;">🚫</div>
                <div class="metric-title">Not Placed</div>
                <div class="metric-value">{not_placed:,}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card purple-card">
                <div style="font-size:36px;">📈</div>
                <div class="metric-title">Placement Rate</div>
                <div class="metric-value">{placement_rate:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Placement Distribution

        left, right = st.columns([1.1, 1])

        with left:

            st.markdown("""
            <div class="section-card">
                <h3 style="color:#102A56;">
                    📊 Placement Distribution
                </h3>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(data=[go.Pie(
                labels=["Placed", "Not Placed"],
                values=[placed_students, not_placed],
                hole=0.55,
                marker=dict(
                    colors=["#087EF5", "#FF9800"],
                    line=dict(color="white", width=3)
                ),
                textinfo="percent"
            )])

            fig.update_layout(
                annotations=[dict(
                    text=f"<b>{placement_rate:.2f}%</b><br>Placed",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20, color="#102A56")
                )],
                height=370,
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig, use_container_width=True)

        with right:

            st.markdown("""
            <div class="section-card">
                <h3 style="color:#102A56;">
                    🎓 Placement Summary
                </h3>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            st.markdown(f"""
            <div style="background:#E5F1FF;padding:22px;
                        border-radius:13px;margin-bottom:15px;">
                <h4 style="color:#1553A5;">🔵 Placed Students</h4>
                <h2 style="color:#1553A5;">{placed_students:,}</h2>
                <p style="color:#334155;">
                    {placement_rate:.2f}% of total students
                </p>
            </div>

            <div style="background:#FFF0DF;padding:22px;
                        border-radius:13px;">
                <h4 style="color:#D84315;">🟠 Not Placed Students</h4>
                <h2 style="color:#D84315;">{not_placed:,}</h2>
                <p style="color:#334155;">
                    {100-placement_rate:.2f}% of total students
                </p>
            </div>
            """, unsafe_allow_html=True)

      

# =====================================
# DATA ANALYSIS PAGE
# =====================================

elif page == "📊 Data Analysis":

    st.title("📊 Data Analysis")

    if df.empty:

        st.error("Dataset not found.")

    else:

        TARGET = "PlacementStatus"

        

        st.markdown("---")

        # Dataset Information

        total_records = len(df)
        total_columns = len(df.columns)
        missing_values = int(df.isnull().sum().sum())

        numeric_df = df.select_dtypes(include="number")
        numeric_columns = numeric_df.columns.tolist()

        

        
        # Metric Cards

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card blue-card">
                <div style="font-size:36px;"></div>
                <div class="metric-title">Total Students</div>
                <div class="metric-value">{total_records:,}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card green-card">
                <div style="font-size:36px;"></div>
                <div class="metric-title">Total Features</div>
                <div class="metric-value">{total_columns:,}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card orange-card">
                <div style="font-size:36px;"></div>
                <div class="metric-title">Numeric Features</div>
                <div class="metric-value">{len(numeric_columns):,}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card purple-card">
                <div style="font-size:36px;"></div>
                <div class="metric-title">Missing values</div>
                <div class="metric-value">{missing_values:,}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        st.write("")

        # Placement Distribution

        st.subheader("🎯 Placement Distribution")

        if TARGET in df.columns:

            placement_count = (
                df[TARGET]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .reset_index()
            )

            placement_count.columns = [
                "Placement Status", "Students"
            ]

            col1, col2 = st.columns(2)

            with col1:

                fig = px.pie(
                    placement_count,
                    names="Placement Status",
                    values="Students",
                    hole=0.58,
                    title="Placement Status Distribution",
                    color_discrete_sequence=[
                        "#1769AA", "#16A36A", "#F59E0B"
                    ]
                )

                fig.update_layout(height=370)

                st.plotly_chart(
                    fig, use_container_width=True
                )

            with col2:

                fig = px.bar(
                    placement_count,
                    x="Placement Status",
                    y="Students",
                    text="Students",
                    color="Placement Status",
                    title="Students Placement Comparison",
                    color_discrete_sequence=[
                        "#1769AA", "#16A36A", "#F59E0B"
                    ]
                )

                fig.update_layout(
                    height=370,
                    showlegend=False
                )

                st.plotly_chart(
                    fig, use_container_width=True
                )

        else:
            st.warning("PlacementStatus column not found.")

        st.markdown("---")

        # Academic Performance

        st.subheader("📚 Academic Performance Analysis")

        if numeric_columns:

            selected_feature = st.selectbox(
                "Select Academic Feature",
                numeric_columns
            )

            col1, col2 = st.columns(2)

            with col1:

                fig = px.histogram(
                    df,
                    x=selected_feature,
                    color=TARGET if TARGET in df.columns else None,
                    title=f"{selected_feature} Distribution",
                    marginal="box",
                    opacity=0.8,
                    color_discrete_sequence=[
                        "#1769AA", "#F59E0B"
                    ]
                )

                fig.update_layout(height=400)

                st.plotly_chart(
                    fig, use_container_width=True
                )

            with col2:

                fig = px.box(
                    df,
                    y=selected_feature,
                    color=TARGET if TARGET in df.columns else None,
                    title=f"{selected_feature} Performance",
                    color_discrete_sequence=[
                        "#1769AA", "#F59E0B"
                    ]
                )

                fig.update_layout(height=400)

                st.plotly_chart(
                    fig, use_container_width=True
                )

        else:
            st.info("No numeric columns available.")

        st.markdown("---")

        # Feature Relationship

        st.subheader("🔍 Feature Relationship Analysis")

        if len(numeric_columns) >= 2:

            col1, col2 = st.columns(2)

            with col1:
                x_feature = st.selectbox(
                    "Select X Axis",
                    numeric_columns,
                    index=0
                )

            with col2:
                y_feature = st.selectbox(
                    "Select Y Axis",
                    numeric_columns,
                    index=1
                )

            if x_feature != y_feature:

                fig = px.scatter(
                    df,
                    x=x_feature,
                    y=y_feature,
                    color=TARGET if TARGET in df.columns else None,
                    title=f"{x_feature} vs {y_feature}",
                    opacity=0.7,
                    color_discrete_sequence=[
                        "#1769AA", "#F59E0B"
                    ]
                )

                fig.update_layout(height=450)

                st.plotly_chart(
                    fig, use_container_width=True
                )

            else:
                st.info("Please select two different features.")

        st.markdown("---")

        # Correlation Heatmap

        st.subheader("🔥 Correlation Heatmap")

        if len(numeric_columns) >= 2:

            correlation = numeric_df.corr()

            fig = px.imshow(
                correlation,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Feature Correlation Matrix",
                aspect="auto"
            )

            fig.update_layout(height=550)

            st.plotly_chart(
                fig, use_container_width=True
            )

        else:
            st.info("Not enough numeric features for correlation.")

        st.markdown("---")

        # Dataset Preview

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Dataset CSV",
            data=csv_data,
            file_name="placement_analysis.csv",
            mime="text/csv"
        )


# =====================================
# PREDICTION PAGE
# =====================================

elif page == "🔮 Prediction":

      
    st.markdown("""
    <div class="main-title">
        🎓 Student Placement Prediction
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    
    st.markdown(
        '<p class="section-title">Student Information</p>',
        unsafe_allow_html=True
    )

    name = st.text_input("Student Name")

    col1, col2 = st.columns(2)

    with col1:

        cgpa = st.number_input(
            "CGPA", 0.0, 10.0, 7.0, step=0.1
        )

        internships = st.number_input(
            "Internships", 0, 20, 1
        )

        projects = st.number_input(
            "Projects", 0, 20, 1
        )

        workshops = st.number_input(
            "Workshops / Certifications", 0, 20, 1
        )

        aptitude = st.number_input(
            "Aptitude Test Score", 0, 100, 70
        )

    with col2:

        soft_skills = st.number_input(
            "Soft Skills Rating", 0.0, 5.0, 4.0, step=0.1
        )

        extracurricular = st.selectbox(
            "Extracurricular Activities", ["Yes", "No"]
        )

        training = st.selectbox(
            "Placement Training", ["Yes", "No"]
        )

        ssc = st.number_input(
            "SSC Marks", 0, 100, 70
        )

        hsc = st.number_input(
            "HSC Marks", 0, 100, 70
        )

    st.markdown(
        '<p class="section-title">Prediction Settings</p>',
        unsafe_allow_html=True
    )

    model_choice = st.selectbox(
        "Select Machine Learning Algorithm",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

    if st.button("Predict Placement"):

        if not name.strip():

            st.warning("Please enter Student Name.")

        else:

            model_files = {
                "Logistic Regression": "logistic_regression.pkl",
                "Decision Tree": "decision_tree.pkl",
                "Random Forest": "random_forest.pkl"
            }

            try:

                selected_data = load_model(
                    model_files[model_choice]
                )

                if selected_data is None:

                    st.error(
                        "Model file not found. "
                        "Please run train_models.py first."
                    )

                else:

                    model = selected_data["model"]
                    scaler = selected_data["scaler"]
                    features = selected_data["features"]

                    input_data = pd.DataFrame([{
                        "CGPA": cgpa,
                        "Internships": internships,
                        "Projects": projects,
                        "Workshops/Certifications": workshops,
                        "AptitudeTestScore": aptitude,
                        "SoftSkillsRating": soft_skills,
                        "ExtracurricularActivities":
                            1 if extracurricular == "Yes" else 0,
                        "PlacementTraining":
                            1 if training == "Yes" else 0,
                        "SSC_Marks": ssc,
                        "HSC_Marks": hsc
                    }])

                    input_data = input_data[features]

                    scaled_input = scaler.transform(input_data)

                    prediction = int(
                        model.predict(scaled_input)[0]
                    )

                    st.markdown("---")
                    st.subheader("Placement Prediction Result")

                    if prediction == 1:

                        st.success("🎉 STUDENT IS PLACED")

                    else:

                        st.error("STUDENT IS NOT PLACED")

                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(
                            scaled_input
                        )[0]

                        classes = list(model.classes_)
                        index = classes.index(prediction)

                        confidence = probabilities[index] * 100

                        st.metric(
                            "Prediction Confidence",
                            f"{confidence:.2f}%"
                        )

                    st.caption(f"Student Name: {name}")
                    st.caption(f"Selected Algorithm: {model_choice}")

            except Exception as e:

                st.error(f"Prediction Error: {e}")


# =====================================
# MODEL COMPARISON PAGE
# =====================================

elif page == "📈 Model Comparison":


    st.markdown("""
    <div class="main-title">
        📈 Model Comparison
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    

    st.subheader("📈 Model Accuracy Comparison")

    results_path = os.path.join(
        BASE_DIR, "model_comparison.pkl"
    )

    if os.path.exists(results_path):

        results = joblib.load(results_path)

        if results:

            comparison_df = pd.DataFrame({
                "Algorithm": list(results.keys()),
                "Accuracy": [
                    value * 100 for value in results.values()
                ]
            })

            st.subheader("Algorithm Performance")

            c1, c2, c3 = st.columns(3)

            for col, (_, row) in zip(
                [c1, c2, c3],
                comparison_df.iterrows()
            ):

                with col:
                    st.metric(
                        row["Algorithm"],
                        f'{row["Accuracy"]:.2f}%'
                    )

            st.markdown("---")

            fig = px.bar(
                comparison_df,
                x="Algorithm",
                y="Accuracy",
                text=comparison_df["Accuracy"].apply(
                    lambda x: f"{x:.2f}%"
                ),
                color="Algorithm",
                title="Accuracy Comparison of ML Algorithms",
                color_discrete_sequence=[
                    "#1769AA", "#16A36A", "#F59E0B"
                ]
            )

            fig.update_layout(
                yaxis_title="Accuracy (%)",
                xaxis_title="Machine Learning Algorithm",
                yaxis_range=[0, 100],
                height=450,
                showlegend=False
            )

            st.plotly_chart(
                fig, use_container_width=True
            )

            best_row = comparison_df.loc[
                comparison_df["Accuracy"].idxmax()
            ]

            st.success(
                f'Best Performing Model: {best_row["Algorithm"]} '
                f'({best_row["Accuracy"]:.2f}%)'
            )

            st.subheader("Comparison Table")

            st.dataframe(
                comparison_df.sort_values(
                    by="Accuracy",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning("No model comparison results available.")

    else:

        st.warning(
            "Model comparison file not found. "
            "Please run train_models.py first."
        )


# =====================================
# ABOUT PAGE
# =====================================

elif page == "ℹ️ About":

    st.title("About Project")

    st.markdown("---")

    st.subheader("🎓 Project Overview")

    st.write("""
    Student Placement Prediction System is a Machine Learning
    based application designed to predict whether a student
    is likely to be placed or not placed.

    The system analyzes academic performance, internships,
    projects, aptitude scores, soft skills and placement training.
    """)

    st.subheader("🧠 Machine Learning Algorithms")

    st.write("1. Logistic Regression")
    st.write("2. Decision Tree")
    st.write("3. Random Forest")

    st.subheader("⚙️ Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Plotly
    - Joblib
    """)

    st.subheader("🎯 Project Objectives")

    st.write("""
    - Analyze student academic performance.
    - Predict student placement status.
    - Compare three Machine Learning algorithms.
    - Visualize placement statistics and trends.
    """)

    st.markdown("---")

   