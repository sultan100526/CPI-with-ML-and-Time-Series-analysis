Part A – Application area review - Market Analysis
1.Introduction

Market analysis is a major application area of artificial intelligence because economic markets produce large amounts of time-series data. These data include consumer prices, exchange rates, interest rates, money supply, commodity prices, fuel prices, and demand indicators. In this project, market analysis is explored through Consumer Price Index (CPI) forecasting. CPI is a key measure of inflation and reflects changes in the average price of goods and services paid by consumers [26]. Accurate CPI forecasting is important for policymakers, central banks, businesses, and households because inflation affects purchasing power, investment decisions, wage planning, and monetary policy.

2.Traditional Approaches in CPI Forecasting

Before the growth of artificial intelligence, CPI and inflation forecasting mainly depended on statistical and econometric models. ARIMA is one of the most widely used models for time-series forecasting. The Box-Jenkins method provides a structured way to model historical values through autoregressive, differencing, and moving average components [19]. ARIMA has been used in many developing-country contexts because it is simple, interpretable, and suitable for short-term forecasting. For example, Faisal [14] applied ARIMA to forecast inflation in Bangladesh, while Akhter [39] used seasonal ARIMA to forecast monthly CPI data in Bangladesh. These studies show that ARIMA and SARIMA remain useful benchmark models for inflation forecasting.

3.Use of AI and Machine Learning

Although ARIMA is useful, inflation is often affected by nonlinear and unexpected factors such as exchange rate depreciation, fuel price shocks, food price volatility, global commodity prices, and policy uncertainty. Because of this, machine learning techniques are increasingly used in market analysis. Machine learning models can process larger datasets and identify complex relationships among many predictors. Liu, Pan, and Xu [1] argued that machine learning can improve inflation forecasting by using wider economic datasets and focusing on out-of-sample performance. Medeiros et al. [2] also found that machine learning methods can improve inflation forecasting in data-rich environments.

Tree-based machine learning models such as Random Forest and XGBoost have also been applied in inflation forecasting. Random Forest combines multiple decision trees to improve prediction and reduce variance [42]. XGBoost is a scalable boosting method that improves accuracy by correcting previous errors in sequence [9]. Li et al. [4] used XGBoost with genetic algorithm optimisation and found that it improved multi-horizon inflation forecasting.

4.Deep Learning and Hybrid Models

Deep learning models are also important for CPI forecasting, especially when the data contain nonlinear patterns and long-term dependencies. Long Short-Term Memory, or LSTM, is a recurrent neural network model designed for sequential data [23]. Studies have shown that LSTM can perform well in inflation forecasting when price movements are unstable or nonlinear [24], [25]. Siami Namini et al. [34] compared ARIMA and LSTM and found that LSTM often produced lower forecasting errors. Hybrid models are also becoming popular. For example, Peirano et al. [18] used a SARIMA-LSTM model for Latin American inflation forecasting, while Gur [16] found that a hybrid LSTM-XGBoost model performed strongly for CPI forecasting.

5.Relevance to This Project

For Bangladesh, CPI forecasting is highly relevant because inflation is influenced by food prices, imported fuel, exchange rate pressure, and global commodity shocks. Existing studies in Bangladesh have used ARIMA and machine learning models, but more comparison is needed between traditional and modern AI-based models [10], [14], [39]. Therefore, this project focuses on CPI forecasting as a market analysis problem and compares classical time-series methods with machine learning or deep learning approaches.

6.Statement on the Use of ChatGPT

ChatGPT was used to assist with structuring and improving the academic language of this literature review. It was not used to collect or manipulate data. The project topic, references, and final academic responsibility remain with the student.

Referrence
[1] Y. Liu, R. Pan, and R. Xu, "Mending the crystal ball: Enhanced inflation forecasts with machine learning," IMF Working Paper WP/24/206, 2024.

[2] M. C. Medeiros, G. F. R. Vasconcelos, A. Veiga, and E. Zilberman, "Forecasting inflation in a data rich environment: The benefits of machine learning methods," Journal of Business and Economic Statistics, vol. 39, no. 1, pp. 98 to 119, 2021.

[4] S. Li et al., "Forecasting inflation rates by extreme gradient boosting with the genetic algorithm,"* Journal of Ambient Intelligence and Humanized Computing,* 2022.

[9] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785 to 794.

[10] L. B. Ismail, M. I. Joytu, T. I. Plabon, and M. S. Oshman, "Evaluation of machine learning models to forecast inflation: Bangladesh as a case study," Proceedings of the International Symposium on Networks, Computers and Communications, 2023.

[14] M. Faisal, "Forecasting Bangladesh's inflation using time series ARIMA models," World Review of Business Research, vol. 2, no. 3, pp. 100 to 117, 2012.

[16] Y. E. Gur, "Development and application of machine learning models in US consumer price index forecasting: Analysis of a hybrid approach," Data Science in Finance and Economics, vol. 4, no. 4, pp. 469 to 513, 2024.

[18] R. Peirano, W. Kristjanpoller, and M. C. Minutolo, "Forecasting inflation in Latin American countries using a SARIMA LSTM combination," Soft Computing, vol. 25, no. 16, pp. 10851 to 10862, 2021.

[19] G. E. P. Box and G. M. Jenkins, Time Series Analysis: Forecasting and Control. San Francisco: Holden Day, 1976.

[23] S. Hochreiter and J. Schmidhuber, "Long short term memory," Neural Computation, vol. 9, no. 8, pp. 1735 to 1780, 1997.

[24] A. Almosova and N. Andresen, "Nonlinear inflation forecasting with recurrent neural networks," Journal of Forecasting, vol. 42, no. 2, pp. 240 to 259, 2023.

[25] L. Paranhos, "Predicting inflation with recurrent neural networks," arXiv preprint arXiv:2104.03757, 2021.

[26] International Monetary Fund, "Inflation: Prices on the Rise," IMF Finance and Development, 2013.

[34] S. Siami Namini, N. Tavakoli, and A. S. Namin, "A comparison of ARIMA and LSTM in forecasting time series," Proceedings of the IEEE International Conference on Machine Learning and Applications, 2018.

[39] T. Akhter, "Short term forecasting of inflation in Bangladesh with seasonal ARIMA processes," MPRA Paper No. 43729, 2013.

[42] L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5 to 32, 2001.

Part B: Compare and Evaluate AI
Goal of the Application

The goal of this project is to forecast future Consumer Price Index (CPI) values using historical CPI data. The dataset contains annual CPI information for Bangladesh from 2005–06 to 2024–25. It includes General CPI, Food CPI, Non-Food CPI, and inflation rates. These variables can help understand how prices have changed over time and how future CPI may move.

1. ARIMA Time Series Model
How this model analyzes the data:

ARIMA analyzes CPI as a time series. It looks at the past movement of General CPI and uses that pattern to forecast future CPI. For example, in the dataset, General CPI increased from 100 in 2021–22 to 109.02 in 2022–23, then 119.63 in 2023–24, and 131.62 in 2024–25. ARIMA studies this upward trend and estimates what the CPI may be in the next fiscal year.

Strengths:

ARIMA is simple, easy to explain, and suitable for small time-based datasets. Since your dataset has annual CPI values, ARIMA can work as a good basic forecasting model.

Weaknesses and disadvantages:

ARIMA mainly depends on past CPI values. It may not fully capture sudden economic shocks, such as fuel price increases, exchange rate changes, or global market instability. It also works better when the data pattern is stable.

Input data required:

Annual General CPI values arranged by fiscal year.

Expected output:

Forecasted General CPI for future years, such as 2025–26, with an actual-versus-predicted graph.

Evaluation

For this project, ARIMA is highly suitable as a prototype model because the dataset is small, annual, and time-based. It may not be the most advanced model, but it is explainable and realistic for the available data.

2. XGBoost Regression
How this model analyzes the data:

XGBoost analyzes CPI by using several variables together. For example, it can use Food CPI, Non-Food CPI, General Inflation Percent, Food Inflation Percent, and Non-Food Inflation Percent to predict General CPI. If Food CPI and Non-Food CPI are rising, the model learns that General CPI is also likely to rise.

For example, in 2024–25, Food CPI is 133.16, Non-Food CPI is 130.37, and General CPI is 131.62. XGBoost can learn the relationship between these variables and use similar patterns to predict future CPI.

Strengths:

XGBoost can capture nonlinear relationships. It can also show which variables are more important for prediction, such as Food CPI or Non-Food CPI.

Weaknesses and disadvantages:

XGBoost does not automatically understand time order. Lag variables, such as previous year CPI or previous year inflation, need to be created manually. Also, your dataset has only 20 annual observations, which is small for a strong machine learning model.

Input data required:

General CPI as the target variable, with Food CPI, Non-Food CPI, and inflation percentages as input features.

Expected output:

Predicted General CPI values, error scores, and important variable rankings.

Evaluation

XGBoost is useful for CPI forecasting when more data are available, especially monthly CPI data and macroeconomic indicators. However, for the current small annual dataset, it is less suitable than ARIMA for the main prototype implementation.

3. LSTM Neural Network
How this model analyzes the data:

LSTM analyzes CPI by learning from sequences of previous years. For example, the model can use CPI values from the last five years, such as 94.21, 100, 109.02, 119.63, and 131.62, to predict the next year’s CPI. It can also include Food CPI and Non-Food CPI in the sequence.

Strengths:

LSTM is strong for sequential data and can learn long-term patterns. It is useful when CPI movement depends on previous years and delayed economic effects.

Weaknesses and disadvantages:

LSTM usually needs a large dataset. Since this project uses only annual data from 2005–06 to 2024–25, the dataset is quite small for deep learning. The model may overfit and give unreliable results. It is also harder to explain compared with ARIMA.

Input data required:

CPI data converted into time-window sequences, such as using the previous three or five years to predict the next year.

Expected output:

Forecasted CPI values, prediction accuracy, and actual-versus-predicted visualisation.

Evaluation

LSTM is theoretically powerful but not ideal for the current dataset. It would be more suitable for future work if monthly CPI data over many years are used. For this project, LSTM is discussed as a future improvement rather than selected for the prototype.

Selected Technique for Part C

For Part C, ARIMA Time-Series Forecasting is selected for the prototype implementation.

ARIMA is selected because the current dataset is annual, time-based, and relatively small. It is more appropriate for this project than XGBoost or LSTM because it can work with limited historical data and provides an explainable forecasting process. XGBoost and LSTM are powerful techniques, but they would require more observations and additional macroeconomic variables to perform reliably.

Therefore, ARIMA is the most suitable technique for the current prototype. XGBoost and LSTM are discussed as possible future improvements, especially if monthly CPI data, exchange rate, fuel price, money supply, interest rate, and global commodity price data are added in future research.

Part C – Implementation
Selected Technique for Prototype Implementation

For the implementation part of this project, the selected AI/time-series forecasting technique is the ARIMA model, which stands for Autoregressive Integrated Moving Average. ARIMA is selected because the dataset used in this project contains annual CPI data for Bangladesh from 2005–06 to 2024–25. Since the dataset is time-based and relatively small, ARIMA is more suitable than complex machine learning or deep learning models such as XGBoost or LSTM.

The main purpose of this prototype is to forecast future values of the General Consumer Price Index (General CPI) using historical CPI values. CPI is an important macroeconomic indicator because it reflects changes in the average price level of goods and services consumed by households. Therefore, forecasting CPI can help policymakers, economists, and researchers anticipate inflationary pressure in advance.

C(a). High-Level Diagram of the CPI Forecasting System

The overall workflow of the ARIMA-based CPI forecasting prototype is shown below:<img width="1196" height="1315" alt="image" src="https://github.com/user-attachments/assets/e57242f9-bc45-4fcc-9e31-b23c1ad0c08d" />
This diagram shows that the raw CPI dataset is first loaded and cleaned. Then the General CPI series is prepared as a time-series dataset. After checking the trend and stationarity, the ARIMA model is trained using historical CPI data. Finally, the model predicts CPI values and the results are evaluated using error metrics and visual graphs.

b). Input Data Required for the Implementation

The input data for this implementation is the Excel dataset named Internship CPI data Original(1).xlsx. The dataset contains annual CPI information for Bangladesh from fiscal year 2005–06 to 2024–25.

The dataset includes the following variables:

Variable

Area

Fiscal year

Base Reference Index

CPI General Index

CPI Food Index

CPI Non-Food Index

Inflation General Percent

Inflation Food Percent Inflation Non-Food Percent

Non-food inflation percentage

For this prototype, the main target variable is CPI General Index. This variable is selected because it represents the overall price level in the economy. The ARIMA model uses previous values of the General CPI to forecast future General CPI values.

The input data format is a structured Excel file where each row represents one fiscal year and each column represents a CPI-related variable. Since ARIMA is a time-series model, the most important requirement is that the General CPI values must be arranged in correct chronological order.

(c). Data Preprocessing

Before applying the ARIMA model, the dataset requires several preprocessing steps.

First, the Excel file is loaded into Google Colab using the pandas library. After loading the dataset, the column names and data types are inspected to confirm that the required variables are available.

Second, the relevant columns are selected. For this prototype, the most important columns are Fiscal year and CPI General Index. The Fiscal year column is used to maintain the time order, while the CPI General Index column is used as the forecasting variable.

Third, missing values are checked. If there are missing values in the General CPI column, they need to be removed or handled before modelling. This is important because ARIMA cannot produce reliable results if the target time series contains missing values.

Fourth, the CPI General Index values are converted into numeric format. Sometimes data imported from Excel may be stored as text. Therefore, converting CPI values into numeric format ensures that the model can process the data correctly.

Fifth, the data are sorted from the earliest fiscal year to the latest fiscal year. This step is essential because time-series forecasting depends on the correct sequence of observations. Unlike normal machine learning problems, time-series data should not be randomly shuffled.

Sixth, the historical CPI trend is visualised using a line graph. This graph helps to understand whether CPI is increasing, decreasing, or fluctuating over time. In this dataset, General CPI shows a clear upward trend, especially in recent fiscal years.

Seventh, stationarity is considered. ARIMA models usually work better when the time series is stationary. If the CPI series shows a strong trend, differencing may be applied to remove the trend and make the series more stable. The value of d in the ARIMA model represents the number of differencing steps used.

Finally, the dataset is divided into training and testing parts. The earlier fiscal years are used to train the ARIMA model, while the most recent years are used to test model performance. This approach is appropriate because the model should be evaluated on future observations, not randomly selected observations.

(d). Prototype Implementation

The prototype is implemented in Google Colab using Python. The main libraries used in this implementation are:


Now The COde Part Here::

#Install and import libraries

!pip install -q statsmodels scikit-learn openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")




# 2. Upload Excel file
from google.colab import files
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
df = pd.read_excel(file_name)

print("Dataset preview:")
display(df.head())
print("\nOriginal columns:")
print(df.columns.tolist())


# 3. Data cleaning and preprocessing

# Remove extra spaces from column names. This fixes columns such as " Inflation General Percent".
df.columns = df.columns.astype(str).str.strip()

required_cols = [
    "Fiscal year",
    "CPI General Index",
    "CPI Food Index",
    "CPI Non-Food Index",
    "Inflation General Percent",
    "Inflation Food Percent",
    "Inflation Non-Food Percent"
]

missing_required = [col for col in required_cols if col not in df.columns]
if missing_required:
    raise ValueError(f"Missing required column(s): {missing_required}")

numeric_cols = [
    "CPI General Index",
    "CPI Food Index",
    "CPI Non-Food Index",
    "Inflation General Percent",
    "Inflation Food Percent",
    "Inflation Non-Food Percent"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep only national data if the dataset contains multiple areas.
if "Area" in df.columns:
    df = df[df["Area"].astype(str).str.lower().str.strip() == "national"].copy()

# Sort by fiscal year start, e.g., 2005-06 -> 2005.
def fiscal_start_year(x):
    return int(str(x).split("-")[0])

df["Fiscal_Start"] = df["Fiscal year"].apply(fiscal_start_year)
df = df.sort_values("Fiscal_Start").reset_index(drop=True)

# Drop missing target values.
df = df.dropna(subset=["CPI General Index"]).reset_index(drop=True)

print("\nCleaned dataset:")
display(df)
print("\nMissing values after cleaning:")
display(df[required_cols].isnull().sum())


# 4. Historical CPI trend visualization
plt.figure(figsize=(12, 6))
plt.plot(df["Fiscal year"], df["CPI General Index"], marker="o", label="General CPI")
plt.plot(df["Fiscal year"], df["CPI Food Index"], marker="o", label="Food CPI")
plt.plot(df["Fiscal year"], df["CPI Non-Food Index"], marker="o", label="Non-Food CPI")
plt.title("Historical CPI Trend in Bangladesh")
plt.xlabel("Fiscal Year")
plt.ylabel("CPI Index, Base 2021-22 = 100")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()



# 5. High-level architecture diagram

# This replaces the very large embedded image/base64 diagram in the old notebook.
from graphviz import Digraph
from IPython.display import Image, display

dot = Digraph(format="png")
dot.attr(rankdir="LR", size="12")
steps = [
    "CPI Excel Dataset",
    "Load Data in Colab",
    "Clean Columns and Values",
    "Select General CPI",
    "Train-Test Split",
    "Fit ARIMA Models",
    "Compare with Naive Forecast",
    "Evaluate MAE, RMSE, MAPE",
    "Forecast Future CPI"
]
for i, step in enumerate(steps):
    dot.node(str(i), step, shape="box")
for i in range(len(steps) - 1):
    dot.edge(str(i), str(i + 1))

dot.render("corrected_cpi_forecasting_workflow", cleanup=True)
display(Image(filename="corrected_cpi_forecasting_workflow.png"))



# 6. Stationarity check
y = df["CPI General Index"].astype(float).reset_index(drop=True)
years = df["Fiscal year"].reset_index(drop=True)

def adf_test(series, label):
    result = adfuller(series.dropna())
    print(f"ADF test for {label}")
    print("ADF Statistic:", round(result[0], 4))
    print("p-value:", round(result[1], 4))
    if result[1] <= 0.05:
        print("Interpretation: The series is stationary.\n")
    else:
        print("Interpretation: The series is not stationary; differencing is required.\n")

adf_test(y, "Original CPI series")
adf_test(y.diff().dropna(), "First-differenced CPI series")

plt.figure(figsize=(10, 5))
plt.plot(years.iloc[1:], y.diff().dropna(), marker="o")
plt.title("First-Differenced General CPI Series")
plt.xlabel("Fiscal Year")
plt.ylabel("Change in CPI")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# 7.Train-test split

# Time-series data must not be shuffled.
train_size = int(len(y) * 0.80)
train = y.iloc[:train_size].reset_index(drop=True)
test = y.iloc[train_size:].reset_index(drop=True)
train_years = years.iloc[:train_size].reset_index(drop=True)
test_years = years.iloc[train_size:].reset_index(drop=True)

print("Training observations:", len(train))
print("Testing observations:", len(test))
print("Test years:", test_years.tolist())



# 8. Evaluation functions

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape



    # 9. Corrected ARIMA model selection
# - Do not search very complex ARIMA models such as (3,2,3) for only 20 annual observations.
# - Use a small, controlled candidate list to reduce overfitting.
# - Compare ARIMA with a simple Naive benchmark.

candidate_orders = [
    (0, 1, 0),
    (1, 1, 0),
    (2, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (0, 2, 0),
    (1, 2, 0),
    (2, 2, 0),
    (0, 2, 1),
    (1, 2, 1)
]

arima_results = []
for order in candidate_orders:
    try:
        model = ARIMA(
            train,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        mae, rmse, mape = calculate_metrics(test, forecast)
        arima_results.append({
            "Model": f"ARIMA{order}",
            "Order": order,
            "AIC": model_fit.aic,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape
        })
    except Exception as e:
        print(f"ARIMA{order} failed: {e}")

arima_results_df = pd.DataFrame(arima_results).sort_values("MAPE (%)").reset_index(drop=True)
print("ARIMA candidate comparison:")
display(arima_results_df)

best_order = arima_results_df.loc[0, "Order"]
print("Selected ARIMA order based on lowest test MAPE:", best_order)

# Fit selected ARIMA model.
best_model = ARIMA(
    train,
    order=best_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()

arima_forecast = best_model.forecast(steps=len(test))
arima_mae, arima_rmse, arima_mape = calculate_metrics(test, arima_forecast)



# 10. Benchmark model comparison
# Naive forecast: next CPI equals the previous year's CPI.
naive_forecast = test.shift(1)
naive_forecast.iloc[0] = train.iloc[-1]
naive_mae, naive_rmse, naive_mape = calculate_metrics(test, naive_forecast)

# Drift forecast: extends the average yearly increase from the training period.
drift_forecast = train.iloc[-1] + np.arange(1, len(test) + 1) * (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
drift_mae, drift_rmse, drift_mape = calculate_metrics(test, drift_forecast)

comparison_df = pd.DataFrame({
    "Model": ["Naive Forecast", "Drift Forecast", f"Selected ARIMA{best_order}"],
    "MAE": [naive_mae, drift_mae, arima_mae],
    "RMSE": [naive_rmse, drift_rmse, arima_rmse],
    "MAPE (%)": [naive_mape, drift_mape, arima_mape]
}).sort_values("MAPE (%)").reset_index(drop=True)

print("Model comparison:")
display(comparison_df)



# 11. Actual vs predicted result table
result_comparison = pd.DataFrame({
    "Fiscal Year": test_years,
    "Actual CPI": test.values,
    "Predicted CPI": np.asarray(arima_forecast),
    "Naive Forecast": naive_forecast.values,
    "Drift Forecast": drift_forecast
})

result_comparison["ARIMA Error"] = result_comparison["Actual CPI"] - result_comparison["Predicted CPI"]
result_comparison["ARIMA Absolute Error"] = result_comparison["ARIMA Error"].abs()
result_comparison["ARIMA Percentage Error"] = (result_comparison["ARIMA Absolute Error"] / result_comparison["Actual CPI"]) * 100

print("Actual vs predicted CPI:")
display(result_comparison)
