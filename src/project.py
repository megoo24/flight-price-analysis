import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv(r"C:\Users\Acer\Documents\Data Analyst-NTI\Final_Project\Clean_Dataset.csv")

print("-"*50)
print("First 5 Rows of the Dataset:")
print(df.head(5))
print("-"*50)
print("Data Information:")
print(df.info())
print("-"*50)
print("Data Description:")
print(df.describe())
print("-"*50)
print("Columns in the dataset:")
print(df.columns)
print("-"*50)
print("Shape of the dataset:")
print(df.shape)
print("-"*50)
print("Number of Unique Values:")
print(df.nunique())
print("-"*50)
print("Data Types:")
print(df.dtypes)
print("-"*50)
print("Check Missing Values:")
print("-"*50)
print("Check Duplicates:")
print(df.duplicated().sum())
print("-"*50)
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("Numeric columns:", numeric_cols)
print("Shape before outlier removal:", df.shape)
plt.boxplot(df[numeric_cols])
plt.title('Boxplot of Numeric Columns')
plt.show()

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

print("-"*50)
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.show()
print("-"*50)
print("Shape after outlier removal:", df.shape)
plt.boxplot(df[numeric_cols])
plt.title('Boxplot of Numeric Columns After Outlier Removal')
plt.show()
print("-"*50)
print("Dataset after outlier removal:")
print(df.head())

profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
profile.to_file("data_profile_report.html")