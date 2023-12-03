# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
# from google.colab import drive
from sklearn.preprocessing import StandardScaler

# Read Dataset
df = pd.read_csv('loan_data_2007_2014.csv')

# First 5 row of data
df.head()

# Data info
df.info()

# Data insight
df.describe()

# Missing value
listItem = []
for col in df.columns:
    listItem.append([col, df[col].dtype, df[col].isnull().sum(), round((df[col].isnull().sum()/len(df[col]))*100, 2), df[col].nunique(), list(df[col].drop_duplicates().sample(5,replace=True).values)]);
df_desc = pd.DataFrame(columns=['Column', 'Dtype', 'null count', 'null perc.', 'unique count', 'unique sample'],
                     data=listItem)
df_desc

# Copy the DataFrame
df_copy = df.copy()

# Data shape before threshold cut
df_copy.shape

# Drop column with missing values more than 70%
threshold = len(df_copy) * 0.7
df_clean = df_copy.dropna(axis=1, thresh=threshold)

# Data shape after threshold cut
df_copy.shape

# Check value for loan_status
df_clean.loan_status.value_counts()

# Define the categories for Excellent loans and bad loans
excellent_loan_statuses = ['Current', 'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']
bad_loan_statuses = ['Charged Off', 'Late (31-120 days)', 'In Grace Period', 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off']

# Create a new column 'loan_category' to classify loans as 'Excellent' or 'Bad'
df['loan_category'] = df['loan_status'].apply(lambda x: 'Excellent' if x in excellent_loan_statuses else 'Bad')
df_clean = pd.concat([df_clean, df['loan_category']], axis=1)

# Count the occurrences of each loan category
loan_category_counts = df['loan_category'].value_counts()

# Set colors for different loan categories
colors = ['green', 'red']

# Plot the distribution of loan categories
plt.bar(loan_category_counts.index, loan_category_counts.values, color=colors)
plt.xlabel('Loan Category')
plt.ylabel('Count')
plt.title('Distribution of Loan Categories')

# Add values on top of the bars
for i, count in enumerate(loan_category_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
plt.show()

# Check remaining columns
df_clean.columns
for column in df_clean.columns:
    value_counts = df_clean[column].value_counts()
    print(f"Value counts for {column}:\n{value_counts}\n")

# Drop Unecessary Column
unused_col = ['policy_code', 'application_type', 'Unnamed: 0', 'id', 'member_id','issue_d', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                   'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                   'last_pymnt_d', 'last_pymnt_amnt', 'zip_code', 'title', 'emp_title','loan_status']
drop_data = df_clean[unused_col]
df_clean.drop(columns=unused_col, axis=1, inplace=True)

# Check df head
df_clean.head()

# Check missing values before imputation
df_clean.isnull().sum()

# List of columns for imputation
categorical_columns = ['emp_length', 'verification_status', 'earliest_cr_line', 'last_credit_pull_d']
numerical_columns = ['annual_inc', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_util', 'total_acc',
                    'collections_12_mths_ex_med', 'acc_now_delinq']

# Impute categorical columns with mode
for col in categorical_columns:
    mode_value = df_clean[col].mode()[0]
    df_clean[col].fillna(mode_value, inplace=True)

# Impute numerical columns with median
for col in numerical_columns:
    median_value = df_clean[col].median()
    df_clean[col].fillna(median_value, inplace=True)

# Check missing values after imputation
df_clean.isnull().sum()

# Check duplication
df_clean.duplicated().any()

# Calculate the correlation matrix
correlation_matrix = df_clean.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Similarity  Check
print(df_clean[['loan_amnt', 'funded_amnt', 'funded_amnt_inv']].describe())

# Drop similar column
unused_col2 = ['funded_amnt', 'funded_amnt_inv','url','pub_rec','installment', 'dti', 'revol_bal', 'total_acc','earliest_cr_line', 'last_credit_pull_d','initial_list_status','tot_coll_amt', 'sub_grade'
,'tot_cur_bal','total_rev_hi_lim','purpose', 'addr_state', ]
df_clean2 = df_clean.drop(columns = unused_col2)

# Check df head
df_clean2.head()

# Check data correlation
correlation_matrix = df_clean2.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Check data type
df_clean2.info()

# Check data describe
df_clean2.describe().T

# Split term column
df_clean2['term'] = df_clean2['term'].apply(lambda x: int(x.split()[0]))
df_clean2['term']

# Split emp_length column
df_clean2['emp_length'] = df_clean2['emp_length'].str.extract('(\d+)').astype(int)
df_clean2['emp_length']

# Feature Encoding Column
encoded_verification = pd.get_dummies(df_clean2['verification_status'], prefix='verification', drop_first=True)
encoded_home_ownership = pd.get_dummies(df_clean2['home_ownership'], prefix='home', drop_first=True)
encoded_grade = pd.get_dummies(df_clean2['grade'], prefix='grade', drop_first=True)

# Combine the encoded features
encoded_categorical = pd.concat([encoded_verification,encoded_home_ownership, encoded_grade], axis=1)
df_clean2 = pd.concat([df_clean2, encoded_categorical], axis=1)

# Drop the original columns
df_clean2.drop(['verification_status','home_ownership', 'grade','term','emp_length'], axis=1, inplace=True)

# Check head
df_clean2.head()

# Check info
df_clean2.info()

# Check value counts
df_clean2['loan_category'].value_counts()

# Define features (X) and target variable (Y)
X = df_clean2.drop('loan_category', axis=1)  # Features
Y = df_clean2['loan_category']  # Target variable

# Get the column names as feature names
feature_names = X.columns.tolist()

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Initialize different models
results = {}
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
}

# Initialize dictionary to store classification reports
classification_reports = {}
model_names = []
accuracies = []

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, Y_train)

    print(f"Evaluating {model_name}...")
    Y_pred = model.predict(X_test)

    confusion = confusion_matrix(Y_test, Y_pred)
    classification_rep = classification_report(
        Y_test, Y_pred, target_names=['Good', 'Bad'], zero_division=1  # Handle zero division
    )

    # Store the classification report in the dictionary
    classification_reports[model_name] = classification_rep

    accuracy = accuracy_score(Y_test, Y_pred)

    model_names.append(model_name)
    accuracies.append(accuracy)

    print("\nClassification Report:")
    print(classification_rep)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("=" * 50)

# Check model names
model_names

# Check accuracies
accuracies

# Create a bar plot to visualize accuracies
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.ylim(0, 1)  # Set y-axis limits to 0-1 for accuracy percentage
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()

# Check value counts
df_clean2['loan_category'].value_counts()

# Split X & y
X_o = df_clean2.drop(['loan_category'], axis=1)
y_o = df_clean2['loan_category']

# Check before oversampling
sns.set_style(style='darkgrid')
sns.countplot(data=pd.DataFrame(y_o),x='loan_category')
plt.title('Number of samples of each class (after oversampling)', fontsize=16)
plt.show()

# Oversampling
oversample = RandomOverSampler(sampling_strategy = 'not majority')
X_over, y_over = oversample.fit_resample(X_o, y_o)

# Check after oversampling
sns.set_style(style='darkgrid')
sns.countplot(data=pd.DataFrame(y_over),x='loan_category')
plt.title('Number of samples of each class (after oversampling)', fontsize=16)
plt.show()

# Split the data into training and testing sets (80% training, 20% testing)
X_train_over, X_test_over, Y_train_over, Y_test_over = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_over = scaler.fit_transform(X_train_over)
X_test_over = scaler.fit_transform(X_test_over)

# Initialize different models
results = {}
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
}

# Initialize dictionary to store classification reports
classification_reports = {}
model_names_over = []
accuracies_over = []

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_over, Y_train_over)

    print(f"Evaluating {model_name}...")
    Y_pred = model.predict(X_test_over)

    confusion = confusion_matrix(Y_test_over, Y_pred)
    classification_rep = classification_report(
        Y_test_over, Y_pred, target_names=['Good', 'Bad'], zero_division=1  # Handle zero division
    )

    # Store the classification report in the dictionary
    classification_reports[model_name] = classification_rep

    accuracy = accuracy_score(Y_test_over, Y_pred)

    model_names_over.append(model_name)
    accuracies_over.append(accuracy)

    print("\nClassification Report:")
    print(classification_rep)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("=" * 50)

# Model name oversampling
model_names_over

# Accuracies oversampling
accuracies_over

# Create a bar plot to visualize accuracies
plt.figure(figsize=(10, 6))
plt.bar(model_names_over, accuracies_over, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.ylim(0, 1)  # Set y-axis limits to 0-1 for accuracy percentage
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()