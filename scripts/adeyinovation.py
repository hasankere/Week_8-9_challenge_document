import pandas as pd
import os
# Define dataset location
data = r"C:\\Users\\Hasan\\Desktop\\data science folder"

# Load datasets
fraud_data_path = os.path.join(data, "Fraud_Data.csv")
ip_data_path = os.path.join(data, "IpAddress_to_Country.csv")
credit_card_path = os.path.join(data, "creditcard.csv")

fraud_data = pd.read_csv(fraud_data_path)
ip_data = pd.read_csv(ip_data_path)
credit_card_data = pd.read_csv(credit_card_path)

# Convert IP address to numeric for merging
fraud_data["ip_address"] = pd.to_numeric(fraud_data["ip_address"])
ip_data["lower_bound_ip_address"] = pd.to_numeric(ip_data["lower_bound_ip_address"])
ip_data["upper_bound_ip_address"] = pd.to_numeric(ip_data["upper_bound_ip_address"])

# Merge Fraud_Data with IpAddress_to_Country based on IP range
merged_fraud_data = fraud_data.merge(
    ip_data, 
    how="left", 
    left_on="ip_address", 
    right_on="lower_bound_ip_address"
)

# Drop unnecessary IP range columns
merged_fraud_data.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"], inplace=True)

# Print dataset information
print("Fraud Data + IP Merged Dataset Info:")
print(merged_fraud_data.info())
print(merged_fraud_data.head())

print("\nCredit Card Data Info:")
print(credit_card_data.info())
print(credit_card_data.head())

# Function to check missing values
def check_missing_values(df, name):
    print(f"\n🔍 Missing Values in {name} Dataset:")
    print(df.isnull().sum())

# Check missing values in each dataset
check_missing_values(fraud_data, "Fraud Data")
check_missing_values(ip_data, "IP Address Data")
check_missing_values(credit_card_data, "Credit Card Data")

# Handling missing values
# 1. Fraud Data
fraud_data.fillna({"browser": "Unknown", "source": "Unknown"}, inplace=True)  # Fill categorical missing values
fraud_data.dropna(inplace=True)  # Drop rows with missing numerical values

# 2. IP Address Data
ip_data.dropna(inplace=True)  # Drop missing rows (important for merging)

# 3. Credit Card Data
credit_card_data.fillna(credit_card_data.median(), inplace=True)  # Impute missing numerical values with median

# Verify missing values after handling
print("\n✅ Missing Values After Handling:")
check_missing_values(fraud_data, "Fraud Data")
check_missing_values(ip_data, "IP Address Data")
check_missing_values(credit_card_data, "Credit Card Data")

# Save cleaned datasets (optional)
fraud_data.to_csv(os.path.join(data, "Fraud_Data_Cleaned.csv"), index=False)
ip_data.to_csv(os.path.join(data, "IpAddress_to_Country_Cleaned.csv"), index=False)
credit_card_data.to_csv(os.path.join(data, "creditcard_Cleaned.csv"), index=False)

print("\n🎯 Data Cleaning Completed. Cleaned files saved!")

### 1️⃣ Remove Duplicates
def remove_duplicates(df, name):
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    print(f"\n🧹 Removed {before - after} duplicate rows from {name} dataset.")
    return df

fraud_data = remove_duplicates(fraud_data, "Fraud Data")
ip_data = remove_duplicates(ip_data, "IP Address Data")
credit_card_data = remove_duplicates(credit_card_data, "Credit Card Data")

### 2️⃣ Correct Data Types
# Convert timestamps to datetime
fraud_data["signup_time"] = pd.to_datetime(fraud_data["signup_time"])
fraud_data["purchase_time"] = pd.to_datetime(fraud_data["purchase_time"])

# Convert IP address fields to numeric
fraud_data["ip_address"] = fraud_data["ip_address"].astype(int)
ip_data["lower_bound_ip_address"] = ip_data["lower_bound_ip_address"].astype(int)
ip_data["upper_bound_ip_address"] = ip_data["upper_bound_ip_address"].astype(int)

# Convert categorical fields
fraud_data["source"] = fraud_data["source"].astype("category")
fraud_data["browser"] = fraud_data["browser"].astype("category")
fraud_data["sex"] = fraud_data["sex"].astype("category")

# Convert fraud class labels to integers (optional, already int but ensuring consistency)
fraud_data["class"] = fraud_data["class"].astype(int)
credit_card_data["Class"] = credit_card_data["Class"].astype(int)

# Convert PCA features (V1 to V28) and amount to float (if needed)
for col in credit_card_data.columns:
    if col.startswith("V") or col == "Amount":
        credit_card_data[col] = credit_card_data[col].astype(float)

### ✅ Verify Changes
print("\n📊 Data Types After Cleaning:")
print("\nFraud Data:")
print(fraud_data.dtypes)

print("\nIP Address Data:")
print(ip_data.dtypes)

print("\nCredit Card Data:")
print(credit_card_data.dtypes)

# Save cleaned datasets
fraud_data.to_csv(os.path.join(data, "Fraud_Data_Final.csv"), index=False)
ip_data.to_csv(os.path.join(data, "IpAddress_to_Country_Final.csv"), index=False)
credit_card_data.to_csv(os.path.join(data, "creditcard_Final.csv"), index=False)

print("\n🎯 Data Cleaning Completed! Final cleaned files are saved.")
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set(style="whitegrid")

# Define dataset location
data = r"C:\\Users\\Hasan\\Desktop\\data science folder"

# Load cleaned datasets
fraud_data = pd.read_csv(os.path.join(data, "Fraud_Data_Final.csv"))
credit_card_data = pd.read_csv(os.path.join(data, "creditcard_Final.csv"))

### 1️⃣ Univariate Analysis - Distribution of Single Variables

# Fraud Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="class", data=fraud_data, palette="coolwarm")
plt.title("Fraud vs Non-Fraud Transactions (E-commerce)")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=credit_card_data, palette="coolwarm")
plt.title("Fraud vs Non-Fraud Transactions (Credit Card)")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

# Distribution of Purchase Values (E-commerce)
plt.figure(figsize=(8, 5))
sns.histplot(fraud_data["purchase_value"], bins=50, kde=True, color="blue")
plt.title("Distribution of Purchase Value")
plt.xlabel("Purchase Value ($)")
plt.ylabel("Frequency")
plt.show()

# Distribution of Transaction Amount (Credit Card)
plt.figure(figsize=(8, 5))
sns.histplot(credit_card_data["Amount"], bins=50, kde=True, color="green")
plt.title("Distribution of Credit Card Transaction Amount")
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Frequency")
plt.show()

# Age Distribution of Users
plt.figure(figsize=(8, 5))
sns.histplot(fraud_data["age"], bins=30, kde=True, color="purple")
plt.title("Distribution of User Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

### 2️⃣ Bivariate Analysis - Relationship Between Two Variables

# Purchase Value vs Fraudulent Transactions
plt.figure(figsize=(8, 5))
sns.boxplot(x="class", y="purchase_value", data=fraud_data, palette="coolwarm")
plt.title("Purchase Value vs Fraudulent Transactions")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Purchase Value ($)")
plt.show()

# Age vs Fraudulent Transactions
plt.figure(figsize=(8, 5))
sns.boxplot(x="class", y="age", data=fraud_data, palette="coolwarm")
plt.title("Age vs Fraudulent Transactions")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Age")
plt.show()

# Credit Card Amount vs Fraudulent Transactions
plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Amount", data=credit_card_data, palette="coolwarm")
plt.title("Transaction Amount vs Fraudulent Transactions (Credit Card)")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Transaction Amount ($)")
plt.show()

print("\n🎯 EDA Completed! Insights Generated.")
# Compute correlation matrix
corr_matrix = credit_card_data.corr()
# Set Seaborn style
sns.set(style="whitegrid")

### 1️⃣ Scatter Plot: Purchase Value vs Fraud (E-commerce Data)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=fraud_data["purchase_value"], y=fraud_data["age"], hue=fraud_data["class"], palette="coolwarm", alpha=0.6)
plt.title("📊 Purchase Value vs Age (E-commerce Fraud)")
plt.xlabel("Purchase Value ($)")
plt.ylabel("Age")
plt.legend(title="Fraud (1 = Yes, 0 = No)")
plt.show()

### 2️⃣ Scatter Plot: Transaction Amount vs Fraud (Credit Card Data)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=credit_card_data["Amount"], y=credit_card_data["Time"], hue=credit_card_data["Class"], palette="coolwarm", alpha=0.6)
plt.title("📊 Transaction Amount vs Time (Credit Card Fraud)")
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Time (Seconds)")
plt.legend(title="Fraud (1 = Yes, 0 = No)")
plt.show()

### 3️⃣ Scatter Plot: Age vs Fraud (E-commerce Data)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=fraud_data["age"], y=fraud_data["purchase_value"], hue=fraud_data["class"], palette="coolwarm", alpha=0.6)
plt.title("📊 Age vs Purchase Value (E-commerce Fraud)")
plt.xlabel("Age")
plt.ylabel("Purchase Value ($)")
plt.legend(title="Fraud (1 = Yes, 0 = No)")
plt.show()

print("\n🎯 EDA Completed! Insights Generated.")
# Focus on correlation with fraud class
fraud_correlation = corr_matrix["Class"].sort_values(ascending=False)
print("\n📊 Features Most Correlated with Fraud:")
print(fraud_correlation.head(10))  # Show top 10 most correlated features
print("\n🚀 Insights Generated! See the heatmap for more details.")
import ipaddress

# Define the dataset location
data = r"C:\\Users\\Hasan\\Desktop\\data science folder"

# Load the cleaned datasets
fraud_data = pd.read_csv(os.path.join(data, "Fraud_Data_Final.csv"))
credit_card_data = pd.read_csv(os.path.join(data, "creditcard_Final.csv"))
ip_address_data = pd.read_csv(os.path.join(data, "IpAddress_to_Country_Final.csv"))
# Function to convert an IP to an integer
def ip_to_int(ip):
    return int(ipaddress.ip_address(ip))

# Convert IP addresses in fraud_data to integers
fraud_data['IP_int'] = fraud_data['ip_address'].apply(ip_to_int)

# Convert IP ranges in ip_address_data to integers
ip_address_data['lower_bound_ip_int'] = ip_address_data['lower_bound_ip_address'].apply(ip_to_int)
ip_address_data['upper_bound_ip_int'] = ip_address_data['upper_bound_ip_address'].apply(ip_to_int)
import pandas as pd
import os
import ipaddress

# Define dataset location
data = r"C:\\Users\\Hasan\\Desktop\\data science folder"

# Load the cleaned datasets
fraud_data = pd.read_csv(os.path.join(data, "Fraud_Data_Final.csv"))
ip_address_data = pd.read_csv(os.path.join(data, "IpAddress_to_Country_Final.csv"))

# Function to convert an IP to an integer
def ip_to_int(ip):
    return int(ipaddress.ip_address(ip))

# Convert IPs to integers in fraud_data
fraud_data['IP_int'] = fraud_data['ip_address'].apply(ip_to_int)

# Convert IP ranges to integers in ip_address_data
ip_address_data['lower_bound_ip_int'] = ip_address_data['lower_bound_ip_address'].apply(ip_to_int)
ip_address_data['upper_bound_ip_int'] = ip_address_data['upper_bound_ip_address'].apply(ip_to_int)

# Sort ip_address_data by lower_bound_ip_int for efficient range lookup
ip_address_data = ip_address_data.sort_values(by='lower_bound_ip_int')

# Function to find the country for a given IP integer efficiently
def find_country(ip_int):
    # Use a more efficient range comparison (binary search approach)
    matched_rows = ip_address_data[(ip_address_data['lower_bound_ip_int'] <= ip_int) & 
                                    (ip_address_data['upper_bound_ip_int'] >= ip_int)]
    if not matched_rows.empty:
        return matched_rows['country'].iloc[0]
    return None  # Return None if no country is found

# Apply the function to fraud_data using a more efficient vectorized approach
fraud_data['country'] = fraud_data['IP_int'].apply(find_country)

# Display the merged data
print(fraud_data.head())

# Optional: Save the merged data to a new CSV
fraud_data.to_csv(os.path.join(data, "Merged_Fraud_Data_with_Country.csv"), index=False)
# Convert 'purchase_time' to datetime format
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
# Extract hour of the day and day of the week
fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
# Calculate transaction frequency per user
fraud_data['transaction_frequency'] = fraud_data.groupby('user_id')['user_id'].transform('count')

# Define dataset location
data = r"C:\\Users\\Hasan\\Desktop\\data science folder"

# Load the cleaned fraud data
fraud_data = pd.read_csv(os.path.join(data, "Fraud_Data_Final.csv"))

# Convert 'purchase_time' to datetime format
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])

# Extract hour of the day and day of the week
fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek

# Calculate transaction frequency per user
fraud_data['transaction_frequency'] = fraud_data.groupby('user_id')['user_id'].transform('count')

# Sort the data by user_id and purchase_time
fraud_data = fraud_data.sort_values(by=['user_id', 'purchase_time'])

# Debugging: Print a sample of the data to check if multiple transactions exist for users
print(fraud_data[['user_id', 'purchase_time', 'transaction_frequency']].head(20))

# Calculate the time difference between consecutive transactions (in seconds)
fraud_data['transaction_velocity'] = fraud_data.groupby('user_id')['purchase_time'].diff().dt.total_seconds()

# Handle NaN values in 'transaction_velocity' for users with a single transaction
fraud_data['transaction_velocity'] = fraud_data['transaction_velocity'].fillna(0)

# Display the updated fraud data with the new features
print(fraud_data[['user_id', 'purchase_time', 'hour_of_day', 'day_of_week', 'transaction_frequency', 'transaction_velocity']].head(20))
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Create a scaler object for Min-Max scaling (Normalization)
min_max_scaler = MinMaxScaler()

# Create a scaler object for Standard scaling (Standardization)
standard_scaler = StandardScaler()

# List of numerical columns that we want to scale/normalize
numerical_columns = ['purchase_value', 'transaction_frequency', 'transaction_velocity']

# Apply Min-Max Scaling (Normalization)
fraud_data[numerical_columns] = min_max_scaler.fit_transform(fraud_data[numerical_columns])

# Alternatively, apply Standard Scaling (Standardization)
# fraud_data[numerical_columns] = standard_scaler.fit_transform(fraud_data[numerical_columns])

# Check the first few rows after normalization
print(fraud_data[numerical_columns].head())
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Create a scaler object for Min-Max scaling (Normalization)
min_max_scaler = MinMaxScaler()

# Create a scaler object for Standard scaling (Standardization)
standard_scaler = StandardScaler()

# List of numerical columns that we want to scale/normalize
numerical_columns = ['purchase_value', 'transaction_frequency', 'transaction_velocity']

# Apply Min-Max Scaling (Normalization)
fraud_data[numerical_columns] = min_max_scaler.fit_transform(fraud_data[numerical_columns])

# Alternatively, apply Standard Scaling (Standardization)
# fraud_data[numerical_columns] = standard_scaler.fit_transform(fraud_data[numerical_columns])

# Apply Label Encoding for binary categorical feature 'sex'
label_encoder = LabelEncoder()
fraud_data['sex'] = label_encoder.fit_transform(fraud_data['sex'])

# Since 'source' is already one-hot encoded as 'source_Direct' and 'source_SEO',
# we don't need to apply pd.get_dummies again to the 'source' column.

# Check the transformed data
print(fraud_data.head())
from sklearn.model_selection import train_test_split
#model building and training
# 1. Feature and Target Separation for Fraud_Data
X_fraud = fraud_data.drop(columns=['class'])  # Features for Fraud_Data
y_fraud = fraud_data['class']  # Target for Fraud_Data

# 2. Feature and Target Separation for creditcard
X_creditcard = credit_card_data.drop(columns=['Class'])  # Features for creditcard
y_creditcard = credit_card_data['Class']  # Target for creditcard

# 3. Train-Test Split for Fraud_Data
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
    X_fraud, y_fraud, test_size=0.2, random_state=42)

# 4. Train-Test Split for creditcard
X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(
    X_creditcard, y_creditcard, test_size=0.2, random_state=42)

# Check the shapes of the split data
print(f"Fraud Data Train shape: {X_fraud_train.shape}, Test shape: {X_fraud_test.shape}")
print(f"Credit Card Data Train shape: {X_creditcard_train.shape}, Test shape: {X_creditcard_test.shape}")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Sample code to load your data
# X_fraud = pd.read_csv('fraud_data.csv')
# y_fraud = X_fraud.pop('target')  # assuming 'target' is the column you want to predict
# X_creditcard = pd.read_csv('creditcard_data.csv')
# y_creditcard = X_creditcard.pop('target')

# Preprocessing function to handle date and categorical columns
def preprocess_data(X):
    # Handle date columns
    if 'timestamp' in X.columns:
        # Convert 'timestamp' to datetime
        X['timestamp'] = pd.to_datetime(X['timestamp'], errors='coerce')  # Coerce invalid dates to NaT
        
        # Extract relevant time features from 'timestamp'
        X['year'] = X['timestamp'].dt.year
        X['month'] = X['timestamp'].dt.month
        X['day'] = X['timestamp'].dt.day
        X['hour'] = X['timestamp'].dt.hour
        X['minute'] = X['timestamp'].dt.minute
        
        # Drop the original timestamp column after extracting features
        X.drop(columns=['timestamp'], inplace=True)

    # Convert categorical columns to numeric using LabelEncoder
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    # Ensure there are no missing values (you can handle this differently based on your dataset)
    X.fillna(0, inplace=True)  # Filling missing values with 0 (can change this to impute with mean/median)

    # Ensure no datetime columns remain (if any other datetime columns exist, they should also be handled)
    for col in X.select_dtypes(include=['datetime64']).columns:
        X.drop(columns=[col], inplace=True)
    
    return X

# Apply preprocessing to the datasets
X_fraud = preprocess_data(X_fraud)
X_creditcard = preprocess_data(X_creditcard)

# Train-test split
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.3, random_state=42)
X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(X_creditcard, y_creditcard, test_size=0.3, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}

# Function to train and evaluate models
def train_and_evaluate_model(X_train, X_test, y_train, y_test, models):
    results = {}
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Evaluate performance
        accuracy = model.score(X_test, y_test)
        auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # AUC-ROC score
        
        results[model_name] = {
            "accuracy": accuracy,
            "AUC-ROC": auc_roc,
            "classification_report": classification_report(y_test, y_pred)
        }
    return results

# Train and evaluate on Fraud Data
results_fraud_data = train_and_evaluate_model(X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, models)

# Train and evaluate on Credit Card Data
results_creditcard_data = train_and_evaluate_model(X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test, models)

# Print results for comparison
print("Fraud Data Results:\n")
for model_name, result in results_fraud_data.items():
    print(f"{model_name}:\n Accuracy: {result['accuracy']}\n AUC-ROC: {result['AUC-ROC']}")
    print(result['classification_report'])
    print("-" * 60)

print("\nCredit Card Data Results:\n")
for model_name, result in results_creditcard_data.items():
    print(f"{model_name}:\n Accuracy: {result['accuracy']}\n AUC-ROC: {result['AUC-ROC']}")
    print(result['classification_report'])
    print("-" * 60)
import mlflow
import mlflow.sklearn
# Sample code to load your data (make sure to load your datasets)
# X_fraud = pd.read_csv('fraud_data.csv')
# y_fraud = X_fraud.pop('target')  # assuming 'target' is the column you want to predict
# X_creditcard = pd.read_csv('creditcard_data.csv')
# y_creditcard = X_creditcard.pop('target')

# Preprocessing function to handle date and categorical columns
def preprocess_data(X):
    # Handle date columns
    if 'timestamp' in X.columns:
        # Convert 'timestamp' to datetime
        X['timestamp'] = pd.to_datetime(X['timestamp'], errors='coerce')  # Coerce invalid dates to NaT
        
        # Extract relevant time features from 'timestamp'
        X['year'] = X['timestamp'].dt.year
        X['month'] = X['timestamp'].dt.month
        X['day'] = X['timestamp'].dt.day
        X['hour'] = X['timestamp'].dt.hour
        X['minute'] = X['timestamp'].dt.minute
        
        # Drop the original timestamp column after extracting features
        X.drop(columns=['timestamp'], inplace=True)

    # Convert categorical columns to numeric using LabelEncoder
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    # Ensure there are no missing values (you can handle this differently based on your dataset)
    X.fillna(0, inplace=True)  # Filling missing values with 0 (can change this to impute with mean/median)

    # Ensure no datetime columns remain (if any other datetime columns exist, they should also be handled)
    for col in X.select_dtypes(include=['datetime64']).columns:
        X.drop(columns=[col], inplace=True)
    
    return X

# Apply preprocessing to the datasets
X_fraud = preprocess_data(X_fraud)
X_creditcard = preprocess_data(X_creditcard)

# Train-test split
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.3, random_state=42)
X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(X_creditcard, y_creditcard, test_size=0.3, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}

# Function to train and evaluate models with MLflow (nested runs for each model)
def train_and_evaluate_model_with_mlflow(X_train, X_test, y_train, y_test, models, dataset_name):
    results = {}
    
    # Start MLflow run to track the experiment
    with mlflow.start_run():
        mlflow.log_param("dataset_name", dataset_name)  # Log dataset name

        for model_name, model in models.items():
            # Use a nested run for each model to avoid parameter conflicts
            with mlflow.start_run(nested=True):
                mlflow.log_param("model", model_name)  # Log model name
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Predict on the test set
                y_pred = model.predict(X_test)
                
                # Evaluate performance
                accuracy = model.score(X_test, y_test)
                auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # AUC-ROC score
                
                # Log metrics
                mlflow.log_metric(f"{model_name}_accuracy", accuracy)
                mlflow.log_metric(f"{model_name}_AUC_ROC", auc_roc)
                
                # Save model
                mlflow.sklearn.log_model(model, f"{model_name}_model")
                
                # Store the results in a dictionary for later display
                results[model_name] = {
                    "accuracy": accuracy,
                    "AUC-ROC": auc_roc,
                    "classification_report": classification_report(y_test, y_pred)
                }
    return results

# Train and evaluate on Fraud Data with MLflow tracking
results_fraud_data = train_and_evaluate_model_with_mlflow(X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, models, "fraud_data")

# Train and evaluate on Credit Card Data with MLflow tracking
results_creditcard_data = train_and_evaluate_model_with_mlflow(X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test, models, "creditcard_data")

# Print results for comparison
print("Fraud Data Results:\n")
for model_name, result in results_fraud_data.items():
    print(f"{model_name}:\n Accuracy: {result['accuracy']}\n AUC-ROC: {result['AUC-ROC']}")
    print(result['classification_report'])
    print("-" * 60)

print("\nCredit Card Data Results:\n")
for model_name, result in results_creditcard_data.items():
    print(f"{model_name}:\n Accuracy: {result['accuracy']}\n AUC-ROC: {result['AUC-ROC']}")
    print(result['classification_report'])
    print("-" * 60)
