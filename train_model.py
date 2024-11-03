import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load the dataset
data = pd.read_csv('./heartdata.csv')  # Replace with your dataset path

# Data Preprocessing
# Check for missing values
print("Missing values in each column:\n", data.isnull().sum())

# Optionally fill or drop missing values
# data.fillna(method='ffill', inplace=True)  # Example of forward filling

# Encode categorical variables if necessary
# If you have categorical variables, use pd.get_dummies or similar methods

# Visualize the distribution of the target variable
sns.countplot(x='target', data=data)
plt.title('Distribution of Heart Disease')
plt.xlabel('Heart Disease (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.show()

# Visualize relationships between features
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Separate features and target variable
X = data.drop('target', axis=1)  # Assuming 'target' is the column name for the label
y = data['target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model to a file
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as heart_disease_model.pkl")




