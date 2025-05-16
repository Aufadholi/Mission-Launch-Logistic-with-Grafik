import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, log_loss, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
data = pd.read_csv('mission_launches.csv')
print("Initial data distribution:\n", data['Mission_Status'].value_counts())  # Check distribution of target variable

# Step 2: Process the Data
# Convert 'Price' to numeric and fill missing values with median
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
data['Price'].fillna(data['Price'].median(), inplace=True)

# Convert 'Date' to datetime and drop rows with missing 'Date' values
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.dropna(subset=['Date'], inplace=True)

# Select features and target variable
X = data[['Price', 'Rocket_Status', 'Location', 'Organisation']]
y = data['Mission_Status']

# Step 3: Preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Rocket_Status', 'Location', 'Organisation']),
        ('num', StandardScaler(), ['Price'])
    ]
)

# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTraining set distribution:\n", y_train.value_counts())  # Check distribution in training set
print("\nTest set distribution:\n", y_test.value_counts())   # Check distribution in test set

# Step 5: Create a Pipeline with Preprocessing and Logistic Regression Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='saga', max_iter=10000, multi_class='ovr'))
])

# Step 6: Train the Model
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
# Predict on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report with zero_division set to handle undefined metrics
report = classification_report(y_test, y_pred, zero_division=1)

# Ensure the number of columns in y_prob matches the number of unique classes in y_test
y_test_dummies = pd.get_dummies(y_test)

# Adjust columns of y_prob to match y_test_dummies
y_prob_df = pd.DataFrame(y_prob, columns=model.named_steps['classifier'].classes_)
y_prob_df = y_prob_df.reindex(columns=y_test_dummies.columns, fill_value=0)

# Generate ROC AUC
auc_score = roc_auc_score(y_test_dummies, y_prob_df, multi_class='ovr', average='macro')

# Calculate log-loss
log_loss_value = log_loss(y_test_dummies, y_prob_df)

# Plot ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(y_test_dummies.columns)):
    fpr[i], tpr[i], _ = roc_curve(y_test_dummies.iloc[:, i], y_prob_df.iloc[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(len(y_test_dummies.columns)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {y_test_dummies.columns[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
labels = y_test.unique()  # Get unique labels from y_test
labels.sort()  # Sort labels to ensure they are in the correct order

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Classification Report:\n{report}')
print(f'ROC AUC: {auc_score:.2f}')
print(f'Log-Loss: {log_loss_value:.2f}')

# Additional Debug Information
print("\nPredicted probabilities:\n", y_prob_df.head())  # Check some predicted probabilities
print("\nPredicted classes:\n", y_pred[:10])  # Check some predictions
print("\nTrue classes:\n", y_test[:10].values)  # Check true values for comparison
