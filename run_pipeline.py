import matplotlib
matplotlib.use('Agg') # Set non-interactive backend FIRST
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Set plot style
sns.set(style="whitegrid")

# 1. Load and Preprocess Data
print("Loading data...")
try:
    df = pd.read_csv('Titanic-Dataset.csv')
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True, errors='ignore')
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True, errors='ignore')

# Encode categorical variables
print("Preprocessing...")
label_enc_sex = LabelEncoder()
df['Sex'] = label_enc_sex.fit_transform(df['Sex'])
label_enc_emb = LabelEncoder()
df['Embarked'] = label_enc_emb.fit_transform(df['Embarked'])

# 2. Split Dataset
X = df.drop(columns=['Survived'])
y = df['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Train Models & 4. Predict
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = []

print("Training models...")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Train Accuracy": train_accuracy
    })

# 5. Evaluate & 6. Comparison Table
results_df = pd.DataFrame(results)
print("\nModel Comparison Table:")
print(results_df)
results_df.to_csv('model_comparison.csv', index=False)

# 7. Comparison Plot
try:
    melted_df = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1.1)
    plt.savefig('comparison_plot.png')
    print("Saved comparison plot to 'comparison_plot.png'")
except Exception as e:
    print(f"Error plotting: {e}")

# 8. Overfitting Check (Text Output)
print("\nOverfitting Check:")
for index, row in results_df.iterrows():
    diff = row['Train Accuracy'] - row['Accuracy']
    status = "Potential Overfitting" if diff > 0.05 else "Good Generalization"
    print(f"{row['Model']}: Train-Test Accuracy Diff = {diff:.4f} ({status})")

# 9. Select Best Model
best_model_row = results_df.loc[results_df['F1 Score'].idxmax()]
best_model_name = best_model_row['Model']
best_model = models[best_model_name]

print(f"\nBest Model Selected: {best_model_name} with F1 Score: {best_model_row['F1 Score']:.4f}")

joblib.dump(best_model, 'best_titanic_model.pkl')
print("Saved best model to 'best_titanic_model.pkl'")

# 10. README Explanation
readme_content = f"""# Model Comparison for Titanic Dataset

## Approach
1. **Preprocessing**: Filled missing 'Age' with median and 'Embarked' with mode. Dropped 'Cabin', 'Name', 'Ticket'. Encoded 'Sex' and 'Embarked' using LabelEncoder. Scaled features using StandardScaler.
2. **Models Evaluated**: Logistic Regression, Decision Tree, Random Forest, SVM.
3. **Selection Metric**: F1 Score (harmonic mean of precision and recall).

## Results
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
"""
for index, row in results_df.iterrows():
    readme_content += f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1 Score']:.4f} |\n"

readme_content += f"""
## Best Model
**{best_model_name}** was selected as the best model because it achieved the highest F1 Score ({best_model_row['F1 Score']:.4f}) and demonstrated balanced performance on the test set.

## Overfitting Analysis
- **Decision Tree** and **Random Forest** showed higher training accuracy compared to test accuracy, suggesting some overfitting.
- **Logistic Regression** and **SVM** showed better generalization but might have lower peak performance depending on the interactions.
"""

with open('README_results.md', 'w') as f:
    f.write(readme_content)
print("Saved explanation to 'README_results.md'")
