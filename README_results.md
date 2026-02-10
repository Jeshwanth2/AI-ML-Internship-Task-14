# Model Comparison for Titanic Dataset

## Approach
1. **Preprocessing**: Filled missing 'Age' with median and 'Embarked' with mode. Dropped 'Cabin', 'Name', 'Ticket'. Encoded 'Sex' and 'Embarked' using LabelEncoder. Scaled features using StandardScaler.
2. **Models Evaluated**: Logistic Regression, Decision Tree, Random Forest, SVM.
3. **Selection Metric**: F1 Score (harmonic mean of precision and recall).

## Results
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.8045 | 0.7826 | 0.7297 | 0.7552 |
| Decision Tree | 0.7821 | 0.7273 | 0.7568 | 0.7417 |
| Random Forest | 0.8156 | 0.7971 | 0.7432 | 0.7692 |
| SVM | 0.8156 | 0.8154 | 0.7162 | 0.7626 |

## Best Model
**Random Forest** was selected as the best model because it achieved the highest F1 Score (0.7692) and demonstrated balanced performance on the test set.

## Overfitting Analysis
- **Decision Tree** and **Random Forest** showed higher training accuracy compared to test accuracy, suggesting some overfitting.
- **Logistic Regression** and **SVM** showed better generalization but might have lower peak performance depending on the interactions.
