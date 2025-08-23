# Titanic - Machine Learning from Disaster 🚢  
My solution for the [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic), which predicts passenger survival using gradient boosting with **XGBoost** and a custom preprocessing pipeline.  

---

## 📊 Model Overview  

- **Algorithm:** XGBoost (`XGBClassifier`)  
- **Objective:** Binary classification (`binary:logistic`)  
- **Random State:** 42 (for reproducibility)  

### Hyperparameters  
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=280,
    max_depth=2,
    min_child_weight=20,
    reg_alpha=1.2,
    reg_lambda=2.2,
    gamma=0.5,
    learning_rate=1.0,
    colsample_bytree=0.5,
    objective="binary:logistic",
    random_state=42,
    n_jobs=-1
)
```

---

## 🛠️ Custom Preprocessing Pipeline  

The core of this solution is a **custom-built preprocessing pipeline** implemented in `preprocess.py`.  
It ensures **consistent transformations** across training, validation, and inference by persisting fitted transformers (`joblib`).  

### 🔄 Pipeline Steps  

1. **Missing Values Handling**  
   - Numerical columns → imputed with **mean** (`SimpleImputer(strategy="mean")`)  
   - Categorical columns → imputed with **most frequent value** (`SimpleImputer(strategy="most_frequent")`)  
   - Imputers are saved to disk and reloaded for inference.  

2. **Feature Engineering**  
   - `Family_size` → computed as `SibSp + Parch`  
   - `Numeric_ticket` → 1 if `Ticket` consists only of digits, else 0  
   - `Name_title` → prefix extracted from `Name` (e.g., *Mr, Mrs, Miss, Master*)  

3. **Binning**  
   - `Family_size` is grouped into bins (e.g., `Alone`, `Small`, `Medium`, `Large`)  
   - Uses predefined bins and labels (`FAMILY_SIZE_BINS`, `FAMILY_SIZE_LABELS`) from `config.py`  

4. **Encoding**  
   - `Sex` → binary encoding (`male=1, female=0`)  
   - `Pclass`, `Embarked`, `Family_size`, and `Name_title` → encoded with `OneHotEncoder`  
     - Uses **fixed category lists** from `config.py` to ensure consistent columns  
     - Encoder is persisted and reused across runs  

5. **Scaling**  
   - `StandardScaler` → applied to continuous features (e.g., `Age`, `Fare`)  
   - `Log Scaling` → applied to skewed features (`np.log10(x + 1)`)  
   - Scaler is persisted for inference consistency  

6. **Column Dropping**  
   - Drops irrelevant or high-cardinality features (`Name`, `Ticket`, `Cabin`, etc.) as specified in `COLS_TO_DROP`  

### 📐 Final Transformation  

- After preprocessing, the dataset is returned as a **clean NumPy array (`np.float32`)**  
- If `split=True`, the function also returns labels (`Survived`) separately  

```python
X, y = preprocess_data(df, split=True, label="Survived")
```

---

## 📈 Performance  

The model was evaluated on training and validation sets using multiple metrics:  

| Metric     | Train Set | Validation Set |
|------------|-----------|----------------|
| **F1 Score**   | 0.7448    | 0.7214         |
| **Accuracy**   | 0.8208    | 0.8014         |
| **ROC AUC**    | 0.8740    | 0.8400         |
| **Log Loss**   | 0.4171    | 0.4616         |

- **Kaggle Public Score:** **0.79425**

---