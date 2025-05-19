# Column names the values of which will be processed by universal encoder
# Sex and Name_title are encoded separately from these
COLS_TO_ENCODE = ["Pclass", "Embarked", "Family_size"]

# Columns that will be dropped from a DataFrame during data processing
COLS_TO_DROP = ["PassengerId", "Name", "Cabin", "Ticket", "Parch", "SibSp"]

# Numerical columns the nan values of which will be filled by numerical imputer
NUMERICAL_COLS_TO_FILL = ["Age", "Fare"]

# Categorical columns the nan values of which will be filled by categorical imputer
CATEGORICAL_COLS_TO_FILL = ["Embarked"]

# Edges that will be used for binning FamilySize column
FAMILY_SIZE_BINS = [-1, 0, 3, 5, 10]

# Labels to name bin categories in FamilySize column
FAMILY_SIZE_LABELS = ["Alone", "Small", "Medium", "Large"]

# Columns that will be scaled using StandardScaler
STANDARD_SCALE_COLS = ["Age", "Fare"]

# Columns that will be scaled using log10
LOG_SCALE_COLS = ["Fare"]

ENV_FILE_PATH = "C:\\Program_code\\AI\\kaggle_titanic\\.env"