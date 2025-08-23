# Column names the values of which will be processed by universal encoder
# Sex is encoded separately from these
COLS_TO_ENCODE = ["Pclass", "Embarked", "Family_size", "Name_title"]

# Categories to encode in Pclass column
PCLASS_CATEGORIES_TO_ENCODE = [1, 3]

# Categories to encode in Embarked column
EMBARKED_CATEGORIES_TO_ENCODE = ['C', 'S']

# Categories to encode in Family_size column
FAMILY_SIZE_CATEGORIES_TO_ENCODE = ['Alone', 'Small', 'Medium']

# Categories to encode in Name_title column
TITLE_CATEGORIES_TO_ENCODE = ['Miss', 'Mrs', 'Master', 'Dr']

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

# Path to env file with dir keys
ENV_FILE_PATH = #your path to env file