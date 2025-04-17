import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Load data
df = pd.read_csv('data.csv')

# Prepare data
df = df.drop(['StudentID', 'GradeClass'], axis=1)
X = df.drop('GPA', axis=1)
y = df['GPA']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['Age', 'StudyTimeWeekly', 'Absences']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train XGBoost model (best performing in our tests)
model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# Save model and scaler
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')