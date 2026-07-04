import pandas as pd
import joblib
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
# Load Dataset
data = pd.read_csv("House_Rent_dataset.csv")
# Separate Features and Target
X = data.drop(
columns=[
"Rent",
"Posted On",
"Floor",
"Area Locality"
]
)
y = data["Rent"]
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.25,
random_state=42
)
# Build Pipeline
pipeline = Pipeline([
("encoder", ce.LeaveOneOutEncoder()),
("scaler", MinMaxScaler()),
("model", LinearRegression())
])
# Train
pipeline.fit(X_train, y_train)
# Evaluate
y_pred = pipeline.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
# Save Pipeline
joblib.dump(pipeline, "house_rent_pipeline.pkl")
# Load Pipeline
loaded_pipeline = joblib.load("house_rent_pipeline.pkl")
# Predict a New House
new_house = pd.DataFrame({
"BHK": [2],
"Size": [1200],
"Area Type": ["Super Area"],
"City": ["Delhi"],
"Furnishing Status": ["Semi-Furnished"],
"Tenant Preferred": ["Family"],
"Bathroom": [2],
"Point of Contact": ["Contact Owner"]
})
prediction = loaded_pipeline.predict(new_house)
print(f"Predicted Rent: ₹{prediction[0]:,.0f}")