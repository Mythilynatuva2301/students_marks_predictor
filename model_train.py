import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Step 1: Create data
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Marks': [10, 20, 30, 40, 50, 55, 65, 75, 85]
}
df = pd.DataFrame(data)

# Step 2: Train model
X = df[['Hours']]
y = df['Marks']
model = LinearRegression()
model.fit(X, y)

# Step 3: Save the model to .pkl file
joblib.dump(model, 'student_mark_model.pkl')
print("✅ Model saved as student_mark_model.pkl")
