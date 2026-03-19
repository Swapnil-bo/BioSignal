import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

X = np.load('data/processed/X.npy')
y = np.load('data/processed/y.npy')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Models
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
rf  = RandomForestClassifier(n_estimators=200, random_state=42)

# Ensemble
ensemble = VotingClassifier(
    estimators=[('svm', svm), ('rf', rf)],
    voting='soft'
)

# Cross-validation on full data
cv_scores = cross_val_score(ensemble, X_scaled, y, cv=5)
print(f"CV Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Final fit on train set, evaluate on test
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['rest', 'left fist', 'right fist']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(ensemble, 'models/ensemble.pkl')
joblib.dump(scaler,   'models/scaler.pkl')
print("\nSaved ensemble.pkl and scaler.pkl to models/")
