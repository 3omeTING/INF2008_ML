#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 📌 Cell 2: Import Libraries & Load Data
import joblib
import optuna
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_recall_fscore_support, confusion_matrix
)
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import SelectFromModel
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


# Load preprocessed dataset
merged_df = pd.read_csv("Multiclass_dataset/cleaned_merged_dataset.csv")

# Combine Headline + Body text
merged_df["combined_text"] = merged_df["Headline"] + " " + merged_df["articleBody"]

# ✅ Manual Label Mapping (Ensures Correct Alignment)
label_mapping = {"agree": 0, "disagree": 1, "discuss": 2, "unrelated": 3}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
merged_df["Stance"] = merged_df["Stance"].map(label_mapping)


# In[2]:


# 📌 Cell 3: Feature Extraction with TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 3))
X = tfidf.fit_transform(merged_df["combined_text"])
y = merged_df["Stance"]


# In[3]:


# ✅ Train-Test Split using TF-IDF transformed data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Apply ADASYN to TF-IDF transformed features
adasyn = ADASYN(sampling_strategy="not majority", random_state=42)
X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

# ✅ Convert X_train_balanced back to a sparse matrix
X_train_balanced = csr_matrix(X_train_balanced)

# ✅ Check new class distribution
print(pd.Series(y_train_balanced).value_counts())


# In[4]:


# Class Distribution Before ADASYN
plt.figure(figsize=(10, 4))
sns.countplot(x=y_train, palette="Set2")
plt.title("Class Distribution Before ADASYN")
plt.show()

# Class Distribution After ADASYN
plt.figure(figsize=(10, 4))
sns.countplot(x=y_train_balanced, palette="Set1")
plt.title("Class Distribution After ADASYN")
plt.show()


# In[6]:


# 📌 Cell 5: Train Optimized Random Forest
start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=2,
    max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)
train_time = time.time() - start_time
print(f"⏳ Random Forest Training Time: {train_time:.2f} seconds")

# ✅ Feature Selection
feature_selector = SelectFromModel(rf_model, threshold="mean")
X_train_selected = feature_selector.transform(X_train_balanced)
X_test_selected = feature_selector.transform(X_test)

# ✅ Train Again with Selected Features
rf_model.fit(X_train_selected, y_train_balanced)

# ✅ Predict on Test Set
y_pred_rf = rf_model.predict(X_test_selected)

# ✅ Convert predictions back to labels
y_pred_labels_rf = pd.Series(y_pred_rf).map(inverse_label_mapping)
y_test_labels = pd.Series(y_test).map(inverse_label_mapping)

# ✅ Evaluate RF Model
print(f"✅ Random Forest Test Accuracy: {accuracy_score(y_test_labels, y_pred_labels_rf):.4f}")
print(classification_report(y_test_labels, y_pred_labels_rf, target_names=list(label_mapping.keys()), zero_division=0))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[7]:


from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ✅ Create a Validation Set from the Training Data
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_balanced, y_train_balanced, test_size=0.1, random_state=42, stratify=y_train_balanced
)

print(f"Training Size: {X_train_final.shape}, Validation Size: {X_val.shape}")

# ✅ Convert Data to Sparse Matrix (Saves Memory)
X_train_balanced = csr_matrix(X_train_balanced)
X_test = csr_matrix(X_test)

# ✅ Use a Smaller Training Set (Reduce Memory Usage)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_balanced[:15000], y_train_balanced[:15000],  # ✅ Use 15,000 samples
    test_size=0.1, random_state=42, stratify=y_train_balanced[:15000]
)

# ✅ Define Optimized XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=250,  # ✅ Reduce trees
    learning_rate=0.09506484283779507,
    max_depth=8,  # ✅ Reduce depth for efficiency
    min_child_weight=3,
    gamma=0.1777302416003415,
    subsample=0.657855522236526,
    colsample_bytree=0.8988814540637224,
    eval_metric="mlogloss",
    early_stopping_rounds=10,  # ✅ Reduce stopping rounds
    tree_method="hist",  # ✅ Use memory-efficient method
    random_state=42
)

# ✅ Train XGBoost with Early Stopping
start_time = time.time()
xgb_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val, y_val)],  
    verbose=10
)
train_time = time.time() - start_time
print(f"⏳ XGBoost Training Time: {train_time:.2f} seconds")

# ✅ Predict on Test Set
y_pred_xgb = xgb_model.predict(X_test)

# ✅ Evaluate XGBoost Model
print(f"✅ XGBoost Test Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=list(label_mapping.keys()), zero_division=0))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[8]:


# Train Decision Tree
start_time = time.time()

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_balanced, y_train_balanced)

# End timing the training
end_time = time.time()
training_time = end_time - start_time

# Predict on test data (note: we didn't balance the test set)
y_pred = dt_classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision, Recall, F1-score, and Support (per class)
precision, recall, f1_score, support = precision_recall_fscore_support(
    y_test, y_pred, zero_division=0
)

# Classification Report
class_report = classification_report(y_test, y_pred, zero_division=0)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Print Evaluation Metrics
print("\n========== MODEL EVALUATION ==========")
print(f"Training Time: {training_time:.4f} seconds\n")
print(f"Test Accuracy: {accuracy:.4f}\n")

print("\n=========== CLASSIFICATION REPORT ===========\n")
print(class_report)

print("\n=========== CONFUSION MATRIX ===========\n")

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=inverse_label_mapping.values(),
            yticklabels=inverse_label_mapping.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[9]:


# ✅ Train KNN model with 'metric=euclidean', 'weights=uniform', 'n_neighbors=1'
knn = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="euclidean", weights="uniform")

# Start timer for prediction time
start_time = time.time()

# Train the model
knn.fit(X_train_balanced, y_train_balanced)

# Predict on the test set
y_pred = knn.predict(X_test)

print("KNN Multiclass Model")

# End timer for prediction time
end_time = time.time()
prediction_time = end_time - start_time
print(f"Prediction Time: {prediction_time:.4f} seconds")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix for visual inspection
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=inverse_label_mapping.values(), yticklabels=inverse_label_mapping.values())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[10]:


# 📌 Cell 5: Train Optimized Naive Bayes
start_time = time.time()

nb_model = MultinomialNB(
    alpha=1.0,
    fit_prior=True,
    class_prior=None
)

nb_model.fit(X_train_balanced, y_train_balanced)
train_time = time.time() - start_time

# Predictions
y_pred = nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

#Print Evaluation Results
print(f"⌛ Naive Bayes Training Time: {train_time:.2f} seconds\n")
print(f"✅ Naive Bayes Test Accuracy: {accuracy:.4f}\n")

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[11]:


import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Step 1: Define XGBoost Model (Without Early Stopping for Stacking)
xgb_stacking_model = XGBClassifier(
    n_estimators=150, learning_rate=0.09506484283779507, max_depth=8, min_child_weight=3,
    gamma=0.1777302416003415, subsample=0.657855522236526, colsample_bytree=0.8988814540637224, eval_metric="mlogloss",
    tree_method="hist", random_state=42
)

# ✅ Step 2: Train XGBoost Separately for Debugging
print("🔍 Debug: Training XGBoost Separately to Verify Functionality...")
start_xgb = time.time()
xgb_stacking_model.fit(X_train_balanced, y_train_balanced)
xgb_train_time = time.time() - start_xgb
xgb_preds = xgb_stacking_model.predict(X_test)
print(f"✅ XGBoost Standalone Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(f"⏳ XGBoost Training Time: {xgb_train_time:.2f} seconds\n")

# ✅ Step 3: Train Random Forest Separately for Debugging
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=2,
    max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
)
print("🔍 Debug: Training Random Forest Separately to Verify Functionality...")
start_rf = time.time()
rf_model.fit(X_train_balanced, y_train_balanced)
rf_train_time = time.time() - start_rf
rf_preds = rf_model.predict(X_test)
print(f"✅ Random Forest Standalone Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"⏳ Random Forest Training Time: {rf_train_time:.2f} seconds\n")

# ✅ Step 4: Define Stacking Classifier (XGBoost + Random Forest + Logistic Regression)
stacked_model = StackingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_stacking_model)],  
    final_estimator=LogisticRegression()
)

# ✅ Step 5: Train Stacking Model
print("\n🚀 Training Stacking Model...")
start_stacking = time.time()
stacked_model.fit(X_train_balanced, y_train_balanced)
stacking_train_time = time.time() - start_stacking
print(f"✅ Stacking Model Training Completed in {stacking_train_time:.2f} seconds")

# ✅ Step 6: Generate Predictions
print("🔍 Generating Predictions...")
start_prediction = time.time()
stacked_preds = stacked_model.predict(X_test)
prediction_time = time.time() - start_prediction

# ✅ Step 7: Debug: Check if Predictions Exist
print(f"✅ Sample Predictions: {stacked_preds[:10]}")

# ✅ Step 8: Evaluate Stacking Model
print(f"✅ Stacked Model Accuracy: {accuracy_score(y_test, stacked_preds):.4f}")
print(classification_report(y_test, stacked_preds, target_names=list(label_mapping.keys()), zero_division=0))

# ✅ Confusion Matrix
conf_matrix = confusion_matrix(y_test, stacked_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ✅ Step 9: Compute Total Execution Time
total_time = xgb_train_time + rf_train_time + stacking_train_time + prediction_time
print("\n⏳ Training Time Summary:")
print(f"🔹 XGBoost Training Time: {xgb_train_time:.2f} seconds")
print(f"🔹 Random Forest Training Time: {rf_train_time:.2f} seconds")
print(f"🔹 Stacking Model Training Time: {stacking_train_time:.2f} seconds")
print(f"🔹 Prediction Time: {prediction_time:.2f} seconds")
print(f"🚀 Total Time for Training & Prediction: {total_time:.2f} seconds")


# In[42]:


get_ipython().system('jupyter nbconvert --to script Untitled.ipynb --output total-3')


# In[35]:


# ✅ Save Models and Transformers After Training for Later Use
joblib.dump(rf_model, "random_forest.pkl")
joblib.dump(xgb_stacking_model, "xgboost.pkl")
joblib.dump(dt_classifier, "decision_tree.pkl")
joblib.dump(knn, "knn.pkl")
joblib.dump(nb_model, "naive_bayes.pkl")
joblib.dump(stacked_model, "stacked_model.pkl")

# ✅ Save TF-IDF Vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# ✅ Save ADASYN Balanced Data (for potential retraining)
joblib.dump((X_train_balanced, y_train_balanced), "adasyn_balanced_data.pkl")

print("✅ All trained models and required transformers have been saved as .pkl files!")


# In[39]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import time

# 📌 Dictionary to store results
model_results = {}

# 📌 Helper function to evaluate models and store results
def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_pred_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred_time

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    # Store results
    model_results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Training Time (s)": train_time,
        "Prediction Time (s)": pred_time
    }

    # 📌 Print classification report
    print(f"\n🔹 {model_name} Model Performance")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys()), zero_division=0))

    # 📌 Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=inverse_label_mapping.values(),
                yticklabels=inverse_label_mapping.values())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# 📌 Evaluate Random Forest
evaluate_model("Random Forest", rf_model, X_train_selected, y_train_balanced, X_test_selected, y_test)

# 📌 Evaluate Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
evaluate_model("Decision Tree", dt_model, X_train_balanced, y_train_balanced, X_test, y_test)

# 📌 Evaluate K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="euclidean", weights="uniform")
evaluate_model("KNN", knn_model, X_train_balanced, y_train_balanced, X_test, y_test)

# 📌 Evaluate Naive Bayes
nb_model = MultinomialNB(alpha=1.0)
evaluate_model("Naive Bayes", nb_model, X_train_balanced, y_train_balanced, X_test, y_test)

# 📌 Evaluate XGBoost
evaluate_model("XGBoost", xgb_stacking_model, X_train_balanced, y_train_balanced, X_test, y_test)

# 📌 Evaluate Stacking Model
evaluate_model("Stacking Model", stacked_model, X_train_balanced, y_train_balanced, X_test, y_test)

# 📌 Convert results to DataFrame
results_df = pd.DataFrame(model_results).T
print("\n🔍 Model Performance Comparison:")
print(results_df)

# 📌 Plot Accuracy Comparison
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x=results_df.index, y="Accuracy")
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.show()

# 📌 Plot Training Time Comparison
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x=results_df.index, y="Training Time (s)")
plt.xticks(rotation=45)
plt.title("Model Training Time Comparison")
plt.ylabel("Training Time (seconds)")
plt.show()


# In[40]:





# In[ ]:




