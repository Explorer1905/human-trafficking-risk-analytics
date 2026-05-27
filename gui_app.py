import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# ----- Load Data -----
df = pd.read_csv("final_combined_dataset.csv")

# ----- Train a simple classification model -----
X = pd.get_dummies(df[['age_group','gender','region_type','state']])
y = df['risk_label'].map({'Low':0, 'High':1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


# ----- GUI -----
root = tk.Tk()
root.title("Human Trafficking Analysis Dashboard")
root.geometry("1200x800")

tab_control = ttk.Notebook(root)

# Dataset Viewer Tab
tab1 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Dataset Viewer')

# Classification Tab
tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text='Classification Analysis')

# Clustering / Patterns Tab
tab3 = ttk.Frame(tab_control)
tab_control.add(tab3, text='Clustering / Patterns')

# Visual Insights Tab
tab4 = ttk.Frame(tab_control)
tab_control.add(tab4, text='Visual Insights')

tab_control.pack(expand=1, fill='both')

# ----- Example: Pie Chart in Visual Insights -----
fig, ax = plt.subplots(figsize=(5,4))
df['risk_label'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
canvas = FigureCanvasTkAgg(fig, master=tab4)
canvas.draw()
canvas.get_tk_widget().pack()

root.mainloop()
