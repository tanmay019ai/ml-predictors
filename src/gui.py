import tkinter as tk
import joblib
from scipy.sparse import hstack

# Load trained model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict():
    desc = entry_desc.get()
    amt = entry_amount.get()

    if not desc or not amt:
        result_label.config(text="Please enter all fields")
        return

    try:
        amt = float(amt)
    except ValueError:
        result_label.config(text="Amount must be a number")
        return

    text_vec = vectorizer.transform([desc])
    combined = hstack((text_vec, [[amt]]))
    category = model.predict(combined)[0]

    result_label.config(text=f"Predicted Category: {category}")

# Create window
root = tk.Tk()
root.title("Expense Category Predictor")
root.geometry("400x250")

# UI Elements
tk.Label(root, text="Expense Description").pack(pady=5)
entry_desc = tk.Entry(root, width=40)
entry_desc.pack()

tk.Label(root, text="Amount").pack(pady=5)
entry_amount = tk.Entry(root, width=20)
entry_amount.pack()

tk.Button(root, text="Predict", command=predict).pack(pady=15)

result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.pack()

# Start GUI
root.mainloop()
