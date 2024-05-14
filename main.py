import tkinter as tk
import numpy as np
from joblib import load
from tkinter import filedialog
import pandas as pd
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

svm = load("svm_model")
target = ['malignant', 'benign']

# Prediction par formulaire
def open_form_prediction():
    def on_closing(window):
        window.destroy()
        root.destroy()

    def validate_float(P):
        if P == '' or (P.replace('.', '', 1).isdigit() and P.count('.') <= 1):
            return True
        return False

    def back_to_main():
        form_prediction_window.destroy()
        root.deiconify()

    def fetch_entries(entries, text_area):
        float_values = []

        for field, entry in entries.items():
            value = entry.get()
            try:
                float_value = float(value)
                float_values.append(float_value)
            except ValueError:
                text_area.configure(state='normal')
                text_area.delete('1.0', tk.END)
                text_area.insert(tk.END, f"Error: {field} contains invalid data.\n")
                text_area.configure(state='disabled')
                return

        data_array = np.array(float_values)
        data_array = data_array.reshape((1, -1))
        text_area.configure(state='normal')
        text_area.delete('1.0', tk.END)
        text_area.insert(tk.END, f"results : {target[svm.predict(data_array)[0]]}")
        text_area.configure(state='disabled')
    def random():
        df = pd.read_csv("./data/breat_cancer.csv")
        random_row = df.sample(n=1)
        random_row = random_row.to_dict(orient="records")[0]
        for field, entry in entries.items():
            if field in random_row:
                entry.delete(0, tk.END)  # Clear existing text in the entry field
                entry.insert(tk.END, random_row[field])

    fields = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
              'mean smoothness', 'mean compactness', 'mean concavity',
              'mean concave points', 'mean symmetry', 'mean fractal dimension',
              'radius error', 'texture error', 'perimeter error', 'area error',
              'smoothness error', 'compactness error', 'concavity error',
              'concave points error', 'symmetry error', 'fractal dimension error',
              'worst radius', 'worst texture', 'worst perimeter', 'worst area',
              'worst smoothness', 'worst compactness', 'worst concavity',
              'worst concave points', 'worst symmetry', 'worst fractal dimension'
              ]

    entries = {}
    form_prediction_window = tk.Toplevel(root)
    form_prediction_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(form_prediction_window))
    back_button = tk.Button(form_prediction_window, text="Back", command=back_to_main)
    back_button.grid(row=1, column=3, pady=5)
    validate_command = root.register(validate_float)
    for i, field in enumerate(fields):
        label = tk.Label(form_prediction_window, text=field, width=25, anchor='w')
        label.grid(row=i, column=0, padx=5, pady=2, sticky='w')  # Align left

        entry = tk.Entry(form_prediction_window, width=25, validate='key', validatecommand=(validate_command, '%P'))
        entry.grid(row=i, column=1, padx=5, pady=2)

        entries[field] = entry
    form_prediction_window.resizable(False, False)
    text_area = tk.Text(form_prediction_window, height=15, width=40)
    text_area.grid(row=0, column=2, rowspan=len(fields), padx=10, pady=5)
    text_area.configure(state='disabled')
    random_generator_button = tk.Button(form_prediction_window, text="random", command=random)
    random_generator_button.grid(row=2, column=3, pady=5)
    fetch_button = tk.Button(form_prediction_window, text="Predire", command=lambda: fetch_entries(entries, text_area))
    fetch_button.grid(row=len(fields), column=1, pady=10, sticky='e')


# Prediction par csv
def open_csv_prediction():
    dataframe = None

    def on_closing(window):
        window.destroy()
        root.destroy()

    def predict():
        global dataframe
        X = dataframe.to_numpy()
        prediction_text.config(state='normal')
        prediction_text.insert(tk.INSERT, "Results : \n")
        for i, row in enumerate(X):
            row = row.reshape((1, -1))
            prediction = target[svm.predict(row)[0]]
            prediction_text.insert(tk.INSERT, f"{i + 1} : {prediction} \n")
        prediction_text.config(state="disabled")

    def get_data():
        global dataframe
        file_types = (
            ("csv file", "*.csv"),
            ("text files", "*.txt"),
            ("all files", "*.*")
        )
        file_name = filedialog.askopenfilename(
            title="Open file",
            initialdir="/Desktop",
            filetypes=file_types
        )
        if file_name:
            dataframe = pd.read_csv(filepath_or_buffer=file_name)

            df_tree = ttk.Treeview(csv_prediction_window)
            y_scrollbar = ttk.Scrollbar(df_tree, orient="vertical")
            y_scrollbar.pack(side="right", fill="y")

            x_scrollbar = ttk.Scrollbar(df_tree, orient="horizontal")
            x_scrollbar.pack(side="bottom", fill="x")
            df_tree.config(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
            y_scrollbar.config(command=df_tree.yview)
            x_scrollbar.config(command=df_tree.xview)

            df_tree.place(x=25, y=55, width=358, height=689)

            df_tree["columns"] = ["index"] + list(dataframe.columns)
            df_tree.heading("#0", text="Index")
            df_tree.column("#0", width=50)  # Set width for index column

            # Configure the treeview columns
            for col in dataframe.columns:
                df_tree.heading(col, text=col)
                df_tree.column(col, anchor="center")

            # Insert data into the treeview
            for i, row in dataframe.iterrows():
                df_tree.insert("", "end", values=[i] + list(row))
            df_tree.grid_columnconfigure(0, weight=1)
            df_tree.grid_rowconfigure(0, weight=1)
            predict_button = tk.Button(csv_prediction_window, text="Predict", command=predict)
            predict_button.place(x=465, y=394, width=61, height=32)
        else:
            pass

    def back_to_main():
        csv_prediction_window.destroy()
        root.deiconify()

    csv_prediction_window = tk.Toplevel(root)
    csv_prediction_window.geometry("1000x1000")
    csv_prediction_window.resizable(False, False)
    csv_prediction_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(csv_prediction_window))
    back_button = tk.Button(csv_prediction_window, text="Back", command=back_to_main)
    back_button.place(x=916, y=13, width=50, height=24)
    text_frame = ttk.Frame(csv_prediction_window)
    text_frame.place(x=608, y=155, width=358, height=689)
    prediction_text = ScrolledText(text_frame, state="disabled")
    prediction_text.pack()
    get_data_button = tk.Button(csv_prediction_window, text="Get data", command=get_data)
    get_data_button.place(x=25, y=13, width=50, height=24)


root = tk.Tk()
root.title("Prediction Selection")


def form_prediction_selected():
    open_form_prediction()
    root.withdraw()


def csv_prediction_selected():
    open_csv_prediction()
    root.withdraw()


root.title("Breast Cancer Prediction")
root.geometry("200x200")
label = tk.Label(root, text="Choose prediction method:")
label.pack(pady=10)

form_button = tk.Button(root, text="Form Prediction", command=form_prediction_selected)
form_button.pack(pady=5)

csv_button = tk.Button(root, text="CSV Prediction", command=csv_prediction_selected)
csv_button.pack(pady=5)

root.mainloop()
