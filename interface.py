import tkinter as tk
from tkinter import ttk
import pandas as pd
import requests
from tkinter import messagebox
from tkinter import filedialog
import os

class BreastCancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer Detection")
        self.root.configure(bg="#f0f0f0")
        
        self.tab_control = ttk.Notebook(root)
        self.train_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.train_tab, text='Train Model')
        self.data_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.data_tab, text='Breast Cancer Detection')
        self.tab_control.pack(expand=1, fill="both")
        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # Labels for input features
        self.labels = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
            "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
            "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
            "symmetry_worst", "fractal_dimension_worst"
        ]
        
        # Widget lists for input labels and entry boxes
        self.label_widgets = []
        self.entry_widgets = []

        # Default values
        default_values = [
            "17.99", "10.38", "122.8", "1001.0", "0.1184",
            "0.2776", "0.3001", "0.1471", "0.2419", "0.07871",
            "1.095", "0.9053", "8.589", "153.4", "0.006399",
            "0.04904", "0.05373", "0.01587", "0.03003", "0.006193",
            "25.38", "17.33", "184.6", "2019.0", "0.1622",
            "0.6656", "0.7119", "0.2654", "0.4601", "0.1189"
        ]

        for i, label_text in enumerate(self.labels):
            label = tk.Label(self.data_tab, text=label_text, bg="#f0f0f0", wraplength=120)
            entry = tk.Entry(self.data_tab, width=20)
            entry.insert(0, default_values[i])

            row = i // 2
            column = i % 2

            label.grid(row=row, column=column * 2, padx=10, pady=5, sticky="w")
            entry.grid(row=row, column=column * 2 + 1, padx=10, pady=5)

            self.label_widgets.append(label)
            self.entry_widgets.append(entry)

        submit_button = tk.Button(self.data_tab, text="Analyze", command=self.get_data, bg="#4CAF50", fg="white")
        submit_button.grid(row=len(self.labels) // 2 + 1, columnspan=4, pady=10, padx=10, sticky="e")

        self.selected_file_path = ""
        self.labels.insert(0, "diagnosis")

        # Train tab
        train_label = tk.Label(self.train_tab, text="Choose an Excel file with labels:", bg="#f0f0f0", font=("Helvetica", 12, "bold"))
        train_label.pack(side=tk.TOP, pady=10, padx=10)

        label_row = tk.Frame(self.train_tab, bg="#f0f0f0")
        label_row.pack(pady=5)

        for i, label_text in enumerate(self.labels):
            label = tk.Label(label_row, text=label_text, wraplength=150)
            label.pack(side=tk.LEFT, padx=10)
            if (i + 1) % 4 == 0:
                label_row = tk.Frame(self.train_tab, bg="#f0f0f0")
                label_row.pack(pady=5)

        self.file_label = tk.Label(self.train_tab, text="Selected File: None", bg="#f0f0f0")
        self.file_label.pack(side=tk.TOP, pady=30, padx=50)

        train_button = tk.Button(self.train_tab, text="Choose File", command=self.choose_file, bg="#4CAF50", fg="white")
        train_button.pack(side=tk.LEFT, pady=10, padx=10)

        train_button = tk.Button(self.train_tab, text="Train", command=self.train_model, bg="#4CAF50", fg="white")
        train_button.pack(side=tk.LEFT, pady=10, padx=10)

        download_button = tk.Button(self.train_tab, text="Create Example Excel", command=self.download_example_excel, bg="#4CAF50", fg="white")
        download_button.pack(side=tk.LEFT, pady=10, padx=10)

        self.result_label = tk.Label(root, text="", bg="#f0f0f0")
        self.result_label.pack(pady=10)

    def on_tab_change(self, event):
        # Clear the result label when the tab changes
        selected_tab_index = self.tab_control.index(self.tab_control.select())
        if selected_tab_index == 0:
            self.result_label.config(text='')

    def get_data(self):
        try:
            input_data = []
            if self.labels and self.labels[0] == "diagnosis":
                self.labels.pop(0)

            for i, label_text in enumerate(self.labels):
                input_data.append(float(self.entry_widgets[i].get()))

            url = "http://127.0.0.1:5000/predict"
            response = requests.post(url, json={"input_data": input_data})
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                self.result_label.config(text=f"Prediction: {prediction}")
            else:
                self.result_label.config(text="API request failed.")
        except Exception as e:
            self.result_label.config(text=f"An error occurred: {str(e)}")

    def choose_file(self):
        self.selected_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if self.selected_file_path:
            self.file_label.config(text=f"Selected File: {self.selected_file_path}")

    def train_model(self):
        try:
            if not self.selected_file_path:
                messagebox.showerror("Error", "Please choose a file first.")
                return

            df = pd.read_excel(self.selected_file_path)
            excel_headers = list(df.columns)
            if self.labels and self.labels[0] == "diagnosis":
                self.labels.pop(0)
            self.labels.insert(0, "diagnosis")
            if excel_headers != self.labels:
                messagebox.showwarning("Warning", "The headers in the Excel file are different from the expected headers. You can download an example Excel file.")
            else:
                self.send_train_api()
                messagebox.showinfo("Info", "Train Success")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while reading the Excel file: {str(e)}")

    def send_train_api(self):
        try:
            data = {"file_path": self.selected_file_path}
            response = requests.post("http://127.0.0.1:5000/train", json=data)
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", "Successfully trained.")
                print(message)
            else:
                print("API request failed. Error code:", response.status_code)
        except Exception as e:
            print(str(e))

    def download_example_excel(self):
        self.labels.pop(0)
        self.labels.insert(0, "diagnosis")
        data = {label: [""] for label in self.labels}
        df = pd.DataFrame(data)
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Success", "Download successful.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BreastCancerApp(root)
    root.mainloop()
