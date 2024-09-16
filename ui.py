import tkinter as tk
from tkinter import messagebox
import requests
import json

# Dummy label encoders for dropdown options
label_encoders = {
    'Type': {'Strength': 0, 'Cardio': 1, 'Flexibility': 2},
    'BodyPart': {'Abdominals': 0, 'Arms': 1, 'Back': 2, 'Chest': 3, 'Legs': 4},
    'Equipment': {'None': 0, 'Dumbbell': 1, 'Barbell': 2, 'Kettlebell': 3, 'Bands': 4},
    'Level': {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
}

def predict_exercise():
    try:
        type_ = type_var.get()
        body_part = body_part_var.get()
        equipment = equipment_var.get()
        level = level_var.get()

        input_data = [label_encoders['Type'][type_],
                      label_encoders['BodyPart'][body_part],
                      label_encoders['Equipment'][equipment],
                      label_encoders['Level'][level]]

        # Predict using RandomForest model
        response_rf = requests.post('http://127.0.0.1:8000/predict_rf', json={'data': input_data})
        rf_prediction = response_rf.json()['prediction']

        # Predict using Deep Learning model
        response_dl = requests.post('http://127.0.0.1:8000/predict_dl', json={'data': input_data})
        dl_prediction = response_dl.json()['prediction']

        messagebox.showinfo("Prediction", f"RandomForest Prediction: {rf_prediction}\nDeep Learning Prediction: {dl_prediction}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

app_ui = tk.Tk()
app_ui.title("Exercise Recommendation System")

tk.Label(app_ui, text="Type").grid(row=0)
tk.Label(app_ui, text="Body Part").grid(row=1)
tk.Label(app_ui, text="Equipment").grid(row=2)
tk.Label(app_ui, text="Level").grid(row=3)

type_var = tk.StringVar(app_ui)
body_part_var = tk.StringVar(app_ui)
equipment_var = tk.StringVar(app_ui)
level_var = tk.StringVar(app_ui)

type_menu = tk.OptionMenu(app_ui, type_var, *label_encoders['Type'].keys())
body_part_menu = tk.OptionMenu(app_ui, body_part_var, *label_encoders['BodyPart'].keys())
equipment_menu = tk.OptionMenu(app_ui, equipment_var, *label_encoders['Equipment'].keys())
level_menu = tk.OptionMenu(app_ui, level_var, *label_encoders['Level'].keys())

type_menu.grid(row=0, column=1)
body_part_menu.grid(row=1, column=1)
equipment_menu.grid(row=2, column=1)
level_menu.grid(row=3, column=1)

predict_button = tk.Button(app_ui, text="Predict Exercise", command=predict_exercise).grid(row=4, columnspan=2)

app_ui.mainloop()
