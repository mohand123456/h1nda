import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # بيانات تدريب بسيطة
    data = pd.DataFrame([
        {"radiation_type": "WiFi", "frequency_GHz": 2.4, "power_dBm": -30, "duration_minutes": 60, "SAR_W_per_kg": 0.3, "label": "Non-Harmful"},
        {"radiation_type": "5G", "frequency_GHz": 28, "power_dBm": -20, "duration_minutes": 180, "SAR_W_per_kg": 1.5, "label": "Harmful"},
        {"radiation_type": "Bluetooth", "frequency_GHz": 2.45, "power_dBm": -35, "duration_minutes": 120, "SAR_W_per_kg": 0.2, "label": "Non-Harmful"},
        {"radiation_type": "4G", "frequency_GHz": 1.8, "power_dBm": -25, "duration_minutes": 200, "SAR_W_per_kg": 1.2, "label": "Harmful"},
    ])

    # تحويل نوع الإشعاع لقيمة رقمية
    data["radiation_type"] = data["radiation_type"].astype("category")
    data["radiation_type_cat"] = data["radiation_type"].cat.codes

    X = data[["radiation_type_cat", "frequency_GHz", "power_dBm", "duration_minutes", "SAR_W_per_kg"]]
    y = data["label"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, X.columns.tolist()
