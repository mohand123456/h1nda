import streamlit as st
import pandas as pd
from train_model import train_model

# تحميل النموذج
model, feature_columns = train_model()

st.title("🧠 تصنيف الإشعاع: هل هو ضار؟")
st.markdown("أدخل خصائص الإشعاع الناتج من الجهاز، وسيتم تحديد ما إذا كان ضارًا أو غير ضار")

# واجهة المستخدم
radiation_type = st.selectbox("نوع الإشعاع", ["WiFi", "4G", "5G", "Bluetooth", "IR"])
frequency_GHz = st.number_input("📡 التردد (GHz)", min_value=0.1, step=0.1)
power_dBm = st.number_input("🔋 شدة الإشارة (dBm)", step=1)
duration_minutes = st.number_input("⏱️ مدة التعرض (دقائق)", min_value=0, step=1)
SAR_W_per_kg = st.number_input("⚡ معدل الامتصاص (SAR - W/kg)", min_value=0.0, step=0.1)

if st.button("🔍 تصنيف الإشعاع"):
    input_data = pd.DataFrame([{
        "radiation_type": radiation_type,
        "frequency_GHz": frequency_GHz,
        "power_dBm": power_dBm,
        "duration_minutes": duration_minutes,
        "SAR_W_per_kg": SAR_W_per_kg,
    }])
    
    input_data["radiation_type"] = input_data["radiation_type"].astype("category")
    input_data["radiation_type_cat"] = input_data["radiation_type"].cat.codes
    
    # إعادة ترتيب الأعمدة حسب النموذج
    X_input = input_data[feature_columns]
    
    prediction = model.predict(X_input)[0]
    st.success(f"✅ التصنيف: {prediction}")
