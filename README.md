# 🌤️ London Weather Temperature Predictor

## 📌 Project Overview
This project is a **machine learning model** that predicts the **mean temperature in London** based on various weather conditions. It uses **Random Forest Regression** for accurate predictions and is deployed as an **interactive web app** using Streamlit. 🚀

---

## 📂 Project Structure
```
📁 LONDON WEATHER/
│-- 📜 london_weather.py   # Machine learning model training & tuning
│-- 📜 app.py              # Streamlit web app
│-- 💜 london_weather.csv  # Dataset used for training
│-- 💜 requirements.txt    # List of dependencies
│-- 🖼️ Feature Importance in Predicting Temperature.png  # Visualization
│-- 📜 README.md           # Project documentation (this file)
```

---

## 🔧 Features & Technologies Used
- **Python** 🐍
- **pandas, NumPy** 📊 (Data Processing)
- **scikit-learn** 🤖 (Machine Learning)
- **Matplotlib** 📈 (Visualization)
- **Streamlit** 🎨 (Web App Deployment)

---

## 📊 Feature Importance
The **most important factors** for predicting temperature (based on model analysis):
1️⃣ **Min Temperature** 🌡️ (Strongest predictor)
2️⃣ **Max Temperature** 🔥
3️⃣ **Global Radiation** ☀️
4️⃣ **Sunshine** ⏳
5️⃣ **Pressure** 📏
6️⃣ **Cloud Cover** ☁️
7️⃣ **Precipitation** 🌧️

![Feature Importance](Feature%20Importance%20in%20Predicting%20Temperature.png)

---

## 🚀 How to Run This Project
### **1️⃣ Install Required Packages**
Run the following command to install dependencies:
```sh
pip install -r requirements.txt
```

### **2️⃣ Run the Streamlit App**
```sh
streamlit run app.py
```
Then, open the browser to interact with the UI and make predictions!

---

## 📈 Model Performance
| Model           | MAE (°C)  | R² Score |
|----------------|----------|---------|
| **Decision Tree**  | 0.96°C  | 0.95    |
| **Random Forest**  | 0.69°C  | 0.97    |
| **Optimized RF**  | 0.68°C  | 0.974   |

✅ **Random Forest performed the best!**

---

## 🎯 Future Improvements
- ✅ **Deploy the app online** (Streamlit Cloud or Hugging Face Spaces)
- ✅ **Enhance the UI** with better visuals & graphs
- ✅ **Improve model performance** by adding new features

---

## 👨‍💻 Author
🚀 **Harjot / Iris** 

