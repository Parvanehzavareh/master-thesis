# Appliance Load Identification Using Sequential Deep Learning Models  
*Towards Predicting and Understanding Household Energy Consumption*  

📖 Master of Science in **Information Systems: Business Analytics**  
Kristiania University College, Oslo, Norway  

---

## 📌 Overview
This project explores **Non-Intrusive Load Monitoring (NILM)** using **deep learning models** to disaggregate household energy consumption into appliance-level usage without installing intrusive sensors.  

We investigate two CNN-based architectures:  
- **Sequence-to-Sequence (Seq2Seq)**  
- **Sequence-to-Point (Seq2Point)**  

The study compares their effectiveness in:  
- Appliance-level energy disaggregation  
- On/Off state detection of household devices  
- Cross-dataset generalization  

---

## 🎯 Research Goals
- Identify how CNNs can model **appliance load profiles** from aggregated signals.  
- Compare **Seq2Seq vs Seq2Point** performance across different appliances.  
- Evaluate **cross-domain generalizability** using multiple benchmark datasets.  
- Study the impact of **hyperparameters** (window size, batch size, dropout, etc.) on performance.  

---

## 🛠️ Methodology
1. **Datasets**:  
   - **UK-DALE** (primary training and validation)  
   - **REFIT** (cross-dataset evaluation)  

2. **Data Processing**:  
   - Cleaning, resampling, normalization, labeling (on/off states)  
   - Sliding-window sequence generation  

3. **Models**:  
   - CNN-based Seq2Seq and Seq2Point architectures  
   - Hyperparameter tuning and parameter studies  

4. **Evaluation Metrics**:  
   - **Regression**: Mean Absolute Error (MAE), Signal Aggregate Error (SAE)  
   - **Classification**: Accuracy, Precision, Recall, F1-score  

---

## 📊 Results
- **Seq2Seq**: Strong for continuous load modeling, but less accurate at boundaries.  
- **Seq2Point**: More effective for **event detection** and on/off classification.  
- **Cross-Dataset Testing**: Models showed varying generalization ability, highlighting dataset and preprocessing importance.  

---

## 🚀 Contributions
- Comparative study of Seq2Seq vs Seq2Point CNNs for NILM.  
- Systematic **parameter optimization** for CNN-based NILM tasks.  
- Insights into improving **appliance-level disaggregation** and **state detection**.  
- Practical implications for **smart homes** and **IoT-based energy management**.  

---

## 📂 Repository Structure
