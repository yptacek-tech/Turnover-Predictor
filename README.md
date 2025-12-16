# Employee Attrition Risk Prediction System

AI-driven predictive model for identifying employees at risk of voluntary departure, enabling proactive retention interventions.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository** (or download the project files)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import pandas, numpy, sklearn; print('✓ All dependencies installed')"
   ```

## 📁 Project Structure

```
AI Class Project/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── FRONTEND_INTEGRATION_GUIDE.md          # Guide for frontend developers
├── employee_attrition_dataset_final.csv   # Main dataset (500 employees)
├── generate_dataset.py                    # Data generator script
├── train_model.py                         # Model training script
├── predict.py                             # ⭐ Main prediction module
└── model/                                 # Trained model files
    ├── attrition_model.pkl
    ├── scaler.pkl
    └── feature_info.pkl
```

## 🔧 Usage

### Generate Dataset (Optional)

If you need to regenerate the employee dataset:

```bash
python generate_dataset.py
```

This creates `employee_attrition_dataset_final.csv` with 500 synthetic employees.

### Train Model (Optional)

If you need to retrain the model:

```bash
python train_model.py
```

This saves the trained model to `model/` folder.

### Make Predictions

```python
import pandas as pd
from predict import predict_attrition_risk, get_high_risk_employees

# Load data
df = pd.read_csv('employee_attrition_dataset_final.csv')

# Predict for one employee
prediction = predict_attrition_risk(df.iloc[0])
print(f"Risk Score: {prediction['risk_score']}%")
print(f"Risk Level: {prediction['risk_level']} {prediction['risk_color']}")

# Get all high-risk employees
high_risk = get_high_risk_employees(df, min_level='High')
for emp in high_risk:
    print(f"{emp['employee_name']}: {emp['risk_score']}%")
```

## 📊 Features

- **4-Tier Risk Scoring**: Low (0-30%), Moderate (30-60%), High (60-85%), Critical (85-100%)
- **Top 3 Risk Drivers**: Identifies most impactful factors with influence percentages
- **Risk Trend Analysis**: Shows if risk is increasing (↑), stable (→), or decreasing (↓)
- **Comprehensive Data**: 3 years of historical records, performance reviews, engagement surveys
- **Explainable AI**: Natural language explanations for each prediction

## 🎯 Risk Drivers

The system monitors 10 key risk factors:

1. No Career Progression (18% weight)
2. Below-Market Compensation (16% weight)
3. Declining Manager Engagement (14% weight)
4. Declining Office Presence (12% weight)
5. Increasing Sick Leave (11% weight)
6. Declining Engagement Score (10% weight)
7. Training & Development Gap (6% weight)
8. Recognition Gap (5% weight)
9. Team Turnover Contagion (4% weight)
10. Communication Slowdown (4% weight)

## 📖 Documentation

- **Frontend Developers**: See `FRONTEND_INTEGRATION_GUIDE.md` for API documentation and integration examples
- **Data Scientists**: See code comments in `predict.py` and `train_model.py` for model details

## 🔬 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 25+ engineered features from employee data
- **Training Data**: 500 employees (90% normal, 10% at-risk)
- **Performance**: ~85% accuracy, ~88% recall on test set

## 📝 License

This project is for educational/academic purposes.

## 👥 Team

- **Coder A (Backend)**: Data generation, model training, prediction engine
- **Coder B (Frontend)**: UI/UX, dashboard development
- **Documenter**: Business documentation
- **Presenter**: Demo and visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## ⚠️ Important Notes

- The dataset contains synthetic data for demonstration purposes
- Model files (`.pkl`) must be present in `model/` folder for predictions
- CSV path may need adjustment based on your file structure
- For production use, retrain with real organizational data

## 📞 Support

For questions or issues, refer to:
- `FRONTEND_INTEGRATION_GUIDE.md` for frontend integration
- Code comments in Python files for implementation details

---

**Built for AI Class Project - Predictive Attrition Risk Modelling**

