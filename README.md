# 🫁 Lung Cancer Prediction System

A machine learning-powered web application that predicts the likelihood of lung cancer based on patient health factors and lifestyle characteristics. This project uses a Logistic Regression model trained on comprehensive health data to provide risk assessments.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Author](#author)

## 🎯 Overview

This application provides a user-friendly interface for lung cancer risk assessment. It evaluates multiple health and lifestyle factors to classify individuals into three risk categories:
- **High Risk**: Elevated probability of lung cancer
- **Medium Risk**: Moderate probability of lung cancer
- **Low Risk**: Lower probability of lung cancer

> ⚠️ **Important**: This tool is for informational purposes only and should never replace professional medical consultation.

## ✨ Features

- 🎨 **Interactive Web Interface**: Built with Streamlit for easy accessibility
- 📊 **Comprehensive Risk Assessment**: Analyzes 16+ health and lifestyle factors
- 🤖 **Machine Learning Model**: Logistic Regression for binary/multi-class classification
- 📈 **Dataset Insights**: View real-time statistics about the training dataset
- 🔍 **Detailed Input Form**: Organized sections for lifestyle, symptoms, and risk factors
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices

## 📁 Project Structure

```
LUNGcancer/
├── nextapp.py                 # Main Streamlit application
├── datasetcancer.CSV          # Training dataset
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # GNU General Public License v3.0
```

## 🛠️ Requirements

- Python 3.7+
- Streamlit
- pandas
- scikit-learn
- numpy (implicit dependency)

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vaibhav82736/LUNGcancer.git
   cd LUNGcancer
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run nextapp.py
   ```

   The application will open in your default web browser at `http://localhost:8501`

## 🚀 Usage

1. **Launch the Application**: Run the command above
2. **Enter Personal Information**: 
   - Age and Gender
3. **Provide Lifestyle Factors**:
   - Alcohol use
   - Dust allergy exposure
   - Occupational hazards
   - Genetic risk factors
   - Chronic lung disease history
   - Diet quality
   - Obesity level
   - Smoking habits
   - Passive smoke exposure
4. **Report Symptoms**:
   - Chest pain
   - Coughing of blood
   - Fatigue
   - Weight loss
   - Shortness of breath
   - Wheezing
   - Swallowing difficulty
   - Clubbing of finger nails
   - Frequent cold
   - Dry cough
   - Snoring
5. **Get Prediction**: Click the "Predict" button to receive risk assessment
6. **View Dataset Info**: Check the sidebar for statistics about the training dataset

## 📊 Dataset

- **Source**: `datasetcancer.CSV`
- **Total Features**: 16+ input features
- **Target Variable**: Lung cancer risk level (High/Medium/Low)
- **Distribution**: Available in the sidebar when running the application
- **Data Size**: ~52KB

### Features Used:
- Age, Gender
- Lifestyle factors (alcohol, diet, obesity, smoking)
- Environmental factors (dust allergy, occupational hazards)
- Medical factors (genetic risk, chronic lung disease)
- Symptoms (chest pain, cough, fatigue, etc.)

## 🧠 Model Details

- **Algorithm**: Logistic Regression
- **Library**: scikit-learn
- **Train-Test Split**: 80-20 stratified split
- **Max Iterations**: 10,000
- **Random State**: 2 (for reproducibility)

### Model Performance:
- Accuracy metrics and performance statistics are available upon request

## ⚠️ Disclaimer

**This application is for educational and informational purposes only.**

- ❌ This tool should NOT be used for medical diagnosis
- ❌ Results are predictions, not medical advice
- ✅ Always consult qualified healthcare professionals for medical concerns
- ✅ Combine with professional medical evaluation and clinical judgment

**In case of health concerns, please seek immediate professional medical consultation.**

## 📄 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Vaibhav82736**  
GitHub: [@Vaibhav82736](https://github.com/Vaibhav82736)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Vaibhav82736/LUNGcancer/issues).

## 📞 Support

For issues or questions:
1. Check existing [GitHub issues](https://github.com/Vaibhav82736/LUNGcancer/issues)
2. Create a new issue with detailed information
3. Include any error messages or unexpected behavior

---

**⭐ If you found this project helpful, please consider giving it a star!**

*Last Updated: 2024*
