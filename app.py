# app.py - Modified version
import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import joblib
import os
app = Flask(__name__)

# Load your trained model
print("🚀 Loading trained ML model...")
try:
    model_artifacts = joblib.load('best_model.pkl')
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    feature_names = model_artifacts['feature_names']
    
    # Handle both 'model_type' and 'model_name'
    if 'model_type' in model_artifacts:
        model_type = model_artifacts['model_type']
    elif 'model_name' in model_artifacts:
        model_type = model_artifacts['model_name']
    else:
        model_type = "Unknown"
    
    print(f"✅ Model loaded: {model_type}")
    print(f"📋 Features: {len(feature_names)}")
    MODEL_LOADED = True
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False
    # Initialize with empty values to avoid errors
    model = None
    scaler = None
    feature_names = []
    model_type = "Error"

def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    """Preprocess input to match training features"""
    if not MODEL_LOADED:
        raise Exception("Model not loaded")
    
    # Create a DataFrame with all expected features, initialized to 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Set basic features
    input_df['Sex'] = 1 if sex == 'female' else 0
    input_df['Age'] = age
    input_df['SibSp'] = sibsp
    input_df['Parch'] = parch
    input_df['Fare'] = fare
    
    # Create engineered features
    input_df['FamilySize'] = sibsp + parch + 1
    input_df['IsAlone'] = 1 if input_df['FamilySize'].iloc[0] == 1 else 0
    
    # Set HasCabin (default to 0)
    input_df['HasCabin'] = 0
    
    # Set Title (default to 2 which is 'Mr' in encoding)
    input_df['Title'] = 2
    
    # Set one-hot encoded features
    # Embarked
    if embarked == 'C':
        input_df['Embarked_C'] = 1
    elif embarked == 'Q':
        input_df['Embarked_Q'] = 1
    else:
        input_df['Embarked_S'] = 1
    
    # Age Group
    if age <= 12:
        input_df['AgeGroup_Child'] = 1
    elif age <= 18:
        input_df['AgeGroup_Teen'] = 1
    elif age <= 35:
        input_df['AgeGroup_Young Adult'] = 1
    elif age <= 60:
        input_df['AgeGroup_Adult'] = 1
    else:
        input_df['AgeGroup_Senior'] = 1
    
    # Fare Group
    if fare <= 7.91:
        input_df['FareGroup_Low'] = 1
    elif fare <= 14.45:
        input_df['FareGroup_Medium'] = 1
    elif fare <= 31.0:
        input_df['FareGroup_High'] = 1
    else:
        input_df['FareGroup_Very High'] = 1
    
    # Pclass
    if pclass == 1:
        input_df['Pclass_1'] = 1
    elif pclass == 2:
        input_df['Pclass_2'] = 1
    else:
        input_df['Pclass_3'] = 1
    
    # Scale numerical features
    numerical_columns = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    if scaler and len(numerical_columns) > 0:
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
    
    return input_df

@app.route('/')
def home():
    model_status = "✅ TRAINED MODEL ACTIVE" if MODEL_LOADED else "⚠️ MODEL LOADING ERROR"
    accuracy = model_artifacts.get('accuracy', 0.6536) if MODEL_LOADED else 0.0
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Titanic Survival Predictor</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; }}
            .form-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }}
            .form-group {{ margin-bottom: 15px; }}
            label {{ display: block; margin-bottom: 5px; font-weight: bold; color: #34495e; }}
            input, select {{ width: 100%; padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px; font-size: 16px; }}
            button {{ background: #3498db; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 18px; width: 100%; }}
            button:hover {{ background: #2980b9; }}
            .model-info {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚢 Titanic Survival Prediction</h1>
            <p>Complete ML Pipeline - Trained Model Deployment</p>
            
            <div class="model-info">
                <h3>Model Information:</h3>
                <p><strong>Status:</strong> {model_status}</p>
                <p><strong>Algorithm:</strong> {model_type}</p>
                <p><strong>Accuracy:</strong> {accuracy:.2%}</p>
                <p><strong>Features:</strong> {len(feature_names)} engineered features</p>
            </div>
            
            <form action="/predict" method="post">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Passenger Class:</label>
                        <select name="pclass" required>
                            <option value="1">1st Class</option>
                            <option value="2">2nd Class</option>
                            <option value="3" selected>3rd Class</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Gender:</label>
                        <select name="sex" required>
                            <option value="male" selected>Male</option>
                            <option value="female">Female</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Age:</label>
                        <input type="number" name="age" value="25" min="0" max="100" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Siblings/Spouses:</label>
                        <input type="number" name="sibsp" value="1" min="0" max="10" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Parents/Children:</label>
                        <input type="number" name="parch" value="2" min="0" max="10" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Fare (£):</label>
                        <input type="number" name="fare" value="30" step="0.01" min="0" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Embarkation Port:</label>
                        <select name="embarked" required>
                            <option value="C">Cherbourg</option>
                            <option value="Q">Queenstown</option>
                            <option value="S" selected>Southampton</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit">Predict Survival</button>
            </form>
            
            <div style="text-align: center; margin-top: 20px;">
                <a href="/health" style="margin-right: 15px;">Health Check</a>
                <a href="/api/info">API Documentation</a>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not MODEL_LOADED:
            return '''
            <div style="max-width: 600px; margin: 40px auto; padding: 20px; background: #f8d7da; color: #721c24; border-radius: 5px;">
                <h1>Model Not Loaded</h1>
                <p>Please train the model first by running: <code>python train_model.py</code></p>
                <button onclick="window.history.back()">Go Back</button>
            </div>
            '''
            
        # Get form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        
        # Preprocess input
        input_features = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]
        
        # Prepare result
        result = "SURVIVED 🎉" if prediction == 1 else "DID NOT SURVIVE 😢"
        confidence = f"{probability:.2%}"
        
        status_class = "survived" if prediction == 1 else "not-survived"
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .result {{ margin: 20px 0; padding: 30px; border-radius: 10px; text-align: center; }}
                .survived {{ background: #d4edda; color: #155724; border: 3px solid #c3e6cb; }}
                .not-survived {{ background: #f8d7da; color: #721c24; border: 3px solid #f5c6cb; }}
                button {{ background: #3498db; color: white; padding: 12px 25px; border: none; border-radius: 5px; cursor: pointer; margin: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prediction Result</h1>
                <div class="result {status_class}">
                    <h2>{result}</h2>
                    <p><strong>Confidence:</strong> {confidence}</p>
                    <p><strong>Probability:</strong> {probability:.4f}</p>
                </div>
                <div style="text-align: center;">
                    <button onclick="window.location.href='/'">Make Another Prediction</button>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        return f'''
        <div style="max-width: 600px; margin: 40px auto; padding: 20px; background: #f8d7da; color: #721c24; border-radius: 5px;">
            <h1>Error</h1>
            <p><strong>Error Details:</strong> {str(e)}</p>
            <button onclick="window.history.back()">Go Back</button>
        </div>
        '''

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_type": model_type,
        "accuracy": model_artifacts.get('accuracy', 0.6536) if MODEL_LOADED else 0.0,
        "features": len(feature_names),
        "endpoints": ["GET /", "POST /predict", "GET /health", "GET /api/info"]
    })

@app.route('/api/info', methods=['GET'])
def api_info():
    return jsonify({
        "api_name": "Titanic Survival Prediction API",
        "model": model_type,
        "accuracy": model_artifacts.get('accuracy', 0.6536) if MODEL_LOADED else 0.0,
        "features": len(feature_names),
        "input_format": {
            "pclass": "int (1, 2, 3)",
            "sex": "string (male, female)",
            "age": "float",
            "sibsp": "int",
            "parch": "int",
            "fare": "float",
            "embarked": "string (C, Q, S)"
        }
    })

class TitanicMLPipeline:
    def __init__(self):
        self.df = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.best_model_name = ""
        self.best_score = 0
        self.feature_names = []
        
    def _create_sample_data(self):
        """Create sample Titanic data if file not found"""
        print("   Creating sample Titanic dataset...")
        np.random.seed(42)
        n_samples = 891
        
        data = {
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.randint(0, 2, n_samples),
            'Pclass': np.random.randint(1, 4, n_samples),
            'Name': [f'Passenger {i}' for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.normal(30, 15, n_samples).clip(0, 80),
            'SibSp': np.random.randint(0, 4, n_samples),
            'Parch': np.random.randint(0, 4, n_samples),
            'Ticket': [f'Ticket_{i}' for i in range(1, n_samples + 1)],
            'Fare': np.random.exponential(50, n_samples).clip(0, 500),
            'Cabin': np.random.choice([None, 'C123', 'B45', 'D56'], n_samples),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples)
        }
        
        self.df = pd.DataFrame(data)
        # Introduce some missing values for realism
        self.df.loc[self.df.sample(frac=0.2).index, 'Age'] = np.nan
        self.df.loc[self.df.sample(frac=0.05).index, 'Embarked'] = np.nan
        
        # Add realistic survival patterns
        self.df.loc[(self.df['Sex'] == 'female') & (self.df['Pclass'] == 1), 'Survived'] = np.random.choice(
            [0, 1], 
            len(self.df[(self.df['Sex'] == 'female') & (self.df['Pclass'] == 1)]), 
            p=[0.1, 0.9]
        )
        self.df.loc[(self.df['Sex'] == 'male') & (self.df['Pclass'] == 3), 'Survived'] = np.random.choice(
            [0, 1], 
            len(self.df[(self.df['Sex'] == 'male') & (self.df['Pclass'] == 3)]), 
            p=[0.8, 0.2]
        )
        
        print(f"   Created sample dataset with {len(self.df)} records")
        
    def load_and_clean_data(self):
        """STEP 1: Data Loading and Cleaning"""
        print("\n📊 STEP 1: Loading and Cleaning Data...")
        
        # Load dataset
        try:
            self.df = pd.read_csv('titanic.csv')
            print(f"   ✅ Loaded Titanic dataset with {len(self.df)} records")
        except FileNotFoundError:
            print("   ⚠️  titanic.csv not found, using sample data...")
            self._create_sample_data()
        
        # Data Quality Report
        print("\n   📋 DATA QUALITY REPORT:")
        print(f"   Total Records: {len(self.df)}")
        print(f"   Features: {list(self.df.columns)}")
        
        # Handle Missing Values
        print("\n   🧹 HANDLING MISSING VALUES:")
        
        # Age - median imputation
        age_median = self.df['Age'].median()
        self.df['Age'].fillna(age_median, inplace=True)
        age_missing = self.df['Age'].isnull().sum()
        print(f"     Age: Filled {age_missing} missing values with median {age_median:.1f}")
        
        # Embarked - mode imputation
        embarked_mode = self.df['Embarked'].mode()[0]
        self.df['Embarked'].fillna(embarked_mode, inplace=True)
        print(f"     Embarked: Filled missing values with mode '{embarked_mode}'")
        
        # Fare - median imputation
        fare_median = self.df['Fare'].median()
        self.df['Fare'].fillna(fare_median, inplace=True)
        print(f"     Fare: Filled missing values with median {fare_median:.2f}")
        
        # Create HasCabin feature
        self.df['HasCabin'] = self.df['Cabin'].notna().astype(int)
        
        # Remove Duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_count - len(self.df)
        print(f"     Removed {duplicates_removed} duplicate records")
        
        # Handle Outliers
        print("\n   📊 HANDLING OUTLIERS (IQR METHOD):")
        numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = len(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)])
            print(f"     {col}: Found {outliers_count} outliers")
            
            # Cap outliers
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
        
        print(f"\n   ✅ CLEANING COMPLETED")
        print(f"   Final dataset: {self.df.shape[0]} records, {self.df.shape[1]} features")
        
    def exploratory_data_analysis(self):
        """STEP 2: Exploratory Data Analysis"""
        print("\n📈 STEP 2: Exploratory Data Analysis (EDA)...")
        
        # Create directory for plots
        os.makedirs('eda_plots', exist_ok=True)
        print("   📊 Creating visualizations...")
        
        # 1. Survival Distribution
        plt.figure(figsize=(8, 6))
        survival_counts = self.df['Survived'].value_counts()
        plt.pie(survival_counts.values, labels=['Not Survived', 'Survived'], 
                autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4'], startangle=90)
        plt.title('Survival Distribution')
        plt.savefig('eda_plots/survival_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Survival by Gender
        plt.figure(figsize=(8, 6))
        survival_by_sex = pd.crosstab(self.df['Sex'], self.df['Survived'])
        survival_by_sex.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'])
        plt.title('Survival by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.legend(['Not Survived', 'Survived'])
        plt.xticks(rotation=0)
        plt.savefig('eda_plots/survival_by_gender.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Survival by Class
        plt.figure(figsize=(8, 6))
        survival_by_class = pd.crosstab(self.df['Pclass'], self.df['Survived'])
        survival_by_class.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'])
        plt.title('Survival by Passenger Class')
        plt.xlabel('Passenger Class')
        plt.ylabel('Count')
        plt.legend(['Not Survived', 'Survived'])
        plt.xticks(rotation=0)
        plt.savefig('eda_plots/survival_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ EDA completed! Visualizations saved in 'eda_plots/' folder")
        
    def feature_engineering(self):
        """STEP 3: Feature Engineering"""
        print("\n🔧 STEP 3: Feature Engineering...")
        
        # Create copy for feature engineering
        feature_df = self.df.copy()
        
        print("   📝 APPLYING FEATURE TRANSFORMATIONS:")
        
        # 1. Create New Features
        feature_df['FamilySize'] = feature_df['SibSp'] + feature_df['Parch'] + 1
        print("     1. Created 'FamilySize' = SibSp + Parch + 1")
        
        feature_df['IsAlone'] = 0
        feature_df.loc[feature_df['FamilySize'] == 1, 'IsAlone'] = 1
        print("     2. Created 'IsAlone' (1 if traveling alone)")
        
        # 2. Extract Title from Name
        feature_df['Title'] = feature_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        feature_df['Title'] = feature_df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                                         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        feature_df['Title'] = feature_df['Title'].replace('Mlle', 'Miss')
        feature_df['Title'] = feature_df['Title'].replace('Ms', 'Miss')
        feature_df['Title'] = feature_df['Title'].replace('Mme', 'Mrs')
        print("     3. Extracted and categorized 'Title' from Name")
        
        # 3. Binning - Age Groups
        feature_df['AgeGroup'] = pd.cut(feature_df['Age'], 
                                       bins=[0, 12, 18, 35, 60, 100], 
                                       labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
        print("     4. Created 'AgeGroup' using binning (5 categories)")
        
        # 4. Binning - Fare Groups
        feature_df['FareGroup'] = pd.qcut(feature_df['Fare'], 4, 
                                         labels=['Low', 'Medium', 'High', 'Very High'])
        print("     5. Created 'FareGroup' using quantile binning")
        
        # 5. Encoding Categorical Variables
        label_encoders = {}
        binary_cols = ['Sex', 'Title']
        for col in binary_cols:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col])
            label_encoders[col] = le
        print("     6. Applied Label Encoding to 'Sex' and 'Title'")
        
        # One-Hot Encoding
        categorical_cols = ['Embarked', 'AgeGroup', 'FareGroup', 'Pclass']
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols, prefix=categorical_cols)
        print(f"     7. Applied One-Hot Encoding to {categorical_cols}")
        
        # 6. Drop unnecessary columns
        columns_to_drop = ['Name', 'Ticket', 'PassengerId', 'Cabin']
        feature_df = feature_df.drop(columns_to_drop, axis=1)
        print(f"     8. Dropped columns: {columns_to_drop}")
        
        # 7. Prepare features and target
        self.feature_names = feature_df.drop('Survived', axis=1).columns.tolist()
        X = feature_df.drop('Survived', axis=1)
        y = feature_df['Survived']
        
        # 8. Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 9. Feature Scaling
        numerical_columns = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        if len(numerical_columns) > 0:
            self.X_train[numerical_columns] = self.scaler.fit_transform(self.X_train[numerical_columns])
            self.X_test[numerical_columns] = self.scaler.transform(self.X_test[numerical_columns])
            print("     9. Applied StandardScaler to numerical features")
        
        print(f"\n   ✅ FEATURE ENGINEERING COMPLETED")
        print(f"   Final features: {len(self.feature_names)} features")
        print(f"   Training set: {self.X_train.shape[0]} records")
        print(f"   Testing set: {self.X_test.shape[0]} records")
        
    def train_and_evaluate_models(self):
        """STEP 4: Train and Evaluate Models"""
        print("\n🤖 STEP 4: Training and Evaluating Models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n   --- {name} ---")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
            
            # Print results
            print(f"     ✅ Accuracy: {accuracy:.4f}")
            print(f"     ✅ Precision: {precision:.4f}")
            print(f"     ✅ Recall: {recall:.4f}")
            print(f"     ✅ F1-Score: {f1:.4f}")
            print(f"     ✅ ROC-AUC: {roc_auc:.4f}")
            
            # Store model
            self.models[name] = model
        
        # Select best model based on accuracy
        self.best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[self.best_model_name]['model']
        self.best_score = results[self.best_model_name]['accuracy']
        
        print(f"\n   🏆 BEST MODEL: {self.best_model_name}")
        print(f"   Accuracy: {results[self.best_model_name]['accuracy']:.4f}")
        print(f"   Precision: {results[self.best_model_name]['precision']:.4f}")
        print(f"   Recall: {results[self.best_model_name]['recall']:.4f}")
        print(f"   F1-Score: {results[self.best_model_name]['f1_score']:.4f}")
        print(f"   ROC-AUC: {results[self.best_model_name]['roc_auc']:.4f}")
        
        # Save comparison table
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('model_comparison.csv', index=False)
        print("\n   📊 Model comparison saved to 'model_comparison.csv'")
        
        return results
    
    def save_best_model(self):
        """STEP 5: Save the best model"""
        print("\n💾 STEP 5: Saving the best model...")
        
        # Create model artifacts
        model_artifacts = {
            'model': self.best_model,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'accuracy': self.best_score
        }
        
        joblib.dump(model_artifacts, 'best_model.pkl')
        print("   ✅ Best model saved as 'best_model.pkl'")
        
        # Save feature list
        with open('features.txt', 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        print("   ✅ Feature list saved as 'features.txt'")

def train_model():
    """Training function - can be called separately"""
    print("🚀 ML MODEL TRAINING STARTING")
    print("=" * 50)
    
    pipeline = TitanicMLPipeline()
    
    try:
        # Execute pipeline steps
        pipeline.load_and_clean_data()
        pipeline.exploratory_data_analysis()
        pipeline.feature_engineering()
        results = pipeline.train_and_evaluate_models()
        pipeline.save_best_model()
        
        print("\n" + "=" * 50)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("   Model saved as 'best_model.pkl'")
        print("   Run 'python app.py' to deploy the model")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Check if we should run training or start the Flask app
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Run training mode
        train_model()
    else:
        # Run Flask app in deployment mode
        print("🚀 Starting Flask app in DEPLOYMENT mode...")
        print("   (Use 'python app.py train' to train the model)")
        print(f"   📍 Web interface: http://localhost:3000")
        print(f"   📍 Health check: http://localhost:3000/health")
        print(f"   📍 API info: http://localhost:3000/api/info")
        
        port = int(os.environ.get('PORT', 3000))
        app.run(host='0.0.0.0', port=port, debug=False)