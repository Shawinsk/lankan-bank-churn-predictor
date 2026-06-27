import joblib 
import pandas as pd
import scipy.sparse
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        self.preprocessor = joblib.load('sl_preprocessor.joblib')
        self.model = joblib.load('sl_model.joblib')
    
    def predict(self, data):
        mapped_data = pd.DataFrame()
        
        # 1. District
        if 'District' in data.columns:
            mapped_data['District'] = data['District'].astype(str)
        elif 'Geography' in data.columns:
            mapped_data['District'] = data['Geography'].astype(str)
        else:
            mapped_data['District'] = 'Colombo'
            
        # 2. CRIB_Status
        if 'CRIB_Status' in data.columns:
            mapped_data['CRIB_Status'] = data['CRIB_Status'].astype(str)
        elif 'cribStatus' in data.columns:
            mapped_data['CRIB_Status'] = data['cribStatus'].astype(str)
        else:
            mapped_data['CRIB_Status'] = 'Clean'
            
        # 3. Age
        if 'Age' in data.columns:
            mapped_data['Age'] = data['Age'].astype(float)
        else:
            mapped_data['Age'] = 35.0
            
        # 4. Tenure_Years
        if 'Tenure_Years' in data.columns:
            mapped_data['Tenure_Years'] = data['Tenure_Years'].astype(float)
        elif 'Tenure' in data.columns:
            mapped_data['Tenure_Years'] = data['Tenure'].astype(float)
        else:
            mapped_data['Tenure_Years'] = 3.0
            
        # 5. Account_Balance_LKR
        if 'Account_Balance_LKR' in data.columns:
            mapped_data['Account_Balance_LKR'] = data['Account_Balance_LKR'].astype(float)
        elif 'Balance' in data.columns:
            mapped_data['Account_Balance_LKR'] = data['Balance'].astype(float)
        else:
            mapped_data['Account_Balance_LKR'] = 0.0
            
        # 6. Recent_Loan_Rejected
        if 'Recent_Loan_Rejected' in data.columns:
            mapped_data['Recent_Loan_Rejected'] = data['Recent_Loan_Rejected'].astype(int)
        elif 'recentLoanRejected' in data.columns:
            mapped_data['Recent_Loan_Rejected'] = data['recentLoanRejected'].astype(int)
        else:
            mapped_data['Recent_Loan_Rejected'] = 0
            
        # 7. Digital_Banking_User
        if 'Digital_Banking_User' in data.columns:
            mapped_data['Digital_Banking_User'] = data['Digital_Banking_User'].astype(int)
        elif 'digitalBankingUser' in data.columns:
            mapped_data['Digital_Banking_User'] = data['digitalBankingUser'].astype(int)
        else:
            mapped_data['Digital_Banking_User'] = 1
            
        # Reorder columns to match exactly the training columns
        model_df = mapped_data[['Age', 'Tenure_Years', 'Account_Balance_LKR', 'District', 'CRIB_Status', 'Recent_Loan_Rejected', 'Digital_Banking_User']]
        
        # Transform using preprocessor - output may be sparse, convert to dense
        transformed_data = self.preprocessor.transform(model_df)
        if scipy.sparse.issparse(transformed_data):
            transformed_data = transformed_data.toarray()
        
        feature_names = self.preprocessor.get_feature_names_out()
        data_df = pd.DataFrame(transformed_data, columns=feature_names)
        prediction = self.model.predict(data_df)
        
        return prediction
