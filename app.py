from flask import Flask, render_template, request, jsonify, send_file
import os 
import pandas as pd
import io
from functools import wraps
from mlFlowProject.pipeline.prediction import PredictionPipeline
from database import get_db_connection, init_db


app = Flask(__name__) # initializing a flask app
init_db() # initialize SQLite database

# API Key security configuration
BOC_API_KEY = os.environ.get("BOC_API_KEY", "BOC-CHURN-SECRET-KEY-2026")

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                api_key = auth_header.split(" ", 1)[1]
        
        if api_key != BOC_API_KEY:
            return jsonify({"error": "Unauthorized: Invalid or missing API Key"}), 401
        return f(*args, **kwargs)
    return decorated



@app.route('/',methods=['GET'])  # redirect home to prediction form
def homePage():
    return render_template('index.html')


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("dvc repro")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    bank = 'BOC'
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user from the form
            surname = request.form['Surname']
            credit_score = int(request.form['creditScore'])
            geography = request.form['geography']
            gender = request.form['gender']
            age = int(request.form['age'])
            tenure = int(request.form['tenure'])
            balance = float(request.form['balance'])
            number_of_products = int(request.form['numberOfProducts'])
            credit_card = int(request.form['creditCard'])
            active_member = int(request.form['activeMember'])
            estimated_salary = float(request.form['estimatedSalary'])
            crib_status = request.form['cribStatus']
            recent_loan_rejected = int(request.form['recentLoanRejected'])
            digital_banking_user = int(request.form['digitalBankingUser'])

            # Include rowNumber and customerId for data schema sake
            rowNumber, customerId = 0, 0
            
            # Creating a data dictionary for the prediction pipeline
            data_dict = {
                'RowNumber': [rowNumber],
                'CustomerId': [customerId],
                'Surname': [surname],
                'CreditScore': [credit_score],
                'Geography': [geography],
                'Gender': [gender],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [number_of_products],
                'HasCrCard': [credit_card],
                'IsActiveMember': [active_member],
                'EstimatedSalary': [estimated_salary],
                'CRIB_Status': [crib_status],
                'Recent_Loan_Rejected': [recent_loan_rejected],
                'Digital_Banking_User': [digital_banking_user]
            }

            data = pd.DataFrame(data_dict)

            prediction_pipeline = PredictionPipeline()
            prediction_result = prediction_pipeline.predict(data)

            if prediction_result[0] == 0:
                prediction = 'No'
            else:
                prediction = 'Yes'

            # Save the record in the SQLite database
            branch = request.form.get('branch', 'N/A')
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(CustomerId) FROM customers")
                max_id = cursor.fetchone()[0]
                new_customer_id = 15620001 if max_id is None else max_id + 1
                
                cursor.execute('''
                    INSERT INTO customers (CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Bank, Branch, CRIB_Status, Recent_Loan_Rejected, Digital_Banking_User, PredictionResult)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    new_customer_id, surname, credit_score, geography,
                    'Female' if gender == '1' else 'Male', age, tenure, balance,
                    number_of_products, credit_card, active_member, estimated_salary,
                    'Bank of Ceylon (BOC)', branch, crib_status, recent_loan_rejected, digital_banking_user, prediction
                ))
                conn.commit()
                conn.close()
            except Exception as db_err:
                print("Database save error:", db_err)

            return render_template('result.html', prediction=prediction, bank='BOC', branch=branch)
        except Exception as e:
            return str(e)
    return render_template('index.html')



@app.route('/batch', methods=['GET', 'POST'])
def batch_upload():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file:
                return "No file uploaded."
            
            # Read CSV into DataFrame
            df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), sep=",")
            
            # Ensure RowNumber and CustomerId are present
            if 'RowNumber' not in df.columns:
                df['RowNumber'] = range(len(df))
            if 'CustomerId' not in df.columns:
                df['CustomerId'] = [15620000 + idx for idx in range(len(df))]
                
            prediction_pipeline = PredictionPipeline()
            

            # Run predictions using the Sri Lankan model (trained on actual LKR scales directly)
            predictions = prediction_pipeline.predict(df)
            
            # Add prediction column
            df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]

            # Save batch prediction records to SQLite database
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                bank_display_name = 'Bank of Ceylon (BOC)'
                
                for _, row in df.iterrows():
                    # Check if customer already exists to avoid unique constraint error
                    cursor.execute("SELECT 1 FROM customers WHERE CustomerId = ?", (int(row['CustomerId']),))
                    exists = cursor.fetchone()
                    
                    # Convert Gender if it is 0/1 to Male/Female string for database display
                    gender_str = str(row['Gender'])
                    if gender_str == '1' or gender_str == '1.0' or gender_str.lower() == 'female':
                        gender_val = 'Female'
                    else:
                        gender_val = 'Male'
                        
                    crib_val = row.get('CRIB_Status', 'Clean')
                    loan_rejected_val = int(row.get('Recent_Loan_Rejected', 0))
                    digital_user_val = int(row.get('Digital_Banking_User', 1))

                    if exists:
                        # Update prediction result
                        cursor.execute('''
                            UPDATE customers 
                            SET Surname = ?, CreditScore = ?, Geography = ?, Gender = ?, Age = ?, 
                                Tenure = ?, Balance = ?, NumOfProducts = ?, HasCrCard = ?, 
                                IsActiveMember = ?, EstimatedSalary = ?, Bank = ?, Branch = ?, 
                                CRIB_Status = ?, Recent_Loan_Rejected = ?, Digital_Banking_User = ?,
                                PredictionResult = ?
                            WHERE CustomerId = ?
                        ''', (
                            row.get('Surname', 'N/A'),
                            int(row['CreditScore']),
                            row['Geography'],
                            gender_val,
                            int(row['Age']),
                            int(row['Tenure']),
                            float(row['Balance']),
                            int(row['NumOfProducts']),
                            int(row['HasCrCard']),
                            int(row['IsActiveMember']),
                            float(row['EstimatedSalary']),
                            row.get('Bank', bank_display_name),
                            row.get('Branch', 'Batch Upload'),
                            crib_val,
                            loan_rejected_val,
                            digital_user_val,
                            row['Churn_Prediction'],
                            int(row['CustomerId'])
                        ))
                    else:
                        cursor.execute('''
                            INSERT INTO customers (CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Bank, Branch, CRIB_Status, Recent_Loan_Rejected, Digital_Banking_User, PredictionResult)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            int(row['CustomerId']),
                            row.get('Surname', 'N/A'),
                            int(row['CreditScore']),
                            row['Geography'],
                            gender_val,
                            int(row['Age']),
                            int(row['Tenure']),
                            float(row['Balance']),
                            int(row['NumOfProducts']),
                            int(row['HasCrCard']),
                            int(row['IsActiveMember']),
                            float(row['EstimatedSalary']),
                            row.get('Bank', bank_display_name),
                            row.get('Branch', 'Batch Upload'),
                            crib_val,
                            loan_rejected_val,
                            digital_user_val,
                            row['Churn_Prediction']
                        ))
                conn.commit()
                conn.close()
            except Exception as batch_db_err:
                print("Batch DB save error:", batch_db_err)
            
            # Create a string buffer and write CSV to it
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode()),
                mimetype="text/csv",
                as_attachment=True,
                download_name="churn_predictions.csv"
            )
        except Exception as e:
            return str(e)
    return render_template('batch.html')


@app.route('/db_dashboard', methods=['GET'])
def db_dashboard():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM customers')
        customers = cursor.fetchall()
        conn.close()
        return render_template('db_dashboard.html', customers=customers)
    except Exception as e:
        return str(e)


@app.route('/db_predict_run', methods=['GET'])
def db_predict_run():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE PredictionResult = 'Pending'")
        pending_customers = cursor.fetchall()
        
        if pending_customers:
            prediction_pipeline = PredictionPipeline()
            
            for cust in pending_customers:
                data_dict = {
                    'RowNumber': [cust['RowNumber']],
                    'CustomerId': [cust['CustomerId']],
                    'Surname': [cust['Surname']],
                    'CreditScore': [cust['CreditScore']],
                    'Geography': [cust['Geography']],
                    'Gender': [cust['Gender']],
                    'Age': [cust['Age']],
                    'Tenure': [cust['Tenure']],
                    'Balance': [cust['Balance']],
                    'NumOfProducts': [cust['NumOfProducts']],
                    'HasCrCard': [cust['HasCrCard']],
                    'IsActiveMember': [cust['IsActiveMember']],
                    'EstimatedSalary': [cust['EstimatedSalary']],
                    'CRIB_Status': [cust['CRIB_Status']],
                    'Recent_Loan_Rejected': [cust['Recent_Loan_Rejected']],
                    'Digital_Banking_User': [cust['Digital_Banking_User']]
                }
                df = pd.DataFrame(data_dict)
                pred = prediction_pipeline.predict(df)
                pred_label = 'Yes' if pred[0] == 1 else 'No'
                
                cursor.execute(
                    "UPDATE customers SET PredictionResult = ? WHERE CustomerId = ?",
                    (pred_label, cust['CustomerId'])
                )
            conn.commit()
            
        conn.close()
        return db_dashboard()
    except Exception as e:
        return str(e)


@app.route('/db_reset', methods=['GET'])
def db_reset():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE customers SET PredictionResult = 'Pending'")
        conn.commit()
        conn.close()
        return db_dashboard()
    except Exception as e:
        return str(e)


@app.route('/api/predict', methods=['POST'])
@require_api_key
def api_predict():
    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        required_fields = ['Age', 'Tenure_Years', 'Account_Balance_LKR', 'District', 'CRIB_Status', 'Recent_Loan_Rejected', 'Digital_Banking_User']
        missing = [f for f in required_fields if f not in req_data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400
            
        data_dict = {
            'RowNumber': [req_data.get('RowNumber', 0)],
            'CustomerId': [req_data.get('CustomerId', 0)],
            'Surname': [req_data.get('Surname', '')],
            'Age': [int(req_data['Age'])],
            'Tenure_Years': [int(req_data['Tenure_Years'])],
            'Account_Balance_LKR': [float(req_data['Account_Balance_LKR'])],
            'District': [req_data['District']],
            'CRIB_Status': [req_data['CRIB_Status']],
            'Recent_Loan_Rejected': [int(req_data['Recent_Loan_Rejected'])],
            'Digital_Banking_User': [int(req_data['Digital_Banking_User'])]
        }
        
        df = pd.DataFrame(data_dict)
        prediction_pipeline = PredictionPipeline()
        pred = prediction_pipeline.predict(df)
        pred_label = 'Yes' if pred[0] == 1 else 'No'
        
        return jsonify({
            "CustomerId": req_data.get('CustomerId', 0),
            "Surname": req_data.get('Surname', ''),
            "Churn_Prediction": pred_label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
@app.route('/api/predict_batch', methods=['POST'])
@require_api_key
def api_predict_batch():
    try:
        req_data = request.get_json()
        if not req_data or not isinstance(req_data, list):
            return jsonify({"error": "JSON payload must be a list of customer objects"}), 400
            
        results = []
        prediction_pipeline = PredictionPipeline()
        
        for item in req_data:
            data_dict = {
                'RowNumber': [item.get('RowNumber', 0)],
                'CustomerId': [item.get('CustomerId', 0)],
                'Surname': [item.get('Surname', '')],
                'Age': [int(item.get('Age', 40))],
                'Tenure_Years': [int(item.get('Tenure_Years', 3))],
                'Account_Balance_LKR': [float(item.get('Account_Balance_LKR', 0.0))],
                'District': [item.get('District', 'Colombo')],
                'CRIB_Status': [item.get('CRIB_Status', 'Clean')],
                'Recent_Loan_Rejected': [int(item.get('Recent_Loan_Rejected', 0))],
                'Digital_Banking_User': [int(item.get('Digital_Banking_User', 1))]
            }
            df = pd.DataFrame(data_dict)
            pred = prediction_pipeline.predict(df)
            pred_label = 'Yes' if pred[0] == 1 else 'No'
            
            results.append({
                "CustomerId": item.get('CustomerId', 0),
                "Surname": item.get('Surname', ''),
                "Churn_Prediction": pred_label
            })
            
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)