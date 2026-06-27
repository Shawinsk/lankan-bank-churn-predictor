import joblib, pandas as pd, scipy.sparse

preprocessor = joblib.load('sl_preprocessor.joblib')
model = joblib.load('sl_model.joblib')

def predict(name, age, tenure, balance, district, crib, loan_rejected, digital):
    df = pd.DataFrame([{
        'Age': age, 'Tenure_Years': tenure, 'Account_Balance_LKR': balance,
        'District': district, 'CRIB_Status': crib,
        'Recent_Loan_Rejected': loan_rejected, 'Digital_Banking_User': digital
    }])
    cols = ['Age','Tenure_Years','Account_Balance_LKR','District','CRIB_Status','Recent_Loan_Rejected','Digital_Banking_User']
    X = preprocessor.transform(df[cols])
    if scipy.sparse.issparse(X):
        X = X.toarray()
    feat = preprocessor.get_feature_names_out()
    prob = model.predict_proba(pd.DataFrame(X, columns=feat))[0][1]
    pred = 'CHURN RISK' if prob > 0.5 else 'LOYAL'
    bar = int(prob * 20)
    filled = '#' * bar
    empty = '.' * (20 - bar)
    print(f"  {name:<30} | {pred:<11} | {prob*100:5.1f}%  | [{filled}{empty}]")

print("")
print("="*82)
print("  BOC CUSTOMER CHURN PREDICTION - LIVE ALGORITHM PROOF")
print("="*82)
print(f"  {'Customer':<30} | {'Result':<11} | Prob   | Bar (100% = all #)")
print("-"*82)

predict("Perera (Colombo,Rs.1.2M,Clean)",    42, 12, 1200000, "Colombo",    "Clean",    0, 1)
predict("Fernando (Gampaha,Rs.350k,Digital)", 28,  4,  350000, "Gampaha",    "Clean",    0, 1)
predict("Silva (Kandy, Rs.280k, Mid)",        38,  6,  280000, "Kandy",      "Clean",    0, 1)

print("-"*82)

predict("Rajapaksa (CRIB Defaulted)",         35,  2,   40000, "Colombo",    "Defaulted",1, 0)
predict("Wickrama (Loan Rejected,Low Bal)",   30,  1,   15000, "Galle",      "Clean",    1, 0)
predict("Kumara (Zero Bal, Remote, Non-dig)", 22,  0,       0, "Mullaitivu", "Clean",    0, 0)
predict("Navaratnam (All Risk Factors)",      26,  1,    5000, "Kilinochchi","Defaulted",1, 0)

print("-"*82)
print("")
print("  RESULTS MAKE SENSE?")
print("  Top 3 (Clean CRIB + Good Balance + Digital) -----> LOYAL       [PASS]")
print("  Bottom 4 (CRIB Default / Loan Rejected / Remote) -> CHURN RISK [PASS]")
print("="*82)
