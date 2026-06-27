import requests

print("=" * 65)
print("  Testing Sri Lankan Banking Churn Model (with API Key Auth)")
print("=" * 65)

API_URL = "http://127.0.0.1:5000/api/predict"
VALID_KEY = "BOC-CHURN-SECRET-KEY-2026"

sample_data = {
    "CustomerId": 15620007,
    "Surname": "Senanayake",
    "Age": 42,
    "Tenure_Years": 10,
    "Account_Balance_LKR": 1200000.0,
    "District": "Colombo",
    "CRIB_Status": "Clean",
    "Recent_Loan_Rejected": 0,
    "Digital_Banking_User": 1
}

# 1. Test case: No API key
print("\n[TEST] Requesting without API Key...")
r = requests.post(API_URL, json=sample_data)
if r.status_code == 401:
    print("  ✅ PASS: Blocked with 401 Unauthorized as expected.")
else:
    print(f"  ❌ FAIL: Got status code {r.status_code} (expected 401).")

# 2. Test case: Invalid API key
print("\n[TEST] Requesting with Invalid API Key...")
headers_invalid = {"X-API-Key": "WRONG-KEY"}
r = requests.post(API_URL, json=sample_data, headers=headers_invalid)
if r.status_code == 401:
    print("  ✅ PASS: Blocked with 401 Unauthorized as expected.")
else:
    print(f"  ❌ FAIL: Got status code {r.status_code} (expected 401).")

# 3. Test case: Valid API key via X-API-Key header
print("\n[TEST] Requesting with Valid API Key (X-API-Key Header)...")
headers_valid = {"X-API-Key": VALID_KEY}
r = requests.post(API_URL, json=sample_data, headers=headers_valid)
if r.status_code == 200:
    res = r.json()
    print(f"  ✅ PASS: Authenticated successfully! Churn prediction: {res.get('Churn_Prediction')}")
else:
    print(f"  ❌ FAIL: Authentication failed with status code {r.status_code}. Response: {r.text}")

# 4. Test case: Valid API key via Authorization Bearer header
print("\n[TEST] Requesting with Valid API Key (Authorization Bearer Header)...")
headers_bearer = {"Authorization": f"Bearer {VALID_KEY}"}
r = requests.post(API_URL, json=sample_data, headers=headers_bearer)
if r.status_code == 200:
    res = r.json()
    print(f"  ✅ PASS: Authenticated successfully! Churn prediction: {res.get('Churn_Prediction')}")
else:
    print(f"  ❌ FAIL: Authentication failed with status code {r.status_code}. Response: {r.text}")

print("\n" + "=" * 65)
