"""
Sri Lankan Banking Customer Churn Data Generator
================================================
Based on real Sri Lankan banking research:

KEY FACTS USED:
- National avg monthly salary: LKR 55,000 (median: LKR 35,000) [DCS Labour Survey 2023]
- Western Province (Colombo/Gampaha) ~40% of GDP, higher salaries
- NPL/CRIB default rate peaked at 13.9% in Q3 2023 [CBSL]
- Loan rejection rate elevated in 2023 due to economic crisis
- Digital banking: HNB ~802K active digital users; BOC ~47% digital txns
- Digital banking users churn LESS (digital stickiness)
- Low balance / zero balance = high churn risk
- Colombo district: ~USD 6,500-7,700 per capita GDP (highest)
- Remote districts (Mullaitivu, Kilinochchi, Mannar): low income, low digital adoption
- Age: Banking customers 18-70, peak 30-45

CHURN DRIVERS (Sri Lanka specific):
1. CRIB Defaulted → very high churn (bank service restricted)
2. Recent Loan Rejected → high churn (customer leaves for another bank)
3. Zero/Low Account Balance → dormant account = churn
4. Non-digital user in urban areas → switching risk
5. Young customers (18-25) → volatile, likely to switch for better rates
6. Short tenure (0-2 years) → not yet "sticky"
7. Northern/Eastern districts → lower financial inclusion, higher churn
8. High balance + long tenure → very loyal, low churn
"""

import pandas as pd
import numpy as np

np.random.seed(2024)  # Use 2024 for reproducibility tied to report year
NUM_SAMPLES = 10000   # Larger dataset for better model accuracy

# =============================================================================
# 1. DISTRICT CONFIGURATION
#    - Population weights based on Census 2022 estimates
#    - Income multiplier: Colombo = 1.0 (benchmark), remote = 0.35
#    - Digital adoption: Urban high, rural low
#    - CRIB default tendency: Higher in economically stressed districts
# =============================================================================
district_config = {
    # District         Pop_weight  Income_mult  Digital_prob  CRIB_default_extra
    "Colombo":        (0.200,      1.00,        0.72,         0.00),
    "Gampaha":        (0.170,      0.90,        0.65,         0.00),
    "Kalutara":       (0.070,      0.75,        0.55,         0.02),
    "Kandy":          (0.075,      0.70,        0.52,         0.02),
    "Matale":         (0.025,      0.58,        0.38,         0.03),
    "Nuwara Eliya":   (0.030,      0.50,        0.30,         0.04),
    "Galle":          (0.055,      0.72,        0.55,         0.02),
    "Matara":         (0.045,      0.65,        0.48,         0.02),
    "Hambantota":     (0.025,      0.58,        0.40,         0.03),
    "Jaffna":         (0.035,      0.62,        0.50,         0.03),
    "Kilinochchi":    (0.008,      0.42,        0.28,         0.06),
    "Mannar":         (0.007,      0.40,        0.25,         0.07),
    "Vavuniya":       (0.010,      0.45,        0.30,         0.05),
    "Mullaitivu":     (0.006,      0.38,        0.22,         0.07),
    "Batticaloa":     (0.025,      0.50,        0.35,         0.05),
    "Trincomalee":    (0.018,      0.52,        0.38,         0.04),
    "Ampara":         (0.022,      0.48,        0.33,         0.05),
    "Kurunegala":     (0.058,      0.65,        0.48,         0.02),
    "Puttalam":       (0.035,      0.58,        0.40,         0.03),
    "Anuradhapura":   (0.030,      0.55,        0.38,         0.03),
    "Polonnaruwa":    (0.015,      0.52,        0.35,         0.04),
    "Badulla":        (0.030,      0.52,        0.35,         0.04),
    "Moneragala":     (0.018,      0.45,        0.28,         0.05),
    "Ratnapura":      (0.035,      0.55,        0.38,         0.03),
    "Kegalle":        (0.028,      0.60,        0.42,         0.03),
}

districts      = list(district_config.keys())
pop_weights    = np.array([v[0] for v in district_config.values()])
income_mult    = {d: v[1] for d, v in district_config.items()}
digital_prob   = {d: v[2] for d, v in district_config.items()}
crib_extra     = {d: v[3] for d, v in district_config.items()}
pop_weights   /= pop_weights.sum()

# =============================================================================
# 2. GENERATE DISTRICT ASSIGNMENT
# =============================================================================
district_arr = np.random.choice(districts, size=NUM_SAMPLES, p=pop_weights)

# =============================================================================
# 3. AGE
#    Sri Lankan bank customers: mostly 25-55
#    Younger (18-25) = university/first job, higher churn
#    Older (56-70)   = retirement, more stable
# =============================================================================
age = np.random.normal(38, 12, NUM_SAMPLES).astype(int)
age = np.clip(age, 18, 72)

# =============================================================================
# 4. TENURE (Years at bank)
#    After economic crisis many switched, so more short tenure customers
#    Max realistic tenure 25 years
# =============================================================================
tenure = np.random.exponential(4.5, NUM_SAMPLES).astype(int)
tenure = np.clip(tenure, 0, 25)

# =============================================================================
# 5. ACCOUNT BALANCE (LKR)
#    Research-based LKR ranges:
#    - National avg monthly salary: LKR 55,000
#    - Savings account balance typically 1-6 months salary
#    - Colombo: higher, remote districts: lower
#    - 22% zero balance accounts (dormant / very low income)
#
#    Realistic tiers:
#    - Zero balance:          22% of customers
#    - Low  (<50,000):        18% — daily wage workers
#    - Mid  (50k-300k):       35% — salaried employees
#    - High (300k-1.5M):      18% — upper-middle class
#    - Very High (>1.5M):      7% — high net worth
# =============================================================================
balance = np.zeros(NUM_SAMPLES)
for i, d in enumerate(district_arr):
    mult = income_mult[d]
    r = np.random.rand()
    if r < 0.22:              # Zero balance
        balance[i] = 0
    elif r < 0.40:            # Low balance
        balance[i] = np.random.uniform(1000, 50000) * mult
    elif r < 0.75:            # Mid balance (salaried)
        balance[i] = np.random.lognormal(11.5, 0.6) * mult  # ~100k center
    elif r < 0.93:            # High balance
        balance[i] = np.random.lognormal(13.0, 0.5) * mult  # ~450k center
    else:                     # Very high balance
        balance[i] = np.random.lognormal(14.5, 0.6) * mult  # ~2M center
balance = np.round(balance, 0).astype(float)

# =============================================================================
# 6. CRIB STATUS
#    National NPL peaked at 13.9% (Q3 2023)
#    But CRIB "Defaulted" ≠ all NPLs; some are borderline
#    Estimating ~16% Defaulted nationally (incl. restructured)
#    Remote districts have higher default tendency
# =============================================================================
crib_status = []
for d in district_arr:
    base_default = 0.135 + crib_extra[d]  # 13.5% base + district extra
    crib = np.random.choice(["Clean", "Defaulted"], p=[1 - base_default, base_default])
    crib_status.append(crib)
crib_status = np.array(crib_status)

# =============================================================================
# 7. RECENT LOAN REJECTED
#    2023: Elevated rejection due to economic crisis tightening
#    Urban districts: banks stricter but more applications
#    Base rejection rate ~14%, higher in remote/defaulted profiles
# =============================================================================
loan_rejected = []
for i, d in enumerate(district_arr):
    base_reject = 0.12
    if crib_status[i] == "Defaulted":
        base_reject += 0.35  # Defaulted CRIB = very high rejection
    if balance[i] < 50000:
        base_reject += 0.08  # Low balance = higher rejection
    base_reject = min(base_reject, 0.95)
    loan_rejected.append(np.random.choice([0, 1], p=[1 - base_reject, base_reject]))
loan_rejected = np.array(loan_rejected)

# =============================================================================
# 8. DIGITAL BANKING USER
#    BOC: ~47% digital txn rate
#    HNB: 802K active digital users
#    National estimate: ~55-60% of bank customers use digital banking
#    Urban >> Rural in digital adoption
# =============================================================================
digital_user = []
for d in district_arr:
    d_prob = digital_prob[d]
    digital_user.append(np.random.choice([0, 1], p=[1 - d_prob, d_prob]))
digital_user = np.array(digital_user)

# =============================================================================
# 9. CHURN PROBABILITY — RESEARCH-BASED RULES
#
#    Churn = customer closing/leaving/going dormant at the bank
#    Base churn rate for Sri Lankan banks: ~12-15%
#    (industry estimate; digital banking stickiness reduces churn)
#
#    Evidence-based factors:
#    a) CRIB Defaulted       → HIGHEST predictor (+35-40%)
#       Banks restrict services to defaulters → they leave
#    b) Recent Loan Rejected → STRONG predictor (+28-32%)
#       Customers switch to another bank for loans
#    c) Zero Balance         → HIGH predictor (+20-25%)
#       Dormant accounts often precede closure
#    d) Non-Digital User     → MODERATE predictor (+8-12%)
#       Less engaged, easier to switch
#    e) Short Tenure (0-2yr) → MODERATE predictor (+10-15%)
#       Not yet "sticky", still exploring banks
#    f) Young Age (18-25)    → SLIGHT predictor (+6-8%)
#       More willing to switch for better digital features
#    g) Remote District      → SLIGHT predictor (+5-8%)
#       Limited service availability, may prefer closer branch
#    h) High Balance (>1M)   → PROTECTOR (-12-15%)
#       Privileged banking, personal relationship manager
#    i) Long Tenure (>8yr)   → PROTECTOR (-8-10%)
#       Strong loyalty, inertia
#    j) Digital + Urban      → PROTECTOR (-5-8%)
#       Sticky ecosystem (mobile payments, etc.)
# =============================================================================
churn_prob = np.full(NUM_SAMPLES, 0.12)  # 12% base churn rate

# Factor a: CRIB Defaulted — strongest predictor
churn_prob += np.where(crib_status == "Defaulted", 0.38, 0.0)

# Factor b: Recent loan rejected — second strongest
churn_prob += np.where(loan_rejected == 1, 0.30, 0.0)

# Factor c: Zero balance — dormant account
churn_prob += np.where(balance == 0, 0.22, 0.0)
# Very low balance (under 20k)
churn_prob += np.where((balance > 0) & (balance < 20000), 0.10, 0.0)
# High balance protection
churn_prob -= np.where(balance > 1000000, 0.12, 0.0)
churn_prob -= np.where(balance > 500000, 0.06, 0.0)

# Factor d: Non-digital user
churn_prob += np.where(digital_user == 0, 0.10, 0.0)
# Digital users in urban areas are sticky
churn_prob -= np.where(
    (digital_user == 1) & (np.isin(district_arr, ["Colombo", "Gampaha", "Kandy", "Galle"])),
    0.06, 0.0
)

# Factor e: Tenure protection/risk
churn_prob += np.where(tenure <= 1, 0.12, 0.0)
churn_prob += np.where((tenure >= 2) & (tenure <= 3), 0.05, 0.0)
churn_prob -= np.where(tenure > 8, 0.09, 0.0)
churn_prob -= np.where(tenure > 15, 0.05, 0.0)

# Factor f: Age risk
churn_prob += np.where(age < 25, 0.08, 0.0)
churn_prob += np.where((age >= 25) & (age < 30), 0.03, 0.0)
# Peak loyalty age 40-55
churn_prob -= np.where((age >= 40) & (age <= 55), 0.03, 0.0)

# Factor g: Remote district risk
remote_districts = ["Kilinochchi", "Mullaitivu", "Mannar", "Moneragala", "Nuwara Eliya", "Batticaloa", "Ampara"]
churn_prob += np.where(np.isin(district_arr, remote_districts), 0.06, 0.0)

# Clip probabilities to valid range
churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05, NUM_SAMPLES), 0.01, 0.97)

# Generate binary churn outcome
churn = (np.random.rand(NUM_SAMPLES) < churn_prob).astype(int)

# =============================================================================
# 10. BUILD DATAFRAME AND SAVE
# =============================================================================
df = pd.DataFrame({
    'Age': age,
    'Tenure_Years': tenure,
    'Account_Balance_LKR': balance,
    'District': district_arr,
    'CRIB_Status': crib_status,
    'Recent_Loan_Rejected': loan_rejected,
    'Digital_Banking_User': digital_user,
    'Churn': churn
})

df.to_csv('sl_bank_data.csv', index=False)

# Print summary stats
print(f"✅ sl_bank_data.csv generated with {len(df)} rows")
print(f"\n📊 Churn Distribution:")
print(df['Churn'].value_counts())
print(f"   Churn Rate: {df['Churn'].mean()*100:.1f}%")
print(f"\n📊 CRIB Status:")
print(df['CRIB_Status'].value_counts())
print(f"\n📊 Balance Stats (LKR):")
print(f"   Mean:   Rs. {df['Account_Balance_LKR'].mean():,.0f}")
print(f"   Median: Rs. {df['Account_Balance_LKR'].median():,.0f}")
print(f"   Zero balance: {(df['Account_Balance_LKR']==0).sum()} customers")
print(f"\n📊 Digital Banking Users: {df['Digital_Banking_User'].mean()*100:.1f}%")
print(f"\n📊 District Top 5:")
print(df['District'].value_counts().head(5))
print(f"\n📊 Age: Mean={df['Age'].mean():.1f}, Median={df['Age'].median():.0f}")
print(f"📊 Tenure: Mean={df['Tenure_Years'].mean():.1f} years")
