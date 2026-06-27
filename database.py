import sqlite3
import os

DB_PATH = 'churn.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Drop table if CRIB_Status column does not exist (schema migration)
    try:
        cursor.execute("SELECT CRIB_Status FROM customers LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, drop to migrate
        cursor.execute("DROP TABLE IF EXISTS customers")
        conn.commit()
    
    # Create customers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            RowNumber INTEGER PRIMARY KEY AUTOINCREMENT,
            CustomerId INTEGER UNIQUE,
            Surname TEXT,
            CreditScore INTEGER,
            Geography TEXT,
            Gender TEXT,
            Age INTEGER,
            Tenure INTEGER,
            Balance REAL,
            NumOfProducts INTEGER,
            HasCrCard INTEGER,
            IsActiveMember INTEGER,
            EstimatedSalary REAL,
            Bank TEXT,
            Branch TEXT,
            CRIB_Status TEXT DEFAULT 'Clean',
            Recent_Loan_Rejected INTEGER DEFAULT 0,
            Digital_Banking_User INTEGER DEFAULT 1,
            PredictionResult TEXT DEFAULT 'Pending'
        )
    ''')
    
    # Check if table is empty
    cursor.execute('SELECT COUNT(*) FROM customers')
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Seed realistic Sri Lankan customer data
        mock_data = [
            (15620001, 'Perera', 619, 'Colombo', 'Female', 42, 2, 0.0, 1, 1, 1, 10108.88, 'Bank of Ceylon (BOC)', 'BOC - Kollupitiya', 'Clean', 0, 1, 'Pending'),
            (15620002, 'Silva', 502, 'Colombo', 'Female', 35, 8, 150000.50, 2, 1, 0, 113931.57, 'Sampath Bank', 'Sampath Bank - Colombo Super Branch (Navam Mawatha)', 'Clean', 0, 1, 'Pending'),
            (15620003, 'Fernando', 720, 'Colombo', 'Male', 40, 1, 85000.75, 1, 1, 1, 89650.00, 'Commercial Bank', 'Commercial Bank - Kollupitiya', 'Defaulted', 1, 0, 'Pending'),
            (15620004, 'Rajapaksa', 480, 'Galle', 'Female', 55, 4, 320000.20, 1, 0, 0, 145000.40, 'Hatton National Bank (HNB)', 'HNB - World Trade Center', 'Clean', 0, 1, 'Pending'),
            (15620005, 'Gunawardena', 850, 'Colombo', 'Male', 29, 3, 0.0, 2, 1, 1, 250000.00, 'People\'s Bank', 'People\'s Bank - Slave Island', 'Clean', 0, 1, 'Pending'),
            (15620006, 'Senanayake', 590, 'Gampaha', 'Male', 48, 6, 180000.10, 1, 1, 0, 75000.00, 'Bank of Ceylon (BOC)', 'BOC - Negombo Super Grade', 'Clean', 0, 0, 'Pending')
        ]
        
        cursor.executemany('''
            INSERT INTO customers (CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Bank, Branch, CRIB_Status, Recent_Loan_Rejected, Digital_Banking_User, PredictionResult)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', mock_data)
        
        conn.commit()
        
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized and seeded successfully!")
