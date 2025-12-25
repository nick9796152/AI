# SALON MULTI-AGENT SYSTEM
# STEP 1: DATABASE SETUP
# ***

# Goal: Create a SQLite database with salon transaction data
# Replace the sample data with your actual CSV export

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import random
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# =============================================================================
# SAMPLE DATA GENERATION (Replace with your actual data import)
# =============================================================================

def generate_sample_salon_data(n_customers=200, n_transactions=1500):
    """
    Generate sample salon transaction data.
    Replace this with: pd.read_csv('your_export.csv') or pd.read_excel('your_export.xlsx')
    """

    random.seed(42)
    np.random.seed(42)

    # Customer IDs
    customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(1, n_customers + 1)]

    # Service categories and items with prices
    services = {
        'Haircuts': [
            ("Men's Haircut", 25.00),
            ("Women's Haircut", 35.00),
            ("Kids Haircut", 18.00),
            ("Bang Trim", 10.00),
        ],
        'Color': [
            ("Color Retouch", 70.00),
            ("Full Color", 95.00),
            ("Highlights & Lowlights", 150.00),
            ("Balayage", 200.00),
            ("Toner", 40.00),
            ("Color Correction", 250.00),
        ],
        'Treatments': [
            ("Deep Conditioning", 35.00),
            ("Keratin Treatment", 250.00),
            ("Scalp Treatment", 45.00),
            ("Olaplex Treatment", 50.00),
        ],
        'Styling': [
            ("Blowout", 45.00),
            ("Updo", 75.00),
            ("Bridal Styling", 150.00),
            ("Special Event Style", 85.00),
        ],
        'Wax': [
            ("Eyebrow Wax", 15.00),
            ("Lip Wax", 10.00),
            ("Full Face Wax", 45.00),
        ],
        'Extensions': [
            ("Tape-In Extensions", 350.00),
            ("Extension Maintenance", 100.00),
        ]
    }

    payment_methods = ['Card', 'Cash', 'Gift Card']
    card_brands = ['Visa', 'Mastercard', 'American Express', 'Discover', None]
    devices = ['iPhone', 'iPad', 'Square Terminal', 'Android']

    # Generate transactions
    transactions = []
    transaction_id = 1

    # Assign customer behavior profiles
    customer_profiles = {}
    for cust_id in customer_ids:
        profile = random.choice(['regular', 'occasional', 'vip', 'at_risk', 'new'])
        primary_service = random.choice(list(services.keys()))
        customer_profiles[cust_id] = {
            'profile': profile,
            'primary_service': primary_service,
            'visit_frequency': {
                'regular': (21, 45),
                'occasional': (45, 90),
                'vip': (14, 30),
                'at_risk': (90, 180),
                'new': (30, 60)
            }[profile]
        }

    # Generate transactions over the past 2 years
    end_date = datetime(2025, 7, 25)
    start_date = end_date - timedelta(days=730)

    for cust_id in customer_ids:
        profile = customer_profiles[cust_id]
        current_date = start_date + timedelta(days=random.randint(0, 60))

        while current_date < end_date:
            # Number of services in this visit
            n_services = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

            visit_time = datetime.combine(
                current_date.date(),
                datetime.strptime(f"{random.randint(9, 18)}:{random.choice(['00', '15', '30', '45'])}:00", "%H:%M:%S").time()
            )

            # Select services
            selected_categories = [profile['primary_service']]
            if n_services > 1:
                other_cats = [c for c in services.keys() if c != profile['primary_service']]
                selected_categories.extend(random.sample(other_cats, min(n_services - 1, len(other_cats))))

            visit_transaction_id = f"TXN{str(transaction_id).zfill(8)}"

            for category in selected_categories[:n_services]:
                item, base_price = random.choice(services[category])

                # Price variation
                price = base_price * random.uniform(0.95, 1.05)

                payment = random.choice(payment_methods)
                card = random.choice(card_brands) if payment == 'Card' else None

                transactions.append({
                    'customer_id': cust_id,
                    'transaction_id': visit_transaction_id,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'time': visit_time.strftime('%H:%M:%S'),
                    'itemization_type': 'Service',
                    'category': category,
                    'item': item,
                    'gross_sales': round(price, 2),
                    'payment_method': payment,
                    'card_brand': card,
                    'device_name': random.choice(devices)
                })

                transaction_id += 1

            # Next visit based on profile
            min_days, max_days = profile['visit_frequency']
            current_date += timedelta(days=random.randint(min_days, max_days))

    return pd.DataFrame(transactions)


# =============================================================================
# CREATE DATABASE
# =============================================================================

def create_salon_database(df, db_path='data/salon.db'):
    """Create SQLite database with salon data"""

    conn = sqlite3.connect(db_path)

    # Drop existing views first (so we can recreate them)
    conn.execute("DROP VIEW IF EXISTS customer_summary")
    conn.execute("DROP VIEW IF EXISTS service_popularity")
    conn.execute("DROP VIEW IF EXISTS monthly_revenue")

    # Main transactions table
    df.to_sql('transactions', conn, if_exists='replace', index=False)

    # Get the max date from the data for calculating days_since_last_visit
    max_date = df['date'].max()
    print(f"Using reference date: {max_date}")

    # Create customer summary view
    customer_summary_query = f"""
    CREATE VIEW IF NOT EXISTS customer_summary AS
    SELECT
        customer_id,
        COUNT(DISTINCT transaction_id) as total_visits,
        COUNT(*) as total_services,
        ROUND(SUM(gross_sales), 2) as total_spent,
        ROUND(AVG(gross_sales), 2) as avg_service_price,
        MIN(date) as first_visit,
        MAX(date) as last_visit,
        CAST(julianday('{max_date}') - julianday(MAX(date)) AS INTEGER) as days_since_last_visit,
        COUNT(DISTINCT category) as unique_categories
    FROM transactions
    GROUP BY customer_id
    """
    conn.execute(customer_summary_query)

    # Create service popularity view
    service_popularity_query = """
    CREATE VIEW IF NOT EXISTS service_popularity AS
    SELECT
        category,
        item,
        COUNT(*) as times_purchased,
        COUNT(DISTINCT customer_id) as unique_customers,
        ROUND(SUM(gross_sales), 2) as total_revenue,
        ROUND(AVG(gross_sales), 2) as avg_price
    FROM transactions
    GROUP BY category, item
    ORDER BY total_revenue DESC
    """
    conn.execute(service_popularity_query)

    # Create monthly revenue view
    monthly_revenue_query = """
    CREATE VIEW IF NOT EXISTS monthly_revenue AS
    SELECT
        strftime('%Y-%m', date) as month,
        category,
        COUNT(DISTINCT customer_id) as unique_customers,
        COUNT(*) as total_services,
        ROUND(SUM(gross_sales), 2) as revenue
    FROM transactions
    GROUP BY strftime('%Y-%m', date), category
    ORDER BY month DESC
    """
    conn.execute(monthly_revenue_query)

    conn.commit()
    conn.close()

    print(f"Database created at: {db_path}")
    print(f"Total transactions: {len(df)}")
    print(f"Unique customers: {df['customer_id'].nunique()}")


# =============================================================================
# LOAD AND CLEAN REAL DATA
# =============================================================================

def load_real_salon_data(csv_path):
    """
    Load and clean the real salon data from Square export
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize column names (lowercase with underscores)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # Clean gross_sales - remove $ and convert to float
    df['gross_sales'] = df['gross_sales'].replace('[\$,]', '', regex=True).astype(float)

    # Handle missing/null categories
    df['category'] = df['category'].fillna('Other')
    df['item'] = df['item'].fillna('Custom Service')
    df['itemization_type'] = df['itemization_type'].fillna('Service')

    # Ensure date is in proper format - handle errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Remove rows with invalid dates
    invalid_dates = df['date'].isna().sum()
    if invalid_dates > 0:
        print(f"Warning: Removing {invalid_dates} rows with invalid dates")
        df = df.dropna(subset=['date'])

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    print(f"Loaded {len(df)} transactions")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    print(f"Categories: {df['category'].unique().tolist()}")

    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Path to real salon data
    REAL_DATA_PATH = "../Sales Mostly Cleaned - sales (1).csv"

    # Check if real data exists, otherwise use sample
    if os.path.exists(REAL_DATA_PATH):
        print("Loading REAL salon data...")
        df = load_real_salon_data(REAL_DATA_PATH)
    else:
        print("Real data not found. Generating sample data...")
        print(f"(Looking for: {REAL_DATA_PATH})")
        df = generate_sample_salon_data(n_customers=200, n_transactions=1500)

    print("\nData Preview:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")

    print("\nCategories and counts:")
    print(df['category'].value_counts())

    # Create the database
    create_salon_database(df)

    # Verify
    conn = sqlite3.connect('data/salon.db')
    print("\n--- Customer Summary Sample ---")
    print(pd.read_sql("SELECT * FROM customer_summary ORDER BY total_spent DESC LIMIT 5", conn))
    print("\n--- Service Popularity ---")
    print(pd.read_sql("SELECT * FROM service_popularity LIMIT 10", conn))
    print("\n--- Monthly Revenue ---")
    print(pd.read_sql("SELECT * FROM monthly_revenue LIMIT 10", conn))
    conn.close()
