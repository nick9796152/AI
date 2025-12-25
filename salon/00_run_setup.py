# SALON MULTI-AGENT SYSTEM
# STEP 0: RUN THIS FIRST TO SET UP EVERYTHING
# ***

"""
This script sets up the entire salon multi-agent system:
1. Creates the transaction database with sample data
2. Creates the service knowledge base (vector DB)
3. Tests the system

To use your own data:
1. Export your salon data as CSV
2. Modify 01_setup_salon_database.py to load your CSV instead of sample data
3. Re-run this setup script
"""

import os
import subprocess
import sys

def run_script(script_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)

    result = subprocess.run([sys.executable, script_name], capture_output=False)

    if result.returncode != 0:
        print(f"Error running {script_name}")
        return False
    return True

def main():
    # Change to salon directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("\n" + "="*60)
    print("SALON MULTI-AGENT SYSTEM - SETUP")
    print("="*60)

    # Step 1: Create database
    print("\n[1/3] Setting up salon transaction database...")
    if not run_script("01_setup_salon_database.py"):
        print("Database setup failed!")
        return

    # Step 2: Create service knowledge base
    print("\n[2/3] Creating service knowledge base (RAG)...")
    if not run_script("02_create_service_knowledge_base.py"):
        print("Knowledge base setup failed!")
        return

    # Step 3: Test the system
    print("\n[3/3] Testing the multi-agent system...")
    print("(This may take a moment as it initializes all agents)")

    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("""
Next steps:

1. To use the system interactively, run:
   python 04_interactive_demo.py

2. To import your own data:
   - Export your Square/salon POS data as CSV
   - Edit 01_setup_salon_database.py to load your CSV
   - Re-run this setup script

3. To customize services:
   - Edit the SALON_SERVICES variable in 02_create_service_knowledge_base.py
   - Re-run this setup script

Files created:
   - data/salon.db (transaction database)
   - data/salon_services_vectordb/ (service knowledge base)
""")

if __name__ == "__main__":
    main()
