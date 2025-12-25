#!/usr/bin/env python3
"""
Simple script to set up your Anthropic API key.
Run this script and paste your API key when prompted.
"""

import os

CREDENTIALS_PATH = "/home/user/credentials.yml"

print("=" * 50)
print("SALON MULTI-AGENT SYSTEM - API KEY SETUP")
print("=" * 50)
print()
print("This will save your Anthropic API key to:")
print(f"  {CREDENTIALS_PATH}")
print()
print("Your key starts with 'sk-ant-...'")
print()

api_key = input("Paste your Anthropic API key here: ").strip()

if not api_key:
    print("No key entered. Exiting.")
    exit(1)

if not api_key.startswith("sk-ant-"):
    print("Warning: Key doesn't start with 'sk-ant-'. Are you sure this is correct?")
    confirm = input("Continue anyway? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Exiting.")
        exit(1)

# Write the credentials file
content = f"""# API Keys for Salon Multi-Agent System
anthropic: '{api_key}'
"""

with open(CREDENTIALS_PATH, 'w') as f:
    f.write(content)

print()
print("✓ API key saved successfully!")
print()
print("You can now run the salon multi-agent system.")
