"""
SSL Certificate Fix for Corporate Networks

This script helps resolve SSL certificate verification issues when 
running the performance analysis tool behind corporate firewalls.
"""

import os
import ssl
import certifi

def fix_ssl_for_corporate_network():
    """
    Apply SSL certificate fixes for corporate networks
    
    WARNING: Option 3 disables SSL verification which is less secure.
    Only use for testing in trusted corporate networks.
    """
    
    print("="*70)
    print("SSL Certificate Fix Options")
    print("="*70)
    print()
    print("Choose a fix option:")
    print()
    print("1. Use certifi package (Recommended)")
    print("   - Uses Mozilla's CA bundle")
    print("   - Secure and widely compatible")
    print()
    print("2. Use Windows certificate store")
    print("   - Uses your Windows system certificates")
    print("   - Includes corporate certificates")
    print()
    print("3. Disable SSL verification (Testing only)")
    print("   - NOT SECURE - Only for testing")
    print("   - Bypasses certificate checks")
    print()
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        apply_certifi_fix()
    elif choice == "2":
        apply_windows_cert_fix()
    elif choice == "3":
        apply_disable_ssl_fix()
    else:
        print("Invalid choice. Exiting.")
        return
    
    print()
    print("="*70)
    print("Now try running your analysis again:")
    print("  python example_usage.py")
    print("="*70)


def apply_certifi_fix():
    """Option 1: Use certifi package"""
    print()
    print("Applying Fix 1: Using certifi package...")
    
    # Set environment variables
    cert_path = certifi.where()
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    
    print(f"✓ Set SSL_CERT_FILE={cert_path}")
    print(f"✓ Set REQUESTS_CA_BUNDLE={cert_path}")
    print()
    print("To make this permanent, add to your PowerShell profile:")
    print(f'  $env:SSL_CERT_FILE="{cert_path}"')
    print(f'  $env:REQUESTS_CA_BUNDLE="{cert_path}"')


def apply_windows_cert_fix():
    """Option 2: Use Windows certificates"""
    print()
    print("Applying Fix 2: Using Windows certificate store...")
    print()
    print("Installing python-certifi-win32...")
    
    import subprocess
    try:
        subprocess.check_call(['pip', 'install', 'python-certifi-win32'])
        print("✓ Installed python-certifi-win32")
        print()
        print("This package automatically uses Windows certificate store.")
        print("Now try running your analysis again.")
    except Exception as e:
        print(f"✗ Error installing: {e}")
        print()
        print("Try manually: pip install python-certifi-win32")


def apply_disable_ssl_fix():
    """Option 3: Disable SSL verification (not recommended)"""
    print()
    print("⚠️  WARNING: This disables SSL verification!")
    print("   Only use for testing in trusted corporate networks.")
    print()
    confirm = input("Type 'YES' to confirm: ").strip()
    
    if confirm != "YES":
        print("Cancelled.")
        return
    
    print()
    print("Creating ssl_disable.py...")
    
    # Create a file to import before running analysis
    with open('ssl_disable.py', 'w') as f:
        f.write("""# SSL Verification Disable (TESTING ONLY)
# Import this before running analysis

import ssl
import os

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Also set environment variables
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''

print("⚠️  SSL verification DISABLED - For testing only!")
""")
    
    print("✓ Created ssl_disable.py")
    print()
    print("Now modify example_usage.py to add at the top:")
    print("  import ssl_disable")
    print()
    print("Or run from Python:")
    print("  python -c 'import ssl_disable; import example_usage'")


if __name__ == "__main__":
    fix_ssl_for_corporate_network()

