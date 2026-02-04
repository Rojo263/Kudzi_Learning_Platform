#!/usr/bin/env python3
"""
Kudzi Chatbot Deployment Helper Script
This script helps with deployment tasks and testing.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

def print_header():
    """Print a beautiful header for the deployment script."""
    print("=" * 60)
    print("🚀 KUDZI CHATBOT - DEPLOYMENT HELPER")
    print("=" * 60)
    print()

def check_requirements():
    """Check if all required files exist."""
    print("📋 Checking deployment requirements...")
    
    required_files = [
        "kudzigemi6.py",
        "requirements.txt",
        ".env",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files found!")
        return True

def check_environment():
    """Check if environment variables are set."""
    print("\n🔑 Checking environment variables...")
    
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   These will need to be set in your deployment platform.")
        return False
    else:
        print("✅ Environment variables configured!")
        return True

def test_app_locally():
    """Test the app locally before deployment."""
    print("\n🧪 Testing app locally...")
    
    try:
        # Check if streamlit is installed
        result = subprocess.run([sys.executable, "-m", "streamlit", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Streamlit is installed")
        else:
            print("❌ Streamlit not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    except Exception as e:
        print(f"❌ Error checking Streamlit: {e}")
        return False
    
    return True

def show_deployment_options():
    """Show available deployment options."""
    print("\n🌐 Available Deployment Options:")
    print("1. 🚀 Streamlit Cloud (FREE - Recommended)")
    print("2. 🐳 Docker")
    print("3. ☁️  Heroku")
    print("4. 🚂 Railway")
    print("5. 🔧 Local Network")
    
    print("\n📖 For detailed instructions, see DEPLOYMENT.md")

def open_streamlit_cloud():
    """Open Streamlit Cloud in browser."""
    print("\n🌐 Opening Streamlit Cloud...")
    try:
        webbrowser.open("https://share.streamlit.io")
        print("✅ Streamlit Cloud opened in your browser!")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
        print("   Please visit: https://share.streamlit.io")

def show_git_status():
    """Show current Git status."""
    print("\n📊 Git Repository Status:")
    try:
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout.strip():
                print("📝 Files ready for commit:")
                for line in result.stdout.strip().split('\n'):
                    if line:
                        status = line[:2]
                        filename = line[3:]
                        print(f"   {status} {filename}")
            else:
                print("✅ Working directory clean")
        else:
            print("❌ Git repository not found")
    except Exception as e:
        print(f"❌ Error checking Git status: {e}")

def main():
    """Main deployment helper function."""
    print_header()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please fix the missing requirements before deploying.")
        return
    
    # Check environment
    check_environment()
    
    # Test locally
    if test_app_locally():
        print("✅ App is ready for local testing!")
    
    # Show Git status
    show_git_status()
    
    # Show deployment options
    show_deployment_options()
    
    # Interactive menu
    while True:
        print("\n" + "=" * 40)
        print("🎯 What would you like to do?")
        print("1. 🚀 Deploy to Streamlit Cloud")
        print("2. 🧪 Test app locally")
        print("3. 📊 Check Git status")
        print("4. 📖 View deployment guide")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            open_streamlit_cloud()
        elif choice == "2":
            print("\n🧪 Starting local test...")
            print("   The app will open in your browser.")
            print("   Press Ctrl+C to stop the server.")
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", "kudzigemi6.py"])
            except KeyboardInterrupt:
                print("\n✅ Local test stopped.")
        elif choice == "3":
            show_git_status()
        elif choice == "4":
            try:
                with open("DEPLOYMENT.md", "r", encoding="utf-8") as f:
                    print("\n" + "=" * 60)
                    print("📖 DEPLOYMENT GUIDE")
                    print("=" * 60)
                    print(f.read())
            except Exception as e:
                print(f"❌ Could not read DEPLOYMENT.md: {e}")
        elif choice == "5":
            print("\n🎉 Good luck with your deployment!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
