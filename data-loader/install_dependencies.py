import subprocess
import sys
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install all required dependencies for the NYC Data Loader"""
    try:
        # Read requirements from requirements.txt
        requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        
        if not os.path.exists(requirements_file):
            logger.error(f"Requirements file not found: {requirements_file}")
            return False
            
        with open(requirements_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
        logger.info(f"Found {len(requirements)} packages to install")
        
        # Install each package individually to better handle errors
        for package in requirements:
            logger.info(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {str(e)}")
                
        logger.info("Dependency installation completed")
        return True
        
    except Exception as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting dependency installation...")
    success = install_dependencies()
    
    if success:
        logger.info("All dependencies installed successfully")
    else:
        logger.error("Failed to install all dependencies")
        sys.exit(1) 