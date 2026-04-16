import sys
import os

# Add src to Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from src.main import main
    main()
