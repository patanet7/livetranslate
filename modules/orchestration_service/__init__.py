"""
Orchestration Service Package
Import alias for orchestration-service directory
"""
import sys
from pathlib import Path

# Add orchestration-service to path
service_path = Path(__file__).parent.parent / "orchestration-service"
if str(service_path) not in sys.path:
    sys.path.insert(0, str(service_path))

# Import from orchestration-service/src
try:
    from src import *
except ImportError:
    pass
