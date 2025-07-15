#!/usr/bin/env python3
"""
Quick test to see if audio router can be imported and what routes it has
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path("src")
sys.path.insert(0, str(src_path))

try:
    print("Testing audio router import...")
    from routers.audio import router as audio_router
    print(f"✅ Audio router imported successfully")
    print(f"Router type: {type(audio_router)}")
    
    # Check if router has routes
    if hasattr(audio_router, 'routes'):
        print(f"Number of routes: {len(audio_router.routes)}")
        for i, route in enumerate(audio_router.routes):
            print(f"  Route {i+1}: {route.methods} {route.path}")
    else:
        print("❌ Router has no routes attribute")
        
except Exception as e:
    print(f"❌ Failed to import audio router: {e}")
    import traceback
    traceback.print_exc()