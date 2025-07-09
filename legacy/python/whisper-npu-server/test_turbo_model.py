#!/usr/bin/env python3
"""
Test whisper-large-v3-turbo model loading
"""

import sys
sys.path.append('.')
from server import ModelManager
import logging

logging.basicConfig(level=logging.INFO)

def main():
    mm = ModelManager()
    print('Available models:', mm.list_models())

    try:
        pipeline = mm.load_model('whisper-large-v3-turbo')
        print('✅ whisper-large-v3-turbo loaded successfully!')
        print(f'Device: {mm.device}')
        return True
    except Exception as e:
        print('❌ Failed to load whisper-large-v3-turbo:', e)
        return False

if __name__ == "__main__":
    main() 