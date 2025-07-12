#!/usr/bin/env python3
"""
Script to fix Pydantic V2 compatibility issues
"""

import os
import re
from pathlib import Path

def fix_pydantic_v2_file(file_path):
    """Fix Pydantic V2 compatibility issues in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix imports
        content = re.sub(
            r'from pydantic import ([^,\n]*,\s*)?validator([,\s])',
            r'from pydantic import \1field_validator\2',
            content
        )
        
        # Fix validator decorators
        content = re.sub(
            r'@validator\("([^"]+)"(?:, always=True)?\)',
            r'@field_validator("\1")',
            content
        )
        
        # Fix validator functions
        content = re.sub(
            r'def (validate_[^(]+)\(cls, v(?:, values)?\):',
            r'@classmethod\n    def \1(cls, v, info=None):',
            content
        )
        
        # Fix values access
        content = re.sub(
            r'values\.get\("([^"]+)"',
            r'(info.data if info else {}).get("\1"',
            content
        )
        
        # Fix Config
        content = re.sub(
            r'allow_population_by_field_name = True',
            r'populate_by_name = True',
            content
        )
        
        content = re.sub(
            r'schema_extra = {',
            r'json_schema_extra = {',
            content
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all Python files in the models directory"""
    models_dir = Path("src/models")
    
    if not models_dir.exists():
        print("❌ src/models directory not found")
        return
    
    for py_file in models_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        print(f"🔧 Processing {py_file}")
        fix_pydantic_v2_file(py_file)

if __name__ == "__main__":
    main()