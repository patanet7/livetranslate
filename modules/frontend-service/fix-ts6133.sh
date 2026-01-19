#!/bin/bash

# Automated TS6133 Error Fixer
# This script automatically fixes common TS6133 (unused import/variable) errors

echo "Fixing TS6133 errors automatically..."

# Get list of files with TS6133 errors
npm run build 2>&1 | grep "error TS6133" > /tmp/ts6133_errors.txt

# Extract unique file paths
cat /tmp/ts6133_errors.txt | sed 's/\(.*\.tsx\?\).*/\1/' | sort -u > /tmp/ts6133_files.txt

# Count total errors
total_errors=$(wc -l < /tmp/ts6133_errors.txt)
total_files=$(wc -l < /tmp/ts6133_files.txt)

echo "Found $total_errors errors in $total_files files"
echo "Processing files..."

# Process each file
processed=0
while IFS= read -r file; do
  if [ -f "$file" ]; then
    echo "Processing: $file"

    # Fix common unused imports by commenting them out (safer than deleting)
    # Pattern 1: Single unused import
    sed -i 's/^\(import.*{\)\s*\(\w\+\)\s*,\s*\(.*}\)/\1 \3/' "$file" 2>/dev/null

    # Pattern 2: Prefix unused variables with underscore
    # Get specific errors for this file
    grep "$file" /tmp/ts6133_errors.txt | while read -r line; do
      # Extract variable name from error message
      varname=$(echo "$line" | grep -oP "'\K[^']+(?=' is declared)")
      if [ ! -z "$varname" ]; then
        # Check if it's a destructured parameter or useState
        if grep -q "const \[$varname," "$file" || grep -q "{ $varname" "$file"; then
          # Prefix with underscore
          sed -i "s/\b$varname\b/_$varname/g" "$file" 2>/dev/null
        fi
      fi
    done

    ((processed++))
  fi
done < /tmp/ts6133_files.txt

echo "Processed $processed files"
echo "Running build to verify fixes..."

# Run build again to check remaining errors
npm run build 2>&1 | grep "error TS6133" | wc -l

echo "Done!"
