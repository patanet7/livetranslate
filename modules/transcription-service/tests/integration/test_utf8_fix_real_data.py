#!/usr/bin/env python3
"""
Quick test: Apply UTF8BoundaryFixer to real Chinese transcription results
"""

import sys

sys.path.insert(0, "src")

from utf8_boundary_fixer import UTF8BoundaryFixer

print("\n" + "=" * 80)
print("UTF8BoundaryFixer - Real Chinese Transcription Data")
print("=" * 80)

# These are the ACTUAL results we got from the test
real_results = [
    {"text": "院子门口不远�", "is_final": False},
    {"text": "�就是一个地铁站,这是一个美丽而神奇的景象,树上长满了又大又甜的�", "is_final": True},
]

print("\n--- BEFORE UTF8BoundaryFixer ---")
for i, result in enumerate(real_results, 1):
    print(f"Result {i} (is_final={result['is_final']}): '{result['text']}'")

# Combine results
combined_before = "".join(r["text"] for r in real_results)
print(f"\nCombined BEFORE: '{combined_before}'")
print(f"Contains �: {'\ufffd' in combined_before}")
print(f"Count �: {combined_before.count('\ufffd')}")

# Now apply the fixer
fixer = UTF8BoundaryFixer()

print("\n--- AFTER UTF8BoundaryFixer ---")
cleaned_results = []
for i, result in enumerate(real_results, 1):
    cleaned_text = fixer.fix_boundaries(result["text"])
    cleaned_results.append(cleaned_text)
    print(f"Result {i} (is_final={result['is_final']}): '{cleaned_text}'")

# Combine cleaned results
combined_after = "".join(cleaned_results)
print(f"\nCombined AFTER: '{combined_after}'")
print(f"Contains �: {'\ufffd' in combined_after}")
print(f"Count �: {combined_after.count('\ufffd')}")

print("\n--- COMPARISON ---")
print(f"Before: {len(combined_before)} chars, {combined_before.count('\ufffd')} � chars")
print(f"After:  {len(combined_after)} chars, {combined_after.count('\ufffd')} � chars")
print(f"Removed: {combined_before.count('\ufffd') - combined_after.count('\ufffd')} � chars")

if "\ufffd" not in combined_after:
    print("\n✅ SUCCESS: All � characters removed!")
else:
    print(f"\n⚠️  Still has {combined_after.count('\ufffd')} � characters")

print("\n" + "=" * 80)
