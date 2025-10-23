#!/usr/bin/env python3
"""
Comprehensive Multi-Language Sentence Segmentation Test

Tests sentence boundary detection across:
- English, Spanish, French, German
- Japanese, Chinese
- Arabic (with Arabic question mark)
- Hindi (with Devanagari danda)
- Edge cases with quotes and punctuation
"""

import sys
sys.path.insert(0, 'src')

from sentence_segmenter import SentenceSegmenter


def test_multilang_sentence_segmentation():
    """Test sentence segmentation with real-world multi-language examples"""

    segmenter = SentenceSegmenter()

    test_cases = [
        # === ENGLISH ===
        {
            "language": "English",
            "text": "Hello world. How are you? I am fine!",
            "expected_segments": ["Hello world. ", "How are you? ", "I am fine!"],
            "expected_final": [True, True, True]
        },
        {
            "language": "English (quoted)",
            "text": 'He said "Hello." She replied "Hi!"',
            "expected_segments": ['He said "Hello." ', 'She replied "Hi!"'],
            "expected_final": [True, True]
        },
        {
            "language": "English (incomplete)",
            "text": "The quick brown fox",
            "expected_segments": ["The quick brown fox"],
            "expected_final": [False]
        },

        # === SPANISH ===
        {
            "language": "Spanish",
            "text": "¡Hola! ¿Cómo estás? Estoy bien.",
            "expected_segments": ["¡Hola! ", "¿Cómo estás? ", "Estoy bien."],
            "expected_final": [True, True, True]
        },

        # === FRENCH ===
        {
            "language": "French",
            "text": "Bonjour! Comment allez-vous? Je vais bien.",
            "expected_segments": ["Bonjour! ", "Comment allez-vous? ", "Je vais bien."],
            "expected_final": [True, True, True]
        },

        # === GERMAN ===
        {
            "language": "German",
            "text": "Guten Tag! Wie geht es Ihnen? Mir geht es gut.",
            "expected_segments": ["Guten Tag! ", "Wie geht es Ihnen? ", "Mir geht es gut."],
            "expected_final": [True, True, True]
        },

        # === JAPANESE ===
        {
            "language": "Japanese",
            "text": "こんにちは。元気ですか？元気です！",
            "expected_segments": ["こんにちは。", "元気ですか？", "元気です！"],
            "expected_final": [True, True, True]
        },
        {
            "language": "Japanese (incomplete)",
            "text": "今日は良い天気",
            "expected_segments": ["今日は良い天気"],
            "expected_final": [False]
        },

        # === CHINESE ===
        {
            "language": "Chinese (Simplified)",
            "text": "你好。你好吗？我很好！",
            "expected_segments": ["你好。", "你好吗？", "我很好！"],
            "expected_final": [True, True, True]
        },
        {
            "language": "Chinese (Traditional + quotes)",
            "text": "他說「你好。」她回答「嗨！」",
            "expected_segments": ["他說「你好。」", "她回答「嗨！」"],
            "expected_final": [True, True]
        },
        {
            "language": "Chinese (incomplete)",
            "text": "今天天气很好",
            "expected_segments": ["今天天气很好"],
            "expected_final": [False]
        },

        # === ARABIC ===
        {
            "language": "Arabic",
            "text": "مرحبا. كيف حالك؟ أنا بخير.",
            "expected_segments": ["مرحبا. ", "كيف حالك؟ ", "أنا بخير."],
            "expected_final": [True, True, True]
        },

        # === KOREAN ===
        {
            "language": "Korean",
            "text": "안녕하세요. 어떻게 지내세요? 잘 지내요!",
            "expected_segments": ["안녕하세요. ", "어떻게 지내세요? ", "잘 지내요!"],
            "expected_final": [True, True, True]
        },

        # === HINDI ===
        {
            "language": "Hindi (Devanagari)",
            "text": "नमस्ते। आप कैसे हैं? मैं ठीक हूं।",
            "expected_segments": ["नमस्ते। ", "आप कैसे हैं? ", "मैं ठीक हूं।"],
            "expected_final": [True, True, True]
        },

        # === MIXED LANGUAGES ===
        {
            "language": "Mixed (English + Chinese)",
            "text": "Hello world. 你好。How are you? 你好吗？",
            "expected_segments": ["Hello world. ", "你好。", "How are you? ", "你好吗？"],
            "expected_final": [True, True, True, True]
        },

        # === EDGE CASES ===
        {
            "language": "Edge: Multiple quotes",
            "text": 'She said "He said "Hello.""',
            "expected_segments": ['She said "He said "Hello.""'],
            "expected_final": [True]
        },
        {
            "language": "Edge: Parentheses",
            "text": "What is this? (It's a test.)",
            "expected_segments": ["What is this? ", "(It's a test.)"],
            "expected_final": [True, True]
        },
    ]

    print("=" * 100)
    print("COMPREHENSIVE MULTI-LANGUAGE SENTENCE SEGMENTATION TEST")
    print("=" * 100)

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_case in test_cases:
        total_tests += 1
        language = test_case["language"]
        text = test_case["text"]
        expected_segments = test_case["expected_segments"]
        expected_final = test_case["expected_final"]

        # Test segmentation
        segments = segmenter(text)

        # Test is_sentence_end for each segment
        actual_final = [segmenter.is_sentence_end(seg) for seg in segments]

        # Check if test passed
        segments_match = segments == expected_segments
        final_match = actual_final == expected_final
        passed = segments_match and final_match

        if passed:
            passed_tests += 1
            status = "✅"
        else:
            failed_tests.append({
                "language": language,
                "text": text,
                "expected_segments": expected_segments,
                "actual_segments": segments,
                "expected_final": expected_final,
                "actual_final": actual_final
            })
            status = "❌"

        print(f"\n{status} {language}")
        print(f"   Text: '{text}'")
        print(f"   Segments: {segments}")
        print(f"   is_final: {actual_final}")

        if not segments_match:
            print(f"   ⚠️  Expected segments: {expected_segments}")
        if not final_match:
            print(f"   ⚠️  Expected is_final: {expected_final}")

    # === SUMMARY ===
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Failed: {len(failed_tests)} ({len(failed_tests)/total_tests*100:.1f}%)")

    if failed_tests:
        print("\n❌ FAILED TESTS:")
        for i, fail in enumerate(failed_tests, 1):
            print(f"\n{i}. {fail['language']}")
            print(f"   Text: '{fail['text']}'")
            print(f"   Expected: {fail['expected_segments']}")
            print(f"   Got:      {fail['actual_segments']}")
            print(f"   Expected is_final: {fail['expected_final']}")
            print(f"   Got is_final:      {fail['actual_final']}")

    print("\n" + "=" * 100)
    if len(failed_tests) == 0:
        print("✅ ALL TESTS PASSED! Multi-language sentence segmentation working perfectly!")
    else:
        print(f"❌ {len(failed_tests)} TESTS FAILED - See details above")
    print("=" * 100 + "\n")

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = test_multilang_sentence_segmentation()
    exit(0 if success else 1)
