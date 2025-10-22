#!/bin/bash

# Test script to record HIGH QUALITY audio (48kHz original)
# To compare with the 16kHz version

OUTPUT_FILE="test_audio_hq.wav"

echo "🎤 Testing HIGH QUALITY microphone capture..."
echo "Recording 5 seconds at ORIGINAL 48kHz quality"
echo "Speak into your microphone now!"
echo ""

# Record at ORIGINAL quality (48kHz)
ffmpeg -f avfoundation -i ":6" -ac 1 -t 5 "$OUTPUT_FILE" -y

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Recording complete!"
    echo "📁 File saved: $OUTPUT_FILE"
    echo ""
    echo "📊 Audio file info:"
    ffmpeg -i "$OUTPUT_FILE" 2>&1 | grep "Duration\|Stream"
    echo ""
    echo "🔊 File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo ""
    echo "Compare this to test_audio.wav (16kHz):"
    echo "  - test_audio_hq.wav = 48kHz (high quality)"
    echo "  - test_audio.wav    = 16kHz (Whisper quality)"
fi
