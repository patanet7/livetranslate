#!/bin/bash

# Test script to verify microphone audio capture
# Records 5 seconds of audio from MacBook Pro Microphone (device :2)

OUTPUT_FILE="test_audio.wav"

echo "🎤 Testing microphone capture..."
echo "Recording 5 seconds from device :2 (MacBook Pro Microphone)"
echo "Speak into your microphone now!"
echo ""

# Record 5 seconds of audio
ffmpeg -f avfoundation -i ":6" -ac 1 -ar 16000 -t 5 "$OUTPUT_FILE" -y

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Recording complete!"
    echo "📁 File saved: $OUTPUT_FILE"
    echo ""
    echo "📊 Audio file info:"
    ffmpeg -i "$OUTPUT_FILE" 2>&1 | grep "Duration\|Stream"
    echo ""
    echo "🔊 File size: $(du -h "$OUTPUT_FILE" | cut -f1)"

    # Check if file has actual audio (not silence)
    SIZE=$(wc -c < "$OUTPUT_FILE")
    if [ $SIZE -gt 1000 ]; then
        echo "✅ Audio data captured (file size looks good)"
    else
        echo "⚠️  File is very small - might be silence or permission issue"
    fi
else
    echo ""
    echo "❌ Recording failed!"
    echo "Check microphone permissions in System Settings → Privacy & Security → Microphone"
fi
