# Meeting Data Fixtures

Real meeting data for pipeline validation. Each fixture is a JSON file with:

```json
{
  "source": "bot_audio|fireflies|meet_cc",
  "meeting_id": "unique-id",
  "language": "en|zh|ja|ko",
  "segments": [
    {
      "text": "transcribed text",
      "speaker_name": "Alice",
      "start_time": 0.0,
      "end_time": 1.25,
      "is_final": true
    }
  ]
}
```

## Adding fixtures

1. Record from real meetings with `LIVETRANSLATE_RECORD_FIXTURES=1`
2. Export from Fireflies API
3. Scrape from Google Meet CC via MeetCaptionsAdapter

Cover: English, Chinese, Japanese, Korean, mixed-language, multi-speaker.
