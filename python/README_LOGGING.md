# LiveTranslate Logging

This document explains how to view logs and save transcription/translation data from the LiveTranslate system.

## Docker Container Logs

When you run `start_livetranslate.bat`, the Docker containers run in detached mode (background). Here's how to view their output:

### View Current Logs
```bash
docker-compose logs
```

### Follow Logs in Real-time
```bash
docker-compose logs -f
```

### View Logs for Specific Service
```bash
docker-compose logs transcription-server
docker-compose logs translation-server
```

### View Last N Lines
```bash
docker-compose logs --tail=50
```

## CSV Data Logging

The `logging_client.py` script connects to both WebSocket servers and saves all transcriptions and translations to CSV files.

### Features

- **Automatic CSV Creation**: Creates timestamped CSV files for each session
- **Dual Logging**: Saves both transcriptions and translations separately  
- **Real-time Display**: Shows transcriptions and translations in the console
- **Summary Report**: Displays session statistics when stopped

### Usage

Start the CSV logger:
```bash
python python/logging_client.py
```

Or use the interactive menu in `start_livetranslate.bat` (option 1).

### CSV File Structure

#### Transcriptions CSV
- **Filename**: `logs/transcriptions_YYYYMMDD_HHMMSS.csv`
- **Columns**:
  - `timestamp`: ISO format timestamp
  - `text`: Transcribed text
  - `is_final`: Whether this is a final transcription
  - `confidence`: Confidence score (if available)

#### Translations CSV  
- **Filename**: `logs/translations_YYYYMMDD_HHMMSS.csv`
- **Columns**:
  - `timestamp`: ISO format timestamp
  - `original_text`: Original text to translate
  - `translated_text`: Translated result
  - `source_lang`: Source language
  - `target_lang`: Target language

### Command Line Options

- `--transcription-server`: WebSocket URL for transcription server (default: ws://localhost:8765)
- `--translation-server`: WebSocket URL for translation server (default: ws://localhost:8010)
- `--output-dir`: Directory to save CSV files (default: logs)

### Example

```bash
# Start logger with custom output directory
python logging_client.py --output-dir my_logs

# Connect to remote servers
python logging_client.py --transcription-server ws://remote-host:8765 --translation-server ws://remote-host:8010
```

## Integration with Audio Client

The CSV logger works automatically with any audio client. When you:

1. Start the Docker services (`start_livetranslate.bat`)
2. Start the CSV logger (`python logging_client.py`)
3. Start the audio client (`python audio_client.py`)

The audio client will stream audio → transcription server → CSV logger will capture transcriptions → send to translation server → CSV logger will capture translations.

## File Management

- CSV files are created in the `logs/` directory by default
- Each session creates new timestamped files
- Files are UTF-8 encoded to support international characters
- Headers are automatically added to new CSV files

## Troubleshooting

### "Connection refused" errors
- Make sure Docker services are running: `docker-compose ps`
- Check if ports are accessible: `docker-compose logs`

### Empty CSV files
- Ensure audio client is sending data to the transcription server
- Check WebSocket connections in the logger output

### Character encoding issues
- CSV files use UTF-8 encoding
- Open with Excel using "Data > From Text/CSV" and select UTF-8 