# LiveTranslate Translation Client

This is a simple WebSocket client that can connect to the LiveTranslate translation server and send text for translation between English and Chinese.

## Features

- Interactive mode for real-time translation
- File streaming mode to translate a text file line by line
- Automatically detects language (English/Chinese) and translates accordingly

## Requirements

- Python 3.7+
- websockets library
- asyncio library

## Usage

### Interactive Mode

Run the client in interactive mode to translate text in real-time:

```bash
python translation_client.py
```

Or specify a different server:

```bash
python translation_client.py --server ws://your-server-address:8010
```

### File Streaming Mode

Stream a text file line by line to the translation server:

```bash
python translation_client.py --file path/to/your/text_file.txt
```

You can also adjust the delay between lines:

```bash
python translation_client.py --file path/to/your/text_file.txt --delay 2.0
```

## Command Line Arguments

- `--server`: WebSocket server URL (default: ws://localhost:8010)
- `--file`: Path to a text file to translate line by line
- `--delay`: Delay between lines when streaming a file in seconds (default: 1.0)

## Example

1. Start the translation server:
   ```bash
   python translation_server.py
   ```

2. In another terminal, start the client:
   ```bash
   python translation_client.py
   ```

3. Type text to translate:
   ```
   > Hello, how are you?
   Translation: 你好，你好吗？
   
   > 我喜欢编程
   Translation: I like programming
   ```

4. Or stream a file:
   ```bash
   python translation_client.py --file test_translation.txt
   ```

## Notes

- Make sure the translation server is running before connecting the client
- The client automatically handles connection errors and will disconnect cleanly 