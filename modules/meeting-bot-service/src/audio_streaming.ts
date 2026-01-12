/**
 * Audio Streaming Module
 *
 * Streams audio from the Google Meet bot to the orchestration service
 * for real-time transcription and translation.
 */

import { WebSocket } from 'ws';
import { Logger } from 'winston';

export interface AudioStreamConfig {
  orchestrationUrl: string;  // WebSocket URL for orchestration service
  botId: string;
  userId: string;
  sampleRate?: number;
  channels?: number;
}

export class AudioStreamer {
  private ws: WebSocket | null = null;
  private config: AudioStreamConfig;
  private logger: Logger;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(config: AudioStreamConfig, logger: Logger) {
    this.config = {
      sampleRate: 16000,
      channels: 1,
      ...config
    };
    this.logger = logger;
  }

  /**
   * Connect to the orchestration service WebSocket
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.logger.info('Connecting to orchestration service', {
          url: this.config.orchestrationUrl,
          botId: this.config.botId
        });

        this.ws = new WebSocket(this.config.orchestrationUrl);

        this.ws.on('open', () => {
          this.logger.info('Audio stream connected to orchestration');
          this.reconnectAttempts = 0;

          // Send initial configuration
          this.send({
            type: 'config',
            botId: this.config.botId,
            userId: this.config.userId,
            sampleRate: this.config.sampleRate,
            channels: this.config.channels
          });

          resolve();
        });

        this.ws.on('message', (data: Buffer) => {
          this.handleMessage(data);
        });

        this.ws.on('error', (error) => {
          this.logger.error('WebSocket error', { error: error.message });
          reject(error);
        });

        this.ws.on('close', () => {
          this.logger.warn('Audio stream closed');
          this.handleReconnect();
        });

      } catch (error: any) {
        this.logger.error('Failed to connect audio stream', { error: error.message });
        reject(error);
      }
    });
  }

  /**
   * Stream audio chunk to orchestration service
   */
  async streamAudio(audioData: Buffer): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.logger.warn('WebSocket not ready, buffering audio chunk');
      return;
    }

    try {
      this.send({
        type: 'audio',
        data: audioData.toString('base64'),
        timestamp: Date.now()
      });
    } catch (error: any) {
      this.logger.error('Failed to stream audio chunk', { error: error.message });
    }
  }

  /**
   * Handle messages from orchestration service (transcriptions, translations)
   */
  private handleMessage(data: Buffer): void {
    try {
      const message = JSON.parse(data.toString());

      switch (message.type) {
        case 'transcription':
          this.logger.info('Received transcription', {
            text: message.text,
            language: message.language
          });
          break;

        case 'translation':
          this.logger.info('Received translation', {
            originalText: message.originalText,
            translatedText: message.translatedText,
            targetLanguage: message.targetLanguage
          });
          break;

        case 'error':
          this.logger.error('Received error from orchestration', {
            error: message.error
          });
          break;

        default:
          this.logger.debug('Unknown message type', { type: message.type });
      }
    } catch (error: any) {
      this.logger.error('Failed to parse message', { error: error.message });
    }
  }

  /**
   * Send JSON message over WebSocket
   */
  private send(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  /**
   * Handle WebSocket reconnection
   */
  private async handleReconnect(): Promise<void> {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.logger.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    this.logger.info(`Attempting to reconnect in ${delay}ms`, {
      attempt: this.reconnectAttempts,
      maxAttempts: this.maxReconnectAttempts
    });

    await new Promise(resolve => setTimeout(resolve, delay));

    try {
      await this.connect();
    } catch (error) {
      // Will retry on next close event
    }
  }

  /**
   * Close the audio stream
   */
  async close(): Promise<void> {
    if (this.ws) {
      this.logger.info('Closing audio stream');
      this.ws.close();
      this.ws = null;
    }
  }
}

/**
 * Factory for creating audio streamers
 */
export function createAudioStreamer(
  config: AudioStreamConfig,
  logger: Logger
): AudioStreamer {
  return new AudioStreamer(config, logger);
}
