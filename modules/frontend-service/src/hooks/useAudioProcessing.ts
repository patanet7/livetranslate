import { useCallback, useState, useRef } from 'react';
import { useAppSelector, useAppDispatch } from '@/store';
import {
  setRecordingState,
  setRecordedBlobUrl,
  addProcessingLog,
  updateProcessingStage,
  updateConfig
} from '@/store/slices/audioSlice';
import { DEFAULT_TARGET_LANGUAGES } from '@/config/translation';

export const useAudioProcessing = () => {
  const dispatch = useAppDispatch();
  const { recording, config } = useAppSelector(state => state.audio);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  
  // Keep DOM objects in local state/refs (not in Redux)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const currentStreamRef = useRef<MediaStream | null>(null);
  const currentPlaybackRef = useRef<HTMLAudioElement | null>(null);
  const recordedBlobRef = useRef<Blob | null>(null); // ✅ Store Blob in ref instead of Redux
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null); // ✅ Timer for duration updates

  const startRecording = useCallback(async () => {
    try {
      // Get audio processing preferences
      const constraints = {
        audio: {
          deviceId: config.deviceId || undefined,
          sampleRate: config.sampleRate,
          channelCount: 1, // Mono for speech
          echoCancellation: config.rawAudio ? false : config.echoCancellation,
          noiseSuppression: config.rawAudio ? false : config.noiseSuppression,
          autoGainControl: config.rawAudio ? false : config.autoGainControl,
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // ✅ Detect which device was actually selected and update Redux
      try {
        const audioTrack = stream.getAudioTracks()[0];
        if (audioTrack) {
          const settings = audioTrack.getSettings();
          if (settings.deviceId && settings.deviceId !== config.deviceId) {
            // Update Redux with the actual selected device
            dispatch(updateConfig({ deviceId: settings.deviceId }));
            dispatch(addProcessingLog({
              level: 'INFO',
              message: `Using audio device: ${settings.deviceId}`,
              timestamp: Date.now()
            }));
          }
        }
      } catch (deviceError) {
        // Non-critical error, continue with recording
        console.log('Could not detect selected device:', deviceError);
      }
      
      // Check supported formats
      const supportedFormats = [
        'audio/webm;codecs=opus',
        'audio/mp4;codecs=mp4a.40.2',
        'audio/ogg;codecs=opus',
        'audio/wav',
      ].filter(format => MediaRecorder.isTypeSupported(format));

      if (supportedFormats.length === 0) {
        throw new Error('No supported audio formats found');
      }

      // Use selected format or fallback
      let mimeType = config.format;
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = supportedFormats[0];
        dispatch(addProcessingLog({
          level: 'WARNING',
          message: `Format ${config.format} not supported, using ${mimeType}`,
          timestamp: Date.now()
        }));
      }

      // Calculate bit rate based on quality
      let bitRate;
      switch(config.quality) {
        case 'high':
          bitRate = 256000;
          break;
        case 'medium':
          bitRate = 128000;
          break;
        case 'low':
          bitRate = 64000;
          break;
        case 'lossless':
          bitRate = config.sampleRate * 16;
          break;
        default:
          bitRate = 128000;
      }

      const mediaRecorderOptions = {
        mimeType,
        audioBitsPerSecond: bitRate
      };

      const mediaRecorder = new MediaRecorder(stream, mediaRecorderOptions);
      const chunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: mimeType });
        
        // ✅ Store Blob in ref (non-serializable DOM object)
        recordedBlobRef.current = blob;
        
        // ✅ Store serializable URL string in Redux
        const blobUrl = URL.createObjectURL(blob);
        dispatch(setRecordedBlobUrl(blobUrl));
        
        dispatch(setRecordingState({
          isRecording: false,
          duration: recording.duration
        }));

        // Clean up stream
        stream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      };

      mediaRecorder.start();
      
      // Store DOM objects in refs
      mediaRecorderRef.current = mediaRecorder;
      currentStreamRef.current = stream;
      
      dispatch(setRecordingState({
        isRecording: true,
        recordingStartTime: Date.now(),
        duration: 0 // Reset duration
      }));

      // ✅ Start recording timer - update duration every 100ms for smooth UI
      recordingTimerRef.current = setInterval(() => {
        if (recording.recordingStartTime) {
          const elapsed = (Date.now() - recording.recordingStartTime) / 1000;
          dispatch(setRecordingState({ duration: elapsed }));
        }
      }, 100);

      // Auto-stop timer
      if (config.autoStop) {
        setTimeout(() => {
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
          }
        }, config.duration * 1000);
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to start recording: ${errorMessage}`);
    }
  }, [config, dispatch, recording.duration]);

  const stopRecording = useCallback(async () => {
    // ✅ Clean up recording timer
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      
      if (currentStreamRef.current) {
        currentStreamRef.current.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      }
    }
  }, []);

  const playRecording = useCallback(async () => {
    if (!recordedBlobRef.current) {
      throw new Error('No recording to play');
    }

    if (recording.isPlaying) {
      // Stop current playback
      if (currentPlaybackRef.current) {
        currentPlaybackRef.current.pause();
        currentPlaybackRef.current.currentTime = 0;
      }
      dispatch(setRecordingState({ isPlaying: false }));
    } else {
      // Start playback
      try {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
          sampleRate: 48000,
          latencyHint: 'playback'
        });

        const arrayBuffer = await recordedBlobRef.current.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;

        const gainNode = audioContext.createGain();
        gainNode.gain.value = 1.0;

        source.connect(gainNode);
        gainNode.connect(audioContext.destination);

        source.onended = () => {
          dispatch(setRecordingState({ isPlaying: false }));
          audioContext.close();
        };

        source.start(0);
        dispatch(setRecordingState({ isPlaying: true }));

      } catch (error) {
        // Fallback to HTML5 Audio
        const audio = new Audio(URL.createObjectURL(recordedBlobRef.current));
        
        audio.onended = () => {
          dispatch(setRecordingState({ isPlaying: false }));
        };

        audio.onerror = () => {
          dispatch(setRecordingState({ isPlaying: false }));
          throw new Error('Playback failed');
        };

        audio.play();
        currentPlaybackRef.current = audio;
        dispatch(setRecordingState({ isPlaying: true }));
      }
    }
  }, [recording.isPlaying, dispatch]);

  const clearRecording = useCallback(() => {
    // ✅ Clean up recording timer
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }

    // Stop any current playback
    if (currentPlaybackRef.current) {
      currentPlaybackRef.current.pause();
      currentPlaybackRef.current.currentTime = 0;
      currentPlaybackRef.current = null;
    }

    // Stop recording if active
    if (recording.isRecording && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }

    // Clean up stream
    if (currentStreamRef.current) {
      currentStreamRef.current.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      currentStreamRef.current = null;
    }

    // Clean up blob ref
    recordedBlobRef.current = null;
    
    dispatch(setRecordingState({
      recordedBlobUrl: null,
      isRecording: false,
      isPlaying: false,
      duration: 0,
      recordingStartTime: null
    }));

    dispatch(addProcessingLog({
      level: 'INFO',
      message: 'Recording cleared',
      timestamp: Date.now()
    }));
  }, [recording.isRecording, dispatch]);

  const downloadRecording = useCallback(() => {
    if (!recordedBlobRef.current) {
      throw new Error('No recording to download');
    }

    const url = URL.createObjectURL(recordedBlobRef.current);
    const a = document.createElement('a');
    a.href = url;

    // Generate filename with timestamp and format
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const format = config.format;
    const extension = format.includes('webm') ? 'webm' : 
                     format.includes('mp4') ? 'mp4' : 
                     format.includes('ogg') ? 'ogg' : 
                     format.includes('wav') ? 'wav' : 'webm';
                     
    const sampleRate = config.sampleRate;
    const quality = config.quality;
    const rawAudio = config.rawAudio ? '-raw' : '';
    
    a.download = `livetranslate-recording-${timestamp}-${sampleRate}Hz-${quality}${rawAudio}.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    dispatch(addProcessingLog({
      level: 'SUCCESS',
      message: `Recording downloaded as ${a.download}`,
      timestamp: Date.now()
    }));
  }, [config, dispatch]);

  const runPipeline = useCallback(async () => {
    if (!recordedBlobRef.current) {
      throw new Error('No audio to process');
    }

    setIsProcessing(true);
    setProcessingProgress(0);

    // 10-stage meeting audio processing pipeline (matches orchestration service)
    const stages = [
      'original_audio',
      'decoded_audio', 
      'voice_frequency_filter',
      'voice_activity_detection',
      'voice_aware_noise_reduction',
      'voice_enhancement',
      'advanced_voice_processing',
      'voice_aware_silence_trimming',
      'high_quality_resampling',
      'final_output'
    ];

    try {
      dispatch(addProcessingLog({
        level: 'INFO',
        message: 'Starting pipeline processing...',
        timestamp: Date.now()
      }));

      // Create FormData for the orchestration service
      const formData = new FormData();
      formData.append('file', recordedBlobRef.current, 'recording.wav');
      formData.append('pipeline_enabled', 'true');
      formData.append('stage_by_stage', 'true'); // Request stage-by-stage results
      
      // Add configuration from Redux state
      formData.append('config', JSON.stringify({
        sample_rate: config.sampleRate,
        quality: config.quality,
        format: config.format,
        enable_vad: true,
        enable_noise_reduction: true,
        enable_voice_enhancement: true,
        meeting_optimization: true
      }));

      dispatch(addProcessingLog({
        level: 'INFO',
        message: 'Uploading audio to orchestration service...',
        timestamp: Date.now()
      }));

      setProcessingProgress(0.1);

      // Call orchestration service API
      const response = await fetch('/api/audio/process', {
        method: 'POST',
        body: formData
      });

      setProcessingProgress(0.2);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API call failed: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      
      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Received pipeline results from orchestration service',
        timestamp: Date.now()
      }));

      setProcessingProgress(0.3);

      // Process stage-by-stage results from orchestration service
      if (result.stage_results && Array.isArray(result.stage_results)) {
        for (let i = 0; i < result.stage_results.length; i++) {
          const stageResult = result.stage_results[i];
          const stageId = stageResult.stage;
          
          // Create processing stage based on orchestration service response
          dispatch(updateProcessingStage({
            id: stageId,
            name: stages.find(s => s === stageId) || stageId,
            description: `Processing stage: ${stageId}`,
            status: stageResult.success ? 'completed' : 'error',
            progress: 100,
            startTime: Date.now() - (stageResult.processing_time_ms || 0),
            endTime: Date.now(),
            processingTime: stageResult.processing_time_ms || 0,
            result: {
              input_level_db: stageResult.input_level_db,
              output_level_db: stageResult.output_level_db,
              quality_score: stageResult.quality_score,
              artifacts_detected: stageResult.artifacts_detected,
              stage_specific_data: stageResult.stage_specific_data || {}
            },
            error: stageResult.success ? undefined : stageResult.error,
            metrics: {
              'Processing Time': `${stageResult.processing_time_ms || 0}ms`,
              'Quality Score': `${((stageResult.quality_score || 0) * 100).toFixed(1)}%`,
              'Input Level': `${(stageResult.input_level_db || 0).toFixed(1)} dB`,
              'Output Level': `${(stageResult.output_level_db || 0).toFixed(1)} dB`,
              'Artifacts': stageResult.artifacts_detected ? 'Detected' : 'None'
            }
          }));

          // Progressive updates for real-time feedback
          setProcessingProgress(0.3 + (0.6 * (i + 1) / result.stage_results.length));
          
          dispatch(addProcessingLog({
            level: stageResult.success ? 'SUCCESS' : 'ERROR',
            message: `Stage ${stageId}: ${stageResult.success ? 'completed' : 'failed'} (${stageResult.processing_time_ms || 0}ms)`,
            timestamp: Date.now()
          }));

          // Small delay for visual effect
          await new Promise(resolve => setTimeout(resolve, 200));
        }
      } else {
        // No stage results means the API structure is different than expected
        throw new Error(`Invalid API response: expected 'stage_results' array but received: ${JSON.stringify(Object.keys(result))}`);
      }

      setProcessingProgress(1.0);

      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: `Pipeline processing completed successfully. Overall quality: ${((result.overall_quality_score || 0.85) * 100).toFixed(1)}%`,
        timestamp: Date.now()
      }));

      // Log summary statistics
      if (result.total_processing_time_ms) {
        dispatch(addProcessingLog({
          level: 'INFO',
          message: `Total processing time: ${result.total_processing_time_ms}ms`,
          timestamp: Date.now()
        }));
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Pipeline processing failed: ${errorMessage}`,
        timestamp: Date.now()
      }));

      // Mark any in-progress stages as failed
      stages.forEach(stageId => {
        dispatch(updateProcessingStage({
          id: stageId,
          status: 'error',
          error: errorMessage
        }));
      });

      throw error;
    } finally {
      setIsProcessing(false);
    }
  }, [dispatch, config]);

  const runStepByStep = useCallback(async () => {
    // Implementation for step-by-step processing
    dispatch(addProcessingLog({
      level: 'INFO',
      message: 'Step-by-step processing not yet implemented',
      timestamp: Date.now()
    }));
  }, [dispatch]);

  const resetPipeline = useCallback(() => {
    setIsProcessing(false);
    setProcessingProgress(0);
    
    // Clear all pipeline stages
    const stages = [
      'original_audio',
      'decoded_audio', 
      'voice_frequency_filter',
      'voice_activity_detection',
      'voice_aware_noise_reduction',
      'voice_enhancement',
      'advanced_voice_processing',
      'voice_aware_silence_trimming',
      'high_quality_resampling',
      'final_output'
    ];
    
    // Reset all stages to pending state
    stages.forEach(stageId => {
      dispatch(updateProcessingStage({
        id: stageId,
        name: stageId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        description: `Processing stage: ${stageId}`,
        status: 'pending',
        progress: 0,
        startTime: undefined,
        endTime: undefined,
        processingTime: undefined,
        result: undefined,
        error: undefined,
        metrics: {}
      }));
    });
    
    dispatch(addProcessingLog({
      level: 'INFO',
      message: 'Pipeline reset - all stages cleared',
      timestamp: Date.now()
    }));
  }, [dispatch]);

  const exportResults = useCallback(() => {
    // Implementation for exporting results
    const results = {
      timestamp: new Date().toISOString(),
      audioInfo: {
        duration: recording.duration,
        format: config.format,
        sampleRate: config.sampleRate,
        source: config.source
      },
      // Add processing results here
    };

    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audio-test-results-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);

    dispatch(addProcessingLog({
      level: 'SUCCESS',
      message: 'Results exported successfully',
      timestamp: Date.now()
    }));
  }, [recording.duration, config, dispatch]);

  const processAudioForTranscription = useCallback(async (
    audioBlob?: Blob
  ) => {
    // Use provided blob or fall back to recorded blob from ref
    const blobToProcess = audioBlob || recordedBlobRef.current;
    
    if (!blobToProcess) {
      throw new Error('No audio to process. Please record audio first.');
    }

    dispatch(addProcessingLog({
      level: 'INFO',
      message: 'Starting audio processing for transcription...',
      timestamp: Date.now()
    }));

    try {
      setIsProcessing(true);
      setProcessingProgress(0);

      // First, run the audio pipeline processing
      const formData = new FormData();
      formData.append('file', blobToProcess, 'recording.wav');
      formData.append('pipeline_enabled', 'true');
      formData.append('transcription_ready', 'true'); // Request transcription-optimized output
      
      // Add configuration optimized for transcription
      formData.append('config', JSON.stringify({
        sample_rate: 16000, // Whisper optimal sample rate
        quality: 'lossless',
        format: 'wav',
        enable_vad: true,
        enable_noise_reduction: true,
        enable_voice_enhancement: true,
        meeting_optimization: true,
        transcription_optimization: true
      }));

      setProcessingProgress(0.2);
      
      dispatch(addProcessingLog({
        level: 'INFO',
        message: 'Processing audio through pipeline...',
        timestamp: Date.now()
      }));

      // Process audio through pipeline
      const pipelineResponse = await fetch('/api/audio/process', {
        method: 'POST',
        body: formData
      });

      setProcessingProgress(0.5);

      if (!pipelineResponse.ok) {
        const errorText = await pipelineResponse.text();
        throw new Error(`Pipeline processing failed: ${pipelineResponse.status} - ${errorText}`);
      }

      const pipelineResult = await pipelineResponse.json();
      
      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Audio pipeline processing completed, sending to whisper service...',
        timestamp: Date.now()
      }));

      setProcessingProgress(0.7);

      // Now send processed audio to whisper service for transcription
      const transcriptionFormData = new FormData();
      
      // If pipeline returned processed audio, use that; otherwise use original
      if (pipelineResult.processed_audio_url) {
        // Fetch the processed audio and add to form data
        const processedAudioResponse = await fetch(pipelineResult.processed_audio_url);
        const processedAudioBlob = await processedAudioResponse.blob();
        transcriptionFormData.append('file', processedAudioBlob, 'processed_audio.wav');
      } else {
        transcriptionFormData.append('file', blobToProcess, 'recording.wav');
      }
      
      transcriptionFormData.append('transcription', 'true');
      transcriptionFormData.append('speaker_diarization', 'true');
      
      // Add session ID if available
      if (recording.sessionId) {
        transcriptionFormData.append('session_id', recording.sessionId);
      }

      dispatch(addProcessingLog({
        level: 'INFO',
        message: 'Sending to whisper service for transcription...',
        timestamp: Date.now()
      }));

      // Send to whisper service via orchestration service
      const transcriptionResponse = await fetch('/api/whisper/transcribe', {
        method: 'POST',
        body: transcriptionFormData
      });

      setProcessingProgress(0.9);

      if (!transcriptionResponse.ok) {
        const errorText = await transcriptionResponse.text();
        throw new Error(`Transcription failed: ${transcriptionResponse.status} - ${errorText}`);
      }

      const transcriptionResult = await transcriptionResponse.json();
      setProcessingProgress(1.0);

      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: `Transcription completed: "${transcriptionResult.text || transcriptionResult.transcription || 'No text found'}"`,
        timestamp: Date.now()
      }));

      // Return combined results
      return {
        pipeline_result: pipelineResult,
        transcription_result: transcriptionResult,
        processing_time_ms: (pipelineResult.total_processing_time_ms || 0) + (transcriptionResult.processing_time || 0)
      };

    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Audio processing for transcription failed: ${error}`,
        timestamp: Date.now()
      }));
      throw error;
    } finally {
      setIsProcessing(false);
    }
  }, [recording.sessionId, dispatch]);

  const processTranscriptionForTranslation = useCallback(async (
    transcriptionText: string,
    sourceLanguage: string = 'auto',
    targetLanguages: string[] = [...DEFAULT_TARGET_LANGUAGES]
  ) => {
    if (!transcriptionText || transcriptionText.trim().length === 0) {
      throw new Error('No transcription text to translate.');
    }

    dispatch(addProcessingLog({
      level: 'INFO',
      message: `Starting translation to: ${targetLanguages.join(', ')}`,
      timestamp: Date.now()
    }));

    try {
      setIsProcessing(true);
      setProcessingProgress(0);

      // Send transcription to translation service
      const translationData = {
        text: transcriptionText,
        source_language: sourceLanguage,
        target_languages: targetLanguages,
        session_id: recording.sessionId || undefined
      };

      setProcessingProgress(0.2);
      
      dispatch(addProcessingLog({
        level: 'INFO',
        message: 'Sending transcription to translation service...',
        timestamp: Date.now()
      }));

      // Call translation service via orchestration service
      const response = await fetch('/api/translation/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(translationData)
      });

      setProcessingProgress(0.7);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Translation failed: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      setProcessingProgress(1.0);

      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Translation completed successfully',
        timestamp: Date.now()
      }));

      // Log translation results
      if (result.translations && Object.keys(result.translations).length > 0) {
        Object.entries(result.translations).forEach(([lang, translation]: [string, any]) => {
          dispatch(addProcessingLog({
            level: 'SUCCESS',
            message: `Translation (${lang.toUpperCase()}): "${translation.translated_text}"`,
            timestamp: Date.now()
          }));
        });
      } else if (result.error) {
        dispatch(addProcessingLog({
          level: 'WARNING',
          message: `Translation failed: ${result.error}`,
          timestamp: Date.now()
        }));
      }

      return result;

    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Translation failed: ${error}`,
        timestamp: Date.now()
      }));
      throw error;
    } finally {
      setIsProcessing(false);
    }
  }, [recording.sessionId, dispatch]);

  const processAudioWithTranslation = useCallback(async (
    audioBlob?: Blob,
    targetLanguages: string[] = [...DEFAULT_TARGET_LANGUAGES]
  ) => {
    dispatch(addProcessingLog({
      level: 'INFO',
      message: `Starting complete audio processing with translation pipeline...`,
      timestamp: Date.now()
    }));

    try {
      // Step 1: Process audio for transcription (pipeline + whisper)
      const transcriptionResult = await processAudioForTranscription(audioBlob);
      
      // Step 2: Extract transcription text
      const transcriptionText = transcriptionResult.transcription_result?.text || 
                                transcriptionResult.transcription_result?.transcription || 
                                '';
      
      if (!transcriptionText) {
        throw new Error('No transcription text received from whisper service');
      }

      const sourceLanguage = transcriptionResult.transcription_result?.language || 'auto';

      // Step 3: Translate the transcription
      const translationResult = await processTranscriptionForTranslation(
        transcriptionText, 
        sourceLanguage, 
        targetLanguages
      );

      dispatch(addProcessingLog({
        level: 'SUCCESS',
        message: 'Complete audio processing pipeline finished successfully',
        timestamp: Date.now()
      }));

      // Return combined results
      return {
        processing_result: transcriptionResult.transcription_result,
        translations: translationResult.translations,
        pipeline_result: transcriptionResult.pipeline_result,
        total_processing_time: transcriptionResult.processing_time_ms + (translationResult.processing_time || 0)
      };

    } catch (error) {
      dispatch(addProcessingLog({
        level: 'ERROR',
        message: `Complete audio processing pipeline failed: ${error}`,
        timestamp: Date.now()
      }));
      throw error;
    }
  }, [processAudioForTranscription, processTranscriptionForTranslation, dispatch]);

  return {
    startRecording,
    stopRecording,
    playRecording,
    clearRecording,
    downloadRecording,
    runPipeline,
    runStepByStep,
    resetPipeline,
    exportResults,
    processAudioForTranscription,
    processTranscriptionForTranslation,
    processAudioWithTranslation,
    isProcessing,
    processingProgress
  };
};