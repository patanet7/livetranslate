/**
 * Professional Audio Level Calculation for Meeting Voice Isolation
 *
 * Provides accurate RMS, peak, and spectral analysis optimized for meeting scenarios.
 * Includes voice activity detection, clipping detection, and quality assessment.
 */

export interface AudioLevelMetrics {
  // Basic Level Metrics
  rms: number; // Root Mean Square (0-1)
  peak: number; // Peak amplitude (0-1)
  rmsDb: number; // RMS in decibels (-60 to 0)
  peakDb: number; // Peak in decibels (-60 to 0)

  // Advanced Metrics
  clipping: number; // Clipping percentage (0-1)
  voiceActivity: number; // Voice activity percentage (0-1)
  spectralCentroid: number; // Frequency brightness (Hz)
  dynamicRange: number; // Peak - RMS difference (dB)

  // Meeting-Specific Metrics
  speechClarity: number; // Speech frequency prominence (0-1)
  backgroundNoise: number; // Background noise level (0-1)
  signalToNoise: number; // SNR estimate (dB)
}

export interface AudioQualityAssessment {
  quality: "excellent" | "good" | "fair" | "poor";
  score: number; // Overall quality score (0-100)
  recommendations: string[];
  issues: string[];
}

/**
 * Calculate comprehensive audio level metrics optimized for meeting scenarios
 */
export function calculateMeetingAudioLevel(
  timeData: Uint8Array,
  frequencyData: Uint8Array,
  sampleRate: number = 16000,
): AudioLevelMetrics {
  // Convert time domain data to normalized float samples [-1, 1]
  const samples: number[] = [];
  for (let i = 0; i < timeData.length; i++) {
    samples.push((timeData[i] - 128) / 128);
  }

  // Calculate RMS (Root Mean Square)
  let rmsSum = 0;
  let peak = 0;
  let voiceActivityCount = 0;
  let clippingCount = 0;

  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    const absSample = Math.abs(sample);

    // RMS calculation
    rmsSum += sample * sample;

    // Peak detection
    peak = Math.max(peak, absSample);

    // Voice activity detection (energy-based)
    if (absSample > 0.01) {
      // -40dB threshold
      voiceActivityCount++;
    }

    // Clipping detection
    if (absSample > 0.95) {
      // -0.4dB threshold
      clippingCount++;
    }
  }

  const rms = Math.sqrt(rmsSum / samples.length);
  const voiceActivity = voiceActivityCount / samples.length;
  const clipping = clippingCount / samples.length;

  // Convert to dB with proper floor (-60dB minimum)
  const rmsDb = rms > 0 ? Math.max(20 * Math.log10(rms), -60) : -60;
  const peakDb = peak > 0 ? Math.max(20 * Math.log10(peak), -60) : -60;

  // Calculate spectral centroid (frequency brightness)
  let spectralSum = 0;
  let magnitudeSum = 0;
  const nyquist = sampleRate / 2;

  for (let i = 0; i < frequencyData.length; i++) {
    const magnitude = frequencyData[i] / 255;
    const frequency = (i / frequencyData.length) * nyquist;
    spectralSum += frequency * magnitude;
    magnitudeSum += magnitude;
  }

  const spectralCentroid = magnitudeSum > 0 ? spectralSum / magnitudeSum : 0;

  // Calculate speech clarity (300-3400Hz range prominence)
  const speechBinStart = Math.floor((300 / nyquist) * frequencyData.length);
  const speechBinEnd = Math.floor((3400 / nyquist) * frequencyData.length);

  let speechEnergy = 0;
  let totalEnergy = 0;

  for (let i = 0; i < frequencyData.length; i++) {
    const magnitude = frequencyData[i] / 255;
    totalEnergy += magnitude;

    if (i >= speechBinStart && i <= speechBinEnd) {
      speechEnergy += magnitude;
    }
  }

  const speechClarity = totalEnergy > 0 ? speechEnergy / totalEnergy : 0;

  // Estimate background noise (low-frequency content)
  const noiseBinEnd = Math.floor((200 / nyquist) * frequencyData.length);
  let noiseEnergy = 0;

  for (let i = 0; i < noiseBinEnd; i++) {
    noiseEnergy += frequencyData[i] / 255;
  }

  const backgroundNoise = noiseEnergy / noiseBinEnd;

  // Calculate dynamic range
  const dynamicRange = peakDb - rmsDb;

  // Estimate Signal-to-Noise Ratio
  const signalToNoise =
    speechClarity > 0 && backgroundNoise > 0
      ? 20 * Math.log10(speechClarity / backgroundNoise)
      : 0;

  return {
    rms,
    peak,
    rmsDb,
    peakDb,
    clipping,
    voiceActivity,
    spectralCentroid,
    dynamicRange,
    speechClarity,
    backgroundNoise,
    signalToNoise,
  };
}

/**
 * Assess audio quality for meeting scenarios with specific recommendations
 */
export function getMeetingAudioQuality(
  metrics: AudioLevelMetrics,
): AudioQualityAssessment {
  const recommendations: string[] = [];
  const issues: string[] = [];
  let score = 100;
  let quality: "excellent" | "good" | "fair" | "poor" = "excellent";

  // Check signal level (optimal range: -12dB to -6dB RMS)
  if (metrics.rmsDb < -40) {
    quality = "poor";
    score -= 30;
    issues.push("Signal level too low");
    recommendations.push("Move closer to microphone or increase input gain");
  } else if (metrics.rmsDb < -30) {
    quality = "fair";
    score -= 15;
    issues.push("Signal level low");
    recommendations.push("Consider increasing input gain slightly");
  } else if (metrics.rmsDb > -6) {
    quality = "fair";
    score -= 10;
    issues.push("Signal level high");
    recommendations.push("Reduce input gain to prevent clipping");
  }

  // Check for clipping (should be < 0.1%)
  if (metrics.clipping > 0.01) {
    quality = "poor";
    score -= 25;
    issues.push("Audio clipping detected");
    recommendations.push("Reduce microphone gain or speaker volume");
  } else if (metrics.clipping > 0.001) {
    quality = "fair";
    score -= 10;
    issues.push("Occasional clipping detected");
    recommendations.push("Slightly reduce input gain");
  }

  // Check voice activity (should be > 10% for active speech)
  if (metrics.voiceActivity < 0.05) {
    score -= 5;
    recommendations.push("Very low voice activity - ensure speaker is audible");
  } else if (metrics.voiceActivity < 0.1) {
    recommendations.push("Low voice activity detected");
  }

  // Check speech clarity (should be > 0.3 for clear speech)
  if (metrics.speechClarity < 0.2) {
    quality = quality === "excellent" ? "good" : "fair";
    score -= 15;
    issues.push("Poor speech frequency balance");
    recommendations.push("Check microphone positioning and room acoustics");
  } else if (metrics.speechClarity < 0.3) {
    score -= 5;
    recommendations.push("Speech clarity could be improved");
  }

  // Check spectral balance for voice (optimal: 500-2000Hz)
  if (metrics.spectralCentroid < 300 || metrics.spectralCentroid > 4000) {
    score -= 10;
    issues.push("Suboptimal frequency balance for voice");
    recommendations.push("Check microphone frequency response");
  }

  // Check dynamic range (should be 6-20dB for natural speech)
  if (metrics.dynamicRange < 3) {
    score -= 10;
    issues.push("Limited dynamic range - over-compressed");
    recommendations.push("Reduce audio processing or compression");
  } else if (metrics.dynamicRange > 25) {
    score -= 5;
    recommendations.push("Wide dynamic range - consider light compression");
  }

  // Check Signal-to-Noise Ratio (should be > 12dB)
  if (metrics.signalToNoise < 6) {
    quality =
      quality === "excellent" ? "good" : quality === "good" ? "fair" : "poor";
    score -= 20;
    issues.push("Poor signal-to-noise ratio");
    recommendations.push(
      "Reduce background noise or move closer to microphone",
    );
  } else if (metrics.signalToNoise < 12) {
    score -= 10;
    issues.push("Background noise present");
    recommendations.push(
      "Consider noise reduction or better microphone positioning",
    );
  }

  // Check background noise level
  if (metrics.backgroundNoise > 0.3) {
    score -= 15;
    issues.push("High background noise");
    recommendations.push("Use noise suppression or find quieter environment");
  } else if (metrics.backgroundNoise > 0.15) {
    score -= 5;
    recommendations.push("Some background noise detected");
  }

  // Final quality assessment based on score
  if (score >= 85) quality = "excellent";
  else if (score >= 70) quality = "good";
  else if (score >= 50) quality = "fair";
  else quality = "poor";

  // Add positive recommendations for excellent quality
  if (quality === "excellent" && recommendations.length === 0) {
    recommendations.push("Audio quality is excellent for transcription");
  }

  return {
    quality,
    score: Math.max(0, Math.min(100, score)),
    recommendations,
    issues,
  };
}

/**
 * Calculate audio levels suitable for UI display (0-100 scale)
 */
export function getDisplayLevel(metrics: AudioLevelMetrics): number {
  // Convert dB to 0-100 scale, with -60dB = 0 and -6dB = 100
  const normalizedDb = Math.max(-60, Math.min(-6, metrics.rmsDb));
  return ((normalizedDb + 60) / 54) * 100;
}

/**
 * Get color coding for audio level display
 */
export function getLevelColor(metrics: AudioLevelMetrics): string {
  if (metrics.clipping > 0.001) return "#f44336"; // Red - clipping
  if (metrics.rmsDb > -6) return "#ff9800"; // Orange - too hot
  if (metrics.rmsDb > -12) return "#4caf50"; // Green - optimal
  if (metrics.rmsDb > -24) return "#ffeb3b"; // Yellow - good
  if (metrics.rmsDb > -40) return "#ff9800"; // Orange - low
  return "#f44336"; // Red - too low
}

/**
 * Format dB value for display
 */
export function formatDbValue(db: number): string {
  if (db <= -60) return "-∞ dB";
  return `${db.toFixed(1)} dB`;
}

/**
 * Get meeting-specific audio recommendations
 */
export function getMeetingRecommendations(
  metrics: AudioLevelMetrics,
  meetingType:
    | "conference"
    | "virtual"
    | "interview"
    | "presentation" = "conference",
): string[] {
  const recommendations: string[] = [];

  switch (meetingType) {
    case "conference":
      if (metrics.backgroundNoise > 0.2) {
        recommendations.push("Use directional microphone to reduce room noise");
      }
      if (metrics.speechClarity < 0.3) {
        recommendations.push("Position microphone closer to primary speaker");
      }
      break;

    case "virtual":
      if (metrics.rmsDb < -20) {
        recommendations.push(
          "Increase microphone gain for virtual meeting clarity",
        );
      }
      if (metrics.backgroundNoise > 0.15) {
        recommendations.push("Enable noise suppression in meeting software");
      }
      break;

    case "interview":
      if (metrics.dynamicRange < 6) {
        recommendations.push(
          "Ensure natural speech dynamics for interview quality",
        );
      }
      if (metrics.voiceActivity < 0.2) {
        recommendations.push("Verify both participants are audible");
      }
      break;

    case "presentation":
      if (metrics.spectralCentroid < 500) {
        recommendations.push(
          "Use presentation microphone for better voice projection",
        );
      }
      if (metrics.signalToNoise < 15) {
        recommendations.push("Minimize audience noise during presentation");
      }
      break;
  }

  return recommendations;
}
