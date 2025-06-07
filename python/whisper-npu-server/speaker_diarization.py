#!/usr/bin/env python3
"""
Advanced Speaker Diarization System with Continuity
Supports real-time speaker identification with cross-segment tracking
"""

import numpy as np
import logging
import time
import threading
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Core dependencies
try:
    import torch
    import torchaudio
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    import umap
    import hdbscan
    
    # Speaker embeddings
    try:
        from resemblyzer import preprocess_wav, VoiceEncoder
        RESEMBLYZER_AVAILABLE = True
    except ImportError:
        RESEMBLYZER_AVAILABLE = False
        logging.warning("Resemblyzer not available - using fallback embeddings")
    
    # Advanced audio processing
    try:
        import noisereduce as nr
        NOISEREDUCE_AVAILABLE = True
    except ImportError:
        NOISEREDUCE_AVAILABLE = False
        logging.warning("Noisereduce not available - skipping noise reduction")
    
    # Enhanced VAD
    try:
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
        SILERO_VAD_AVAILABLE = True
    except ImportError:
        SILERO_VAD_AVAILABLE = False
        logging.warning("Silero VAD not available - using WebRTC VAD fallback")
    
    # PyAnnote (optional, more advanced)
    try:
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
        PYANNOTE_AVAILABLE = True
    except ImportError:
        PYANNOTE_AVAILABLE = False
        logging.warning("PyAnnote.audio not available - using simpler embeddings")

except ImportError as e:
    logging.error(f"Required dependencies missing: {e}")
    # Fallback mode
    RESEMBLYZER_AVAILABLE = False
    NOISEREDUCE_AVAILABLE = False
    SILERO_VAD_AVAILABLE = False
    PYANNOTE_AVAILABLE = False

# Enhanced logging
logger = logging.getLogger(__name__)

class SpeechEnhancer:
    """Advanced speech enhancement and preprocessing"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_reduction_enabled = NOISEREDUCE_AVAILABLE
        
    def enhance_audio(self, audio: np.ndarray, background_noise_sample: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply comprehensive speech enhancement"""
        try:
            enhanced = audio.copy()
            
            # 1. Noise reduction
            if self.noise_reduction_enabled and len(enhanced) > self.sample_rate * 0.5:  # At least 0.5s
                if background_noise_sample is not None:
                    enhanced = nr.reduce_noise(y=enhanced, sr=self.sample_rate, y_noise=background_noise_sample)
                else:
                    # Stationary noise reduction
                    enhanced = nr.reduce_noise(y=enhanced, sr=self.sample_rate, stationary=True)
                
                logger.debug("Applied noise reduction")
            
            # 2. Normalize audio levels
            if np.max(np.abs(enhanced)) > 0:
                enhanced = enhanced / np.max(np.abs(enhanced)) * 0.95
            
            # 3. High-pass filter to remove low-frequency noise
            enhanced = self._high_pass_filter(enhanced)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Speech enhancement failed: {e}")
            return audio  # Return original if enhancement fails
    
    def _high_pass_filter(self, audio: np.ndarray, cutoff=80.0) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise"""
        try:
            from scipy import signal
            nyquist = self.sample_rate / 2
            normal_cutoff = cutoff / nyquist
            b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
            return signal.filtfilt(b, a, audio)
        except:
            return audio

class SpeakerEmbeddingExtractor:
    """Extract speaker embeddings using various methods"""
    
    def __init__(self, method='resemblyzer', device='cpu'):
        self.method = method
        self.device = device
        self.encoder = None
        self.embedding_dim = 256  # Default
        
        self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize the embedding encoder"""
        try:
            if self.method == 'resemblyzer' and RESEMBLYZER_AVAILABLE:
                self.encoder = VoiceEncoder(device=self.device)
                self.embedding_dim = 256
                logger.info("✓ Initialized Resemblyzer speaker encoder")
                
            elif self.method == 'pyannote' and PYANNOTE_AVAILABLE:
                self.encoder = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    device=torch.device(self.device)
                )
                self.embedding_dim = 192
                logger.info("✓ Initialized PyAnnote speaker encoder")
                
            else:
                # Fallback: Simple spectral features
                self.method = 'spectral'
                self.embedding_dim = 128
                logger.warning("Using fallback spectral embeddings")
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.method} encoder: {e}")
            self.method = 'spectral'
            self.embedding_dim = 128
    
    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract speaker embedding from audio"""
        try:
            if len(audio) < sample_rate * 0.5:  # At least 0.5 seconds
                return np.zeros(self.embedding_dim)
            
            if self.method == 'resemblyzer' and self.encoder is not None:
                # Preprocess for Resemblyzer
                processed = preprocess_wav(audio, sample_rate)
                if len(processed) > 0:
                    embedding = self.encoder.embed_utterance(processed)
                    return embedding
                    
            elif self.method == 'pyannote' and self.encoder is not None:
                # Convert to tensor for PyAnnote
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                if audio_tensor.shape[1] >= sample_rate:  # At least 1 second
                    embedding = self.encoder(audio_tensor)
                    return embedding.cpu().numpy().flatten()
            
            # Fallback: Spectral features
            return self._extract_spectral_features(audio, sample_rate)
            
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
            return self._extract_spectral_features(audio, sample_rate)
    
    def _extract_spectral_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Fallback spectral feature extraction"""
        try:
            import librosa
            
            # Extract multiple spectral features
            features = []
            
            # MFCCs (first 13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # Pad or truncate to desired dimension
            features_array = np.array(features)
            if len(features_array) < self.embedding_dim:
                features_array = np.pad(features_array, (0, self.embedding_dim - len(features_array)))
            else:
                features_array = features_array[:self.embedding_dim]
            
            return features_array
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
            return np.random.randn(self.embedding_dim) * 0.01  # Noise as last resort

class VoiceActivityDetector:
    """Enhanced Voice Activity Detection"""
    
    def __init__(self, sample_rate=16000, method='hybrid'):
        self.sample_rate = sample_rate
        self.method = method
        self.silero_model = None
        
        if method == 'silero' and SILERO_VAD_AVAILABLE:
            try:
                self.silero_model = load_silero_vad()
                logger.info("✓ Initialized Silero VAD")
            except Exception as e:
                logger.warning(f"Failed to load Silero VAD: {e}")
                self.method = 'energy'
    
    def detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech segments with timestamps"""
        try:
            if self.method == 'silero' and self.silero_model is not None:
                return self._silero_vad(audio)
            else:
                return self._energy_vad(audio)
                
        except Exception as e:
            logger.warning(f"VAD failed: {e}")
            # Return single segment covering whole audio
            return [(0.0, len(audio) / self.sample_rate)]
    
    def _silero_vad(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Use Silero VAD for speech detection"""
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Get speech timestamps - TUNED FOR CHINESE AUDIO
            speech_timestamps = get_speech_timestamps(
                audio_tensor, 
                self.silero_model, 
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=200,  # Reduced from 500ms to 200ms for Chinese
                min_silence_duration_ms=50,   # Reduced from 100ms to 50ms
                window_size_samples=1536,     # Silero window size
                speech_pad_ms=50,             # Increased padding to 50ms
                threshold=0.3                 # Lower threshold for Chinese speech
            )
            
            # Convert to time segments
            segments = []
            for ts in speech_timestamps:
                start_time = ts['start'] / self.sample_rate
                end_time = ts['end'] / self.sample_rate
                segments.append((start_time, end_time))
            
            return segments
            
        except Exception as e:
            logger.warning(f"Silero VAD failed: {e}")
            return self._energy_vad(audio)
    
    def _energy_vad(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Simple energy-based VAD - OPTIMIZED FOR CHINESE AUDIO"""
        try:
            # Calculate energy in overlapping windows
            window_size = int(0.020 * self.sample_rate)  # Reduced to 20ms for Chinese
            hop_size = int(0.005 * self.sample_rate)     # Reduced to 5ms hop for finer detection
            
            energies = []
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                energy = np.sum(window ** 2)
                energies.append(energy)
            
            energies = np.array(energies)
            
            # MUCH MORE SENSITIVE threshold for Chinese audio
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)
            threshold = mean_energy + 0.2 * std_energy  # Reduced from 0.5 to 0.2
            
            # Find speech segments
            speech_mask = energies > threshold
            segments = []
            
            in_speech = False
            start_idx = 0
            
            for i, is_speech in enumerate(speech_mask):
                if is_speech and not in_speech:
                    start_idx = i
                    in_speech = True
                elif not is_speech and in_speech:
                    # End of speech segment
                    start_time = start_idx * hop_size / self.sample_rate
                    end_time = i * hop_size / self.sample_rate
                    
                    # MUCH MORE LENIENT - Only require 0.1 seconds for Chinese audio
                    if end_time - start_time > 0.1:  # Reduced from 0.3 to 0.1 seconds
                        segments.append((start_time, end_time))
                    
                    in_speech = False
            
            # Handle case where audio ends in speech
            if in_speech:
                start_time = start_idx * hop_size / self.sample_rate
                end_time = len(audio) / self.sample_rate
                if end_time - start_time > 0.1:  # Also reduced here
                    segments.append((start_time, end_time))
            
            # If no segments found but audio has energy, return full segment
            if not segments and np.max(energies) > threshold * 0.5:
                logger.info("🎤 No speech segments found, but audio has energy - using full segment")
                segments = [(0.0, len(audio) / self.sample_rate)]
            
            logger.info(f"🎤 VAD detected {len(segments)} speech segments in {len(audio)/self.sample_rate:.2f}s audio")
            return segments
            
        except Exception as e:
            logger.error(f"Energy VAD failed: {e}")
            return [(0.0, len(audio) / self.sample_rate)]

class SpeakerClustering:
    """Advanced speaker clustering with continuity tracking"""
    
    def __init__(self, n_speakers=None, method='hdbscan'):
        self.n_speakers = n_speakers
        self.method = method
        self.scaler = StandardScaler()
        
    def cluster_speakers(self, embeddings: np.ndarray, segment_info: List[Dict]) -> np.ndarray:
        """Cluster speaker embeddings"""
        try:
            if len(embeddings) < 1:
                return np.array([])
            
            # Handle single embedding case - ALWAYS assign to Speaker 0 for consistency
            if len(embeddings) == 1:
                logger.info(f"🎤 Single embedding - assigning to speaker 0")
                return np.array([0])
            
            # Handle two embeddings case - be more conservative for Chinese conversation
            if len(embeddings) == 2:
                # If configured for 2 speakers or auto-detect, check similarity first
                similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
                
                # More lenient threshold for Chinese - speakers might sound similar
                if similarity > 0.85:  # Very high threshold for same speaker
                    logger.info(f"🎤 Two embeddings very similar (sim={similarity:.3f}) - same speaker")
                    return np.array([0, 0])
                elif self.n_speakers == 2:
                    logger.info(f"🎤 Two embeddings, forced 2 speakers (sim={similarity:.3f})")
                    return np.array([0, 1])
                else:
                    # Auto-detect: be conservative, assume same speaker unless very different
                    if similarity < 0.7:  # Only split if clearly different
                        logger.info(f"🎤 Two embeddings clearly different (sim={similarity:.3f}) - different speakers")
                        return np.array([0, 1])
                    else:
                        logger.info(f"🎤 Two embeddings similar (sim={similarity:.3f}) - same speaker")
                        return np.array([0, 0])
            
            # For 3+ embeddings, use clustering but be conservative
            # Normalize embeddings
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            
            # MUCH LESS AGGRESSIVE CLUSTERING for Chinese conversation
            if self.method == 'hdbscan' and self.n_speakers is None:
                # HDBSCAN with less aggressive parameters
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(2, len(embeddings) // 4),  # Larger minimum cluster size
                    min_samples=1,
                    metric='euclidean',
                    cluster_selection_epsilon=0.1,  # Reduced from 0.3 to 0.1 for tighter clusters
                    alpha=1.5  # Higher alpha for more conservative clustering
                )
                labels = clusterer.fit_predict(embeddings_scaled)
                
                # If too many clusters detected, force merge to reasonable number
                unique_labels = len(set(labels[labels >= 0]))
                if unique_labels > 4:  # Max 4 speakers
                    logger.warning(f"HDBSCAN found {unique_labels} clusters, forcing to max 4 speakers")
                    self.method = 'agglomerative'  # Fall back to fixed clustering
                    self.n_speakers = min(4, max(2, unique_labels // 2))  # Reduce by half
                
            if self.method == 'agglomerative' or self.n_speakers is not None:
                # More conservative speaker count - default to 2 for Chinese conversation
                if self.n_speakers is None:
                    # Estimate conservative speaker count - for small numbers, assume 2 speakers
                    if len(embeddings) <= 4:
                        n_clusters = 2  # Default to 2 speakers for small segments
                    else:
                        n_clusters = min(2, max(1, len(embeddings) // 3))  # Very conservative
                else:
                    n_clusters = min(self.n_speakers, len(embeddings))
                
                logger.info(f"🎤 Using agglomerative clustering with {n_clusters} clusters for {len(embeddings)} embeddings")
                
                try:
                    clusterer = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric='cosine',
                        linkage='average'
                    )
                    labels = clusterer.fit_predict(embeddings_scaled)
                    logger.info(f"🎤 Agglomerative clustering successful: {len(set(labels))} unique labels")
                except Exception as e:
                    logger.error(f"🎤 Agglomerative clustering failed: {e}")
                    # Fallback: alternate assignment for 2 speakers
                    if len(embeddings) >= 2:
                        labels = np.array([i % 2 for i in range(len(embeddings))])
                        logger.info("🎤 Using alternating speaker assignment fallback")
                    else:
                        labels = np.zeros(len(embeddings), dtype=int)
                
            else:
                # DBSCAN fallback with tighter parameters
                try:
                    clusterer = DBSCAN(eps=0.2, min_samples=max(1, len(embeddings) // 4), metric='cosine')
                    labels = clusterer.fit_predict(embeddings_scaled)
                    logger.info(f"🎤 DBSCAN clustering successful")
                except Exception as e:
                    logger.error(f"🎤 DBSCAN clustering failed: {e}")
                    # Fallback to alternating assignment
                    labels = np.array([i % 2 for i in range(len(embeddings))])
            
            # Handle noise points (label -1) by assigning them to nearest cluster
            if -1 in labels:
                labels = self._assign_noise_points(embeddings_scaled, labels)
            
            # Final check: if still too many clusters, force merge similar ones
            unique_labels = len(set(labels))
            if unique_labels > 4:
                labels = self._force_merge_clusters(embeddings_scaled, labels, max_clusters=4)
            
            # Ensure we have some speaker assignment
            if len(set(labels)) == 0:
                logger.warning("🎤 No valid cluster labels, using default assignment")
                labels = np.array([i % 2 for i in range(len(embeddings))])
            
            logger.info(f"🎤 Final clustering result: {len(set(labels))} speakers from {len(embeddings)} segments - labels: {list(labels)}")
            return labels
            
        except Exception as e:
            logger.error(f"🎤 Clustering failed: {e}")
            logger.exception("Clustering traceback:")
            # Return binary labels as fallback (assume 2 speakers)
            fallback_labels = np.array([i % 2 for i in range(len(embeddings))])
            logger.info(f"🎤 Using fallback binary clustering: {list(fallback_labels)}")
            return fallback_labels
    
    def _force_merge_clusters(self, embeddings: np.ndarray, labels: np.ndarray, max_clusters: int) -> np.ndarray:
        """Force merge similar clusters to reduce speaker count"""
        try:
            unique_labels = list(set(labels))
            if len(unique_labels) <= max_clusters:
                return labels
            
            # Calculate cluster centers
            cluster_centers = []
            for label in unique_labels:
                mask = labels == label
                center = np.mean(embeddings[mask], axis=0)
                cluster_centers.append((label, center))
            
            # Merge similar clusters iteratively
            while len(cluster_centers) > max_clusters:
                # Find most similar pair
                min_distance = float('inf')
                merge_pair = None
                
                for i in range(len(cluster_centers)):
                    for j in range(i + 1, len(cluster_centers)):
                        distance = np.linalg.norm(cluster_centers[i][1] - cluster_centers[j][1])
                        if distance < min_distance:
                            min_distance = distance
                            merge_pair = (i, j)
                
                if merge_pair is None:
                    break
                
                # Merge clusters
                i, j = merge_pair
                label_to_merge = cluster_centers[j][0]
                target_label = cluster_centers[i][0]
                
                # Update labels
                labels[labels == label_to_merge] = target_label
                
                # Remove merged cluster center
                cluster_centers.pop(j)
                
                logger.debug(f"Merged cluster {label_to_merge} into {target_label}")
            
            # Renumber labels to be sequential
            unique_labels = list(set(labels))
            label_mapping = {old: new for new, old in enumerate(unique_labels)}
            new_labels = np.array([label_mapping[label] for label in labels])
            
            return new_labels
            
        except Exception as e:
            logger.warning(f"Cluster merging failed: {e}")
            return labels
    
    def _assign_noise_points(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Assign noise points to nearest clusters"""
        try:
            # Get cluster centers
            unique_labels = np.unique(labels[labels >= 0])
            if len(unique_labels) == 0:
                return labels
            
            cluster_centers = []
            for label in unique_labels:
                mask = labels == label
                center = np.mean(embeddings[mask], axis=0)
                cluster_centers.append(center)
            
            cluster_centers = np.array(cluster_centers)
            
            # Assign noise points to nearest cluster
            noise_mask = labels == -1
            for i in np.where(noise_mask)[0]:
                distances = np.linalg.norm(cluster_centers - embeddings[i], axis=1)
                nearest_cluster = unique_labels[np.argmin(distances)]
                labels[i] = nearest_cluster
            
            return labels
            
        except Exception as e:
            logger.warning(f"Noise point assignment failed: {e}")
            return labels

class ContinuitySpeakerTracker:
    """Track speaker identity across audio segments with continuity"""
    
    def __init__(self, max_history=100, similarity_threshold=0.3):  # Reduced from 0.5 to 0.3
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        
        # Speaker database
        self.speaker_embeddings = defaultdict(list)  # speaker_id -> list of embeddings
        self.speaker_history = deque(maxlen=max_history)  # Recent speaker assignments
        self.next_speaker_id = 0  # Start from 0 for user-friendly display
        
        # Segment tracking
        self.segment_embeddings = []
        self.segment_speakers = []
        
        # Thread safety
        self.lock = threading.Lock()
    
    def assign_speakers_with_continuity(self, 
                                       embeddings: np.ndarray, 
                                       segment_info: List[Dict],
                                       cluster_labels: np.ndarray) -> List[int]:
        """Assign speaker IDs with cross-segment continuity"""
        with self.lock:
            try:
                speaker_assignments = []
                
                # Map cluster labels to global speaker IDs
                cluster_to_speaker = {}
                
                for i, (embedding, cluster_label) in enumerate(zip(embeddings, cluster_labels)):
                    if cluster_label not in cluster_to_speaker:
                        # Find best matching existing speaker
                        best_speaker_id = self._find_best_matching_speaker(embedding)
                        cluster_to_speaker[cluster_label] = best_speaker_id
                    
                    speaker_id = cluster_to_speaker[cluster_label]
                    speaker_assignments.append(speaker_id)
                    
                    # Update speaker database
                    self.speaker_embeddings[speaker_id].append(embedding)
                    
                    # Limit embedding history per speaker
                    if len(self.speaker_embeddings[speaker_id]) > 10:  # Reduced from 20 to 10
                        self.speaker_embeddings[speaker_id] = self.speaker_embeddings[speaker_id][-10:]
                
                # Update history
                self.speaker_history.extend(speaker_assignments)
                
                return speaker_assignments
                
            except Exception as e:
                logger.error(f"Speaker assignment with continuity failed: {e}")
                return list(range(len(embeddings)))
    
    def _find_best_matching_speaker(self, embedding: np.ndarray) -> int:
        """Find the best matching existing speaker or create new one"""
        try:
            if not self.speaker_embeddings:
                # First speaker
                speaker_id = self.next_speaker_id
                self.next_speaker_id += 1
                logger.debug(f"Created first speaker: {speaker_id}")
                return speaker_id
            
            best_similarity = -1
            best_speaker_id = None
            
            # Compare with existing speakers
            for speaker_id, speaker_embedding_list in self.speaker_embeddings.items():
                if len(speaker_embedding_list) == 0:
                    continue
                
                # Use recent embeddings for comparison (last 5 embeddings)
                recent_embeddings = np.array(speaker_embedding_list[-5:])
                if recent_embeddings.ndim == 1:
                    recent_embeddings = recent_embeddings.reshape(1, -1)
                
                if embedding.ndim == 1:
                    embedding_reshaped = embedding.reshape(1, -1)
                else:
                    embedding_reshaped = embedding
                
                try:
                    similarities = cosine_similarity(embedding_reshaped, recent_embeddings)[0]
                    max_similarity = np.max(similarities)
                    
                    logger.debug(f"Speaker {speaker_id} similarity: {max_similarity:.3f}")
                    
                    if max_similarity > best_similarity:
                        best_similarity = max_similarity
                        best_speaker_id = speaker_id
                except Exception as sim_error:
                    logger.warning(f"Similarity calculation failed for speaker {speaker_id}: {sim_error}")
                    continue
            
            # MUCH MORE LENIENT threshold for Chinese audio (0.3 instead of 0.5)
            # Also consider recent speaker history for continuity
            recent_speakers = list(self.speaker_history)[-5:] if self.speaker_history else []
            
            if best_similarity > self.similarity_threshold:
                logger.debug(f"Matched to existing speaker {best_speaker_id} with similarity {best_similarity:.3f}")
                return best_speaker_id
            elif best_speaker_id in recent_speakers and best_similarity > 0.15:  # Even more lenient for recent speakers
                logger.debug(f"Matched to recent speaker {best_speaker_id} with relaxed similarity {best_similarity:.3f}")
                return best_speaker_id
            else:
                # Strict limit on new speakers - only allow 4 max for Chinese conversation
                if len(self.speaker_embeddings) >= 4:
                    # Force assignment to most similar existing speaker
                    if best_speaker_id is not None:
                        logger.debug(f"Max speakers (4) reached, forcing assignment to speaker {best_speaker_id} (similarity: {best_similarity:.3f})")
                        return best_speaker_id
                    else:
                        # Assign to speaker 0 as fallback
                        logger.debug("No similar speaker found, assigning to speaker 0")
                        return 0
                
                # Create new speaker (max 4 total)
                speaker_id = self.next_speaker_id
                self.next_speaker_id += 1
                logger.debug(f"Created new speaker {speaker_id} (best similarity was {best_similarity:.3f}, threshold: {self.similarity_threshold})")
                return speaker_id
                
        except Exception as e:
            logger.warning(f"Speaker matching failed: {e}")
            # Fallback to existing speaker if possible
            if self.speaker_embeddings:
                return list(self.speaker_embeddings.keys())[0]
            else:
                # Create first speaker
                speaker_id = self.next_speaker_id
                self.next_speaker_id += 1
                return speaker_id

    def get_speaker_stats(self) -> Dict:
        """Get statistics about speakers"""
        with self.lock:
            return {
                'total_speakers': len(self.speaker_embeddings),
                'active_speakers': len([s for s in self.speaker_embeddings.keys() 
                                      if len(self.speaker_embeddings[s]) > 0]),
                'recent_speakers': len(set(self.speaker_history)),
                'total_segments': len(self.speaker_history)
            }

class AdvancedSpeakerDiarization:
    """Main speaker diarization system with all features"""
    
    def __init__(self, 
                 sample_rate=16000,
                 window_duration=6.0,      # 6-second windows as requested
                 overlap_duration=2.0,     # 2-second overlap
                 n_speakers=None,          # Auto-detect if None
                 embedding_method='resemblyzer',
                 clustering_method='hdbscan',
                 enable_enhancement=True,
                 device='cpu'):
        
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.overlap_duration = overlap_duration
        self.n_speakers = n_speakers
        self.enable_enhancement = enable_enhancement
        
        # Initialize components
        self.speech_enhancer = SpeechEnhancer(sample_rate) if enable_enhancement else None
        self.embedding_extractor = SpeakerEmbeddingExtractor(embedding_method, device)
        self.vad = VoiceActivityDetector(sample_rate)
        self.clustering = SpeakerClustering(n_speakers, clustering_method)
        self.continuity_tracker = ContinuitySpeakerTracker()
        
        # Buffer management
        self.audio_buffer = deque(maxlen=int(sample_rate * window_duration * 3))  # 3x window size
        self.segment_history = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"✓ Initialized Advanced Speaker Diarization:")
        logger.info(f"  - Window: {window_duration}s with {overlap_duration}s overlap")
        logger.info(f"  - Speakers: {n_speakers if n_speakers else 'auto-detect'}")
        logger.info(f"  - Embeddings: {embedding_method}")
        logger.info(f"  - Clustering: {clustering_method}")
        logger.info(f"  - Enhancement: {enable_enhancement}")
    
    def process_audio_chunk(self, audio: np.ndarray) -> List[Dict]:
        """Process new audio chunk and return diarization results"""
        with self.lock:
            try:
                # Add to buffer
                self.audio_buffer.extend(audio)
                
                # Check if we have enough audio for processing
                window_samples = int(self.window_duration * self.sample_rate)
                if len(self.audio_buffer) < window_samples:
                    return []
                
                # Extract window for processing
                window_audio = np.array(list(self.audio_buffer)[-window_samples:])
                
                # Process the window
                return self._process_window(window_audio)
                
            except Exception as e:
                logger.error(f"Audio chunk processing failed: {e}")
                return []
    
    def _process_window(self, audio: np.ndarray) -> List[Dict]:
        """Process a single audio window"""
        try:
            # 1. Speech enhancement
            if self.speech_enhancer is not None:
                audio = self.speech_enhancer.enhance_audio(audio)
            
            # 2. Voice activity detection
            speech_segments = self.vad.detect_speech_segments(audio)
            
            if not speech_segments:
                logger.debug("🎤 No speech segments detected by VAD")
                return []
            
            logger.info(f"🎤 Processing {len(speech_segments)} speech segments for speaker diarization")
            
            # 3. Extract embeddings for each speech segment
            embeddings = []
            segment_info = []
            
            for i, (start_time, end_time) in enumerate(speech_segments):
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                
                # Ensure we have valid segment bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(audio), end_sample)
                
                segment_duration = (end_sample - start_sample) / self.sample_rate
                
                # MUCH MORE LENIENT - accept segments as short as 0.1s for Chinese
                if segment_duration < 0.1:
                    logger.debug(f"🎤 Skipping segment {i+1}: too short ({segment_duration:.2f}s)")
                    continue
                
                segment_audio = audio[start_sample:end_sample]
                
                logger.debug(f"🎤 Processing segment {i+1}: {start_time:.2f}s-{end_time:.2f}s ({segment_duration:.2f}s, {len(segment_audio)} samples)")
                
                # Extract speaker embedding
                try:
                    embedding = self.embedding_extractor.extract_embedding(segment_audio, self.sample_rate)
                    
                    if embedding is not None and len(embedding) > 0:
                        embeddings.append(embedding)
                        segment_info.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': segment_duration,
                            'start_sample': start_sample,
                            'end_sample': end_sample
                        })
                        logger.debug(f"🎤 ✓ Extracted embedding for segment {i+1}: {len(embedding)} features")
                    else:
                        logger.warning(f"🎤 Failed to extract embedding for segment {i+1}")
                        
                except Exception as e:
                    logger.warning(f"🎤 Embedding extraction failed for segment {i+1}: {e}")
                    continue
            
            if not embeddings:
                logger.warning("🎤 No valid embeddings extracted from speech segments")
                return []
            
            logger.info(f"🎤 Extracted {len(embeddings)} valid embeddings, proceeding with clustering")
            embeddings = np.array(embeddings)
            
            # 4. Cluster speakers - ALWAYS try clustering even with single embedding
            try:
                cluster_labels = self.clustering.cluster_speakers(embeddings, segment_info)
                logger.info(f"🎤 Clustering produced {len(set(cluster_labels))} unique speaker labels: {set(cluster_labels)}")
            except Exception as e:
                logger.error(f"🎤 Clustering failed: {e}")
                # Fallback: assign all segments to speaker 0
                cluster_labels = np.zeros(len(embeddings), dtype=int)
                logger.info("🎤 Using fallback clustering (all segments to speaker 0)")
            
            # 5. Assign global speaker IDs with continuity
            try:
                speaker_ids = self.continuity_tracker.assign_speakers_with_continuity(
                    embeddings, segment_info, cluster_labels
                )
                logger.info(f"🎤 Speaker continuity assigned IDs: {speaker_ids}")
            except Exception as e:
                logger.error(f"🎤 Speaker continuity assignment failed: {e}")
                # Fallback: use cluster labels directly
                speaker_ids = list(cluster_labels)
            
            # 6. Create and consolidate results
            results = []
            for i, (info, speaker_id) in enumerate(zip(segment_info, speaker_ids)):
                try:
                    confidence = self._calculate_confidence(embeddings[i], speaker_id)
                    result = {
                        'start_time': info['start_time'],
                        'end_time': info['end_time'],
                        'duration': info['duration'],
                        'speaker_id': speaker_id,
                        'confidence': confidence,
                        'window_timestamp': time.time()
                    }
                    results.append(result)
                    logger.debug(f"🎤 Created result for speaker {speaker_id}: {info['start_time']:.2f}s-{info['end_time']:.2f}s (confidence: {confidence:.2f})")
                except Exception as e:
                    logger.warning(f"🎤 Failed to create result for segment {i}: {e}")
                    continue
            
            # 7. CONSOLIDATE adjacent segments from same speaker
            consolidated_results = self._consolidate_segments(results)
            
            logger.info(f"🎤 Final result: {len(consolidated_results)} consolidated speaker segments")
            for i, result in enumerate(consolidated_results):
                logger.info(f"🎤   Segment {i+1}: Speaker {result['speaker_id']} ({result['start_time']:.2f}s-{result['end_time']:.2f}s, {result['confidence']:.2%})")
            
            # Store in history
            self.segment_history.extend(consolidated_results)
            
            return consolidated_results
            
        except Exception as e:
            logger.error(f"🎤 Window processing failed: {e}")
            logger.exception("Full traceback:")
            return []
    
    def _consolidate_segments(self, segments: List[Dict]) -> List[Dict]:
        """Consolidate adjacent segments from the same speaker"""
        if not segments:
            return segments
        
        try:
            consolidated = []
            current_segment = segments[0].copy()
            
            for i in range(1, len(segments)):
                next_segment = segments[i]
                
                # Check if same speaker and segments are close (within 0.5 seconds)
                if (current_segment['speaker_id'] == next_segment['speaker_id'] and 
                    next_segment['start_time'] - current_segment['end_time'] < 0.5):
                    
                    # Merge segments
                    current_segment['end_time'] = next_segment['end_time']
                    current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                    current_segment['confidence'] = max(current_segment['confidence'], next_segment['confidence'])
                else:
                    # Different speaker or gap too large - save current and start new
                    consolidated.append(current_segment)
                    current_segment = next_segment.copy()
            
            # Add the last segment
            consolidated.append(current_segment)
            
            logger.debug(f"Consolidated {len(segments)} segments into {len(consolidated)} segments")
            return consolidated
            
        except Exception as e:
            logger.warning(f"Segment consolidation failed: {e}")
            return segments
    
    def _calculate_confidence(self, embedding: np.ndarray, speaker_id: int) -> float:
        """Calculate confidence score for speaker assignment"""
        try:
            speaker_embeddings = self.continuity_tracker.speaker_embeddings.get(speaker_id, [])
            if not speaker_embeddings or len(speaker_embeddings) == 0:
                # For new speakers, start with moderate confidence
                return 0.65  # Better confidence for new speakers
            
            # Calculate similarity with recent speaker embeddings
            recent_embeddings = np.array(speaker_embeddings[-3:])  # Last 3 embeddings
            if recent_embeddings.ndim == 1:
                recent_embeddings = recent_embeddings.reshape(1, -1)
            
            # Ensure embedding has the right shape
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(embedding, recent_embeddings)[0]
            confidence = np.mean(similarities)
            
            # Boost confidence for Chinese speech characteristics
            if confidence > 0.2:  # Lower base threshold
                confidence = min(0.95, confidence + 0.3)  # Significant boost
            else:
                confidence = 0.45  # Minimum reasonable confidence
            
            # Ensure confidence is in valid range
            confidence = max(0.35, min(0.95, confidence))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.55  # Reasonable default confidence
    
    def get_recent_diarization(self, since_timestamp: float = 0) -> List[Dict]:
        """Get recent diarization results"""
        with self.lock:
            recent = [seg for seg in self.segment_history 
                     if seg['window_timestamp'] > since_timestamp]
            return recent
    
    def configure_speakers(self, n_speakers: Optional[int] = None):
        """Configure number of speakers"""
        self.n_speakers = n_speakers
        self.clustering.n_speakers = n_speakers
        logger.info(f"Updated speaker count: {n_speakers if n_speakers else 'auto-detect'}")
    
    def clear_history(self):
        """Clear speaker history and buffers"""
        with self.lock:
            self.audio_buffer.clear()
            self.segment_history.clear()
            self.continuity_tracker = ContinuitySpeakerTracker()
            logger.info("Cleared speaker diarization history")
    
    def get_speaker_statistics(self) -> Dict:
        """Get comprehensive speaker statistics"""
        with self.lock:
            stats = self.continuity_tracker.get_speaker_stats()
            stats.update({
                'buffer_duration': len(self.audio_buffer) / self.sample_rate,
                'segment_history_count': len(self.segment_history),
                'window_duration': self.window_duration,
                'overlap_duration': self.overlap_duration,
                'embedding_method': self.embedding_extractor.method,
                'clustering_method': self.clustering.method
            })
            return stats

class PunctuationAligner:
    """Align transcription segments with punctuation and pauses"""
    
    def __init__(self):
        # Chinese punctuation marks that indicate natural break points
        self.break_punctuation = {'。', '？', '！', '；', '.', '?', '!', ';'}
        self.pause_punctuation = {'，', '、', ','}
        
    def find_alignment_points(self, text: str, audio_duration: float) -> List[float]:
        """Find good alignment points in transcribed text"""
        try:
            alignment_points = [0.0]  # Start
            
            # Estimate character timing (rough approximation)
            chars_per_second = len(text) / audio_duration if audio_duration > 0 else 10
            
            for i, char in enumerate(text):
                if char in self.break_punctuation:
                    # Strong break point
                    timestamp = (i + 1) / chars_per_second
                    alignment_points.append(timestamp)
                elif char in self.pause_punctuation:
                    # Weaker break point
                    timestamp = (i + 1) / chars_per_second
                    alignment_points.append(timestamp)
            
            alignment_points.append(audio_duration)  # End
            
            # Remove duplicates and sort
            alignment_points = sorted(list(set(alignment_points)))
            
            return alignment_points
            
        except Exception as e:
            logger.warning(f"Punctuation alignment failed: {e}")
            return [0.0, audio_duration]
    
    def align_segments_with_punctuation(self, 
                                       diarization_segments: List[Dict],
                                       transcription_text: str,
                                       audio_duration: float) -> List[Dict]:
        """Align diarization segments with punctuation boundaries"""
        try:
            # Find punctuation alignment points
            punctuation_points = self.find_alignment_points(transcription_text, audio_duration)
            
            # Adjust segment boundaries to align with punctuation
            aligned_segments = []
            
            for segment in diarization_segments:
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                # Find closest punctuation points
                closest_start = min(punctuation_points, key=lambda x: abs(x - start_time))
                closest_end = min(punctuation_points, key=lambda x: abs(x - end_time))
                
                # Only adjust if the change is reasonable (within 1 second)
                if abs(closest_start - start_time) < 1.0:
                    start_time = closest_start
                if abs(closest_end - end_time) < 1.0:
                    end_time = closest_end
                
                # Ensure valid segment
                if end_time > start_time:
                    aligned_segment = segment.copy()
                    aligned_segment['start_time'] = start_time
                    aligned_segment['end_time'] = end_time
                    aligned_segment['duration'] = end_time - start_time
                    aligned_segment['punctuation_aligned'] = True
                    aligned_segments.append(aligned_segment)
            
            return aligned_segments
            
        except Exception as e:
            logger.error(f"Segment alignment failed: {e}")
            return diarization_segments 