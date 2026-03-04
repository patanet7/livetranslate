/**
 * Diarization API client for dashboard.
 */

const BASE = '/api/diarization';

export interface DiarizationJob {
	job_id: string;
	meeting_id: number;
	status:
		| 'queued'
		| 'downloading'
		| 'processing'
		| 'mapping'
		| 'completed'
		| 'failed'
		| 'cancelled';
	triggered_by: string;
	detected_language?: string;
	num_speakers_detected?: number;
	processing_time_seconds?: number;
	speaker_map?: Record<string, SpeakerMapEntry>;
	unmapped_speakers?: number[];
	merge_applied?: boolean;
	error_message?: string;
	created_at?: string;
	completed_at?: string;
}

export interface SpeakerMapEntry {
	name: string;
	confidence: number;
	method: string;
}

export interface SpeakerProfile {
	id: number;
	name: string;
	email?: string;
	enrollment_source: string;
	sample_count: number;
}

export interface DiarizationRules {
	enabled: boolean;
	participant_patterns: string[];
	title_patterns: string[];
	min_duration_minutes: number;
	exclude_empty: boolean;
}

export interface TranscriptComparison {
	meeting_id: number;
	fireflies_sentences: Record<string, unknown>[];
	vibevoice_segments: Record<string, unknown>[];
	speaker_map?: Record<string, SpeakerMapEntry>;
}

export function diarizationApi(fetchFn: typeof fetch = fetch) {
	return {
		// Jobs
		async createJob(meetingId: number, hotwords?: string[]): Promise<DiarizationJob> {
			const res = await fetchFn(`${BASE}/jobs`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ meeting_id: meetingId, hotwords })
			});
			return res.json();
		},

		async listJobs(status?: string): Promise<DiarizationJob[]> {
			const params = status ? `?status=${status}` : '';
			const res = await fetchFn(`${BASE}/jobs${params}`);
			return res.json();
		},

		async getJob(jobId: string): Promise<DiarizationJob> {
			const res = await fetchFn(`${BASE}/jobs/${jobId}`);
			return res.json();
		},

		async cancelJob(jobId: string): Promise<{ status: string }> {
			const res = await fetchFn(`${BASE}/jobs/${jobId}/cancel`, { method: 'POST' });
			return res.json();
		},

		// Speakers
		async listSpeakers(): Promise<SpeakerProfile[]> {
			const res = await fetchFn(`${BASE}/speakers`);
			return res.json();
		},

		async createSpeaker(name: string, email?: string): Promise<SpeakerProfile> {
			const res = await fetchFn(`${BASE}/speakers`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ name, email })
			});
			return res.json();
		},

		async mergeSpeakers(sourceId: number, targetId: number): Promise<{ status: string }> {
			const res = await fetchFn(`${BASE}/speakers/merge`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ source_id: sourceId, target_id: targetId })
			});
			return res.json();
		},

		// Rules
		async getRules(): Promise<DiarizationRules> {
			const res = await fetchFn(`${BASE}/rules`);
			return res.json();
		},

		async updateRules(rules: DiarizationRules): Promise<DiarizationRules> {
			const res = await fetchFn(`${BASE}/rules`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(rules)
			});
			return res.json();
		},

		// Comparison
		async compareTranscripts(meetingId: number): Promise<TranscriptComparison> {
			const res = await fetchFn(`${BASE}/meetings/${meetingId}/compare`);
			return res.json();
		},

		async applyDiarization(meetingId: number): Promise<{ status: string }> {
			const res = await fetchFn(`${BASE}/meetings/${meetingId}/apply`, { method: 'POST' });
			return res.json();
		}
	};
}
