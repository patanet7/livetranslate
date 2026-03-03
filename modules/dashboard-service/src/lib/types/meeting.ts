// modules/dashboard-service/src/lib/types/meeting.ts

export interface Meeting {
	id: string;
	fireflies_transcript_id: string | null;
	title: string | null;
	meeting_link: string | null;
	organizer_email: string | null;
	participants: string[];
	start_time: string | null;
	end_time: string | null;
	duration: number | null;
	source: 'fireflies' | 'upload';
	status: 'live' | 'completed' | 'error' | 'archived';
	sync_status: 'none' | 'live' | 'syncing' | 'synced' | 'failed';
	sync_error: string | null;
	synced_at: string | null;
	audio_url: string | null;
	video_url: string | null;
	transcript_url: string | null;
	created_at: string;
	updated_at: string;
	// Computed counts from backend JOIN
	chunk_count: number;
	sentence_count: number;
	translation_count?: number;
	insight_count?: number;
}

export interface MeetingListResponse {
	meetings: Meeting[];
	limit: number;
	offset: number;
}

export interface MeetingSearchResponse {
	results: Meeting[];
	query: string;
	count: number;
}

export interface MeetingSentence {
	id: string;
	meeting_id: string;
	text: string;
	speaker_name: string | null;
	start_time: number;
	end_time: number;
	boundary_type: string | null;
	chunk_ids: string[];
	created_at: string;
	translations: MeetingTranslation[];
}

export interface MeetingTranslation {
	translated_text: string;
	target_language: string;
	confidence: number;
	model_used: string | null;
}

export interface MeetingTranscriptResponse {
	meeting_id: string;
	sentences: MeetingSentence[];
	count: number;
	source?: 'chunks';
}

export interface MeetingInsight {
	id: string;
	meeting_id: string;
	insight_type: string;
	content: Record<string, unknown>;
	source: string;
	model_used: string | null;
	generated_at: string | null;
	created_at: string;
}

export interface MeetingInsightsResponse {
	meeting_id: string;
	insights: MeetingInsight[];
	count: number;
}

export interface MeetingSpeaker {
	id: string;
	meeting_id: string;
	speaker_name: string;
	email: string | null;
	talk_time_seconds: number;
	word_count: number;
	sentiment_score: number | null;
	analytics: Record<string, unknown> | null;
	created_at: string;
}

export interface MeetingSpeakersResponse {
	meeting_id: string;
	speakers: MeetingSpeaker[];
	count: number;
}

export interface InsightGenerateResponse {
	meeting_id: string;
	generated: Array<{ type: string; content: Record<string, unknown> }>;
	count: number;
}
