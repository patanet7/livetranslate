export interface FirefliesSession {
	session_id: string;
	transcript_id: string;
	connection_status: 'CONNECTING' | 'CONNECTED' | 'ERROR' | 'DISCONNECTED';
	chunks_received: number;
	sentences_produced: number;
	translations_completed: number;
	speakers_detected: string[];
	connected_at: string;
	error_count: number;
	last_error: string | null;
	persistence_failures: number;
	persistence_healthy: boolean;
}

export interface ConnectRequest {
	api_key?: string | null;
	transcript_id: string;
	target_languages?: string[] | null;
	glossary_id?: string | null;
	domain?: string | null;
	translation_model?: string | null;
	pause_threshold_ms?: number | null;
	max_buffer_words?: number | null;
	context_window_size?: number | null;
	api_base_url?: string | null;
}

export interface ConnectResponse {
	success: boolean;
	message: string;
	session_id: string;
	connection_status: string;
	transcript_id: string;
}

export interface DisconnectRequest {
	session_id: string;
}
