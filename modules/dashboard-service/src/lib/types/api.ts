export interface HealthStatus {
	status: 'healthy' | 'degraded' | 'down' | 'unknown';
	timestamp: number;
	services: ServiceStatus[];
}

export interface ServiceStatus {
	name: string;
	status: 'healthy' | 'unhealthy' | 'degraded';
	response_time_ms: number;
	uptime_seconds: number;
}

export interface TranslateRequest {
	text: string;
	target_language: string;
	source_language?: string | null;
	service?: string;
	quality?: 'fast' | 'balanced' | 'quality';
	session_id?: string | null;
	context?: string;
}

export interface TranslateResponse {
	translated_text: string;
	source_language: string;
	target_language: string;
	confidence: number;
	processing_time: number;
	model_used: string;
	backend_used: string;
	timestamp: string;
}

export interface TranslationTestResponse {
	success: boolean;
	original_text: string;
	translated_text: string;
	target_language: string;
	confidence: number;
	processing_time_ms: number;
}

export interface CaptionStats {
	session_id: string;
	captions_added: number;
	captions_expired: number;
	current_count: number;
	unique_speakers: number;
	connection_count: number;
	timestamp: string;
}
