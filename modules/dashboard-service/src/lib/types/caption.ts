export interface Caption {
	id: string;
	text: string;
	original_text: string;
	translated_text: string;
	speaker_name: string;
	speaker_color: string;
	target_language: string;
	confidence: number;
	duration_seconds: number;
	created_at: string;
	expires_at: string;
	receivedAt?: number;
}

export interface InterimCaption {
	chunk_id: string;
	text: string;
	speaker_name: string;
	is_final: boolean;
}

export type CaptionEvent =
	| { event: 'connected'; session_id: string; current_captions: Caption[]; timestamp: string }
	| { event: 'caption_added'; caption: Caption }
	| { event: 'caption_expired'; caption_id: string }
	| { event: 'caption_updated'; caption: Caption }
	| { event: 'interim_caption'; caption: InterimCaption }
	| { event: 'session_cleared' };

export type DisplayMode = 'both' | 'translated' | 'english';
