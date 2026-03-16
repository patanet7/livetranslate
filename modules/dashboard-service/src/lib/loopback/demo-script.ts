/**
 * Client-side demo script that injects draft → final → translation
 * sequences into the loopback store. No backend needed.
 *
 * Drives all three display modes (Split, Subtitle, Transcript) with
 * realistic bilingual EN/ZH content.
 */

import type { SegmentMessage, TranslationMessage, ServerMessage } from '$lib/types/ws-messages';

type LoopbackStore = {
	addSegment: (msg: SegmentMessage) => void;
	addTranslation: (msg: TranslationMessage) => void;
	clear: () => void;
	transcriptionStatus: 'up' | 'down';
	translationStatus: 'up' | 'down';
	connectionState: 'disconnected' | 'connecting' | 'connected' | 'error';
	isCapturing: boolean;
};

interface DemoStep {
	delayMs: number;
	action: (store: LoopbackStore, onMessage?: (msg: ServerMessage) => void) => void;
}

export interface DemoHandle {
	stop: () => void;
}

function segment(
	segmentId: number,
	text: string,
	language: string,
	opts: {
		stableText?: string;
		unstableText?: string;
		isDraft?: boolean;
		isFinal?: boolean;
	} = {}
): SegmentMessage {
	return {
		type: 'segment',
		segment_id: segmentId,
		text,
		language,
		confidence: opts.isDraft ? 0.7 : 0.95,
		stable_text: opts.stableText ?? text,
		unstable_text: opts.unstableText ?? '',
		is_final: opts.isFinal ?? false,
		is_draft: opts.isDraft ?? false,
		speaker_id: null,
		start_ms: null,
		end_ms: null,
	};
}

function translation(
	transcriptId: number,
	text: string,
	sourceLang: string,
	targetLang: string
): TranslationMessage {
	return {
		type: 'translation',
		text,
		source_lang: sourceLang,
		target_lang: targetLang,
		transcript_id: transcriptId,
		context_used: 0,
	};
}

/** Send a message through the real handleMessage path if available, else direct store call. */
function send(
	store: LoopbackStore,
	msg: ServerMessage,
	onMessage?: (msg: ServerMessage) => void
): void {
	if (onMessage) {
		onMessage(msg);
	} else if (msg.type === 'segment') {
		store.addSegment(msg as SegmentMessage);
	} else if (msg.type === 'translation') {
		store.addTranslation(msg as TranslationMessage);
	}
}

const DEMO_STEPS: DemoStep[] = [
	// 0.0s — Set status indicators to green
	{
		delayMs: 0,
		action: (store, onMessage) => {
			store.transcriptionStatus = 'up';
			store.translationStatus = 'up';
			store.connectionState = 'connected';
			if (onMessage) {
				onMessage({
					type: 'service_status',
					transcription: 'up',
					translation: 'up',
				} as ServerMessage);
			}
		},
	},

	// 0.5s — Draft EN segment 1
	{
		delayMs: 500,
		action: (store, onMessage) => {
			send(
				store,
				segment(1, 'Hello everyone welcome to the', 'en', {
					stableText: 'Hello everyone',
					unstableText: 'welcome to the',
					isDraft: true,
				}),
				onMessage
			);
		},
	},

	// 3.0s — Final EN segment 1
	{
		delayMs: 3000,
		action: (store, onMessage) => {
			send(
				store,
				segment(1, 'Hello everyone, welcome to the live demo', 'en', {
					stableText: 'Hello everyone, welcome to the live demo',
					isFinal: true,
				}),
				onMessage
			);
		},
	},

	// 4.0s — Translation for segment 1
	{
		delayMs: 4000,
		action: (store, onMessage) => {
			send(store, translation(1, '大家好，欢迎来到现场演示', 'en', 'zh'), onMessage);
		},
	},

	// 6.0s — Draft ZH segment 2
	{
		delayMs: 6000,
		action: (store, onMessage) => {
			send(
				store,
				segment(2, '今天我们要展示实时翻译', 'zh', {
					stableText: '今天我们要',
					unstableText: '展示实时翻译',
					isDraft: true,
				}),
				onMessage
			);
		},
	},

	// 9.0s — Final ZH segment 2
	{
		delayMs: 9000,
		action: (store, onMessage) => {
			send(
				store,
				segment(2, '今天我们要展示实时翻译系统的功能', 'zh', {
					stableText: '今天我们要展示实时翻译系统的功能',
					isFinal: true,
				}),
				onMessage
			);
		},
	},

	// 10.0s — Translation for segment 2
	{
		delayMs: 10000,
		action: (store, onMessage) => {
			send(
				store,
				translation(2, 'Today we will demonstrate the real-time translation system', 'zh', 'en'),
				onMessage
			);
		},
	},

	// 12.0s — Draft EN segment 3
	{
		delayMs: 12000,
		action: (store, onMessage) => {
			send(
				store,
				segment(3, 'The system processes audio in real time', 'en', {
					stableText: 'The system processes',
					unstableText: 'audio in real time',
					isDraft: true,
				}),
				onMessage
			);
		},
	},

	// 15.0s — Final EN segment 3
	{
		delayMs: 15000,
		action: (store, onMessage) => {
			send(
				store,
				segment(3, 'The system processes audio in real time using Whisper', 'en', {
					stableText: 'The system processes audio in real time using Whisper',
					isFinal: true,
				}),
				onMessage
			);
		},
	},

	// 16.0s — Translation for segment 3
	{
		delayMs: 16000,
		action: (store, onMessage) => {
			send(store, translation(3, '系统使用Whisper实时处理音频', 'en', 'zh'), onMessage);
		},
	},

	// 18.0s — Draft ZH segment 4
	{
		delayMs: 18000,
		action: (store, onMessage) => {
			send(
				store,
				segment(4, '翻译延迟非常低', 'zh', {
					stableText: '翻译延迟',
					unstableText: '非常低',
					isDraft: true,
				}),
				onMessage
			);
		},
	},

	// 21.0s — Final ZH segment 4
	{
		delayMs: 21000,
		action: (store, onMessage) => {
			send(
				store,
				segment(4, '翻译延迟非常低，可以用于实时会议', 'zh', {
					stableText: '翻译延迟非常低，可以用于实时会议',
					isFinal: true,
				}),
				onMessage
			);
		},
	},

	// 22.0s — Translation for segment 4
	{
		delayMs: 22000,
		action: (store, onMessage) => {
			send(
				store,
				translation(4, 'Translation latency is very low, suitable for real-time meetings', 'zh', 'en'),
				onMessage
			);
		},
	},
];

export function runDemo(
	store: LoopbackStore,
	opts: { onComplete: () => void; onMessage?: (msg: ServerMessage) => void }
): DemoHandle {
	const timerIds: ReturnType<typeof setTimeout>[] = [];
	let stopped = false;

	for (const step of DEMO_STEPS) {
		const id = setTimeout(() => {
			if (!stopped) step.action(store, opts.onMessage);
		}, step.delayMs);
		timerIds.push(id);
	}

	// Auto-complete after all steps + 2s buffer
	const completeId = setTimeout(() => {
		if (!stopped) {
			store.transcriptionStatus = 'down';
			store.translationStatus = 'down';
			store.connectionState = 'disconnected';
			opts.onComplete();
		}
	}, 24000);
	timerIds.push(completeId);

	return {
		stop() {
			stopped = true;
			for (const id of timerIds) clearTimeout(id);
			store.transcriptionStatus = 'down';
			store.translationStatus = 'down';
			store.connectionState = 'disconnected';
		},
	};
}
