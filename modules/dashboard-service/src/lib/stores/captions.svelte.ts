import { browser } from '$app/environment';
import type { Caption } from '$lib/types';

export class CaptionStore {
	captions = $state<(Caption & { receivedAt: number })[]>([]);
	interim = $state('');
	maxCaptions = $state(50);
	/** Time window in ms for aggregating consecutive captions from the same speaker. 0 = disabled. */
	aggregateWindowMs = $state(0);
	/** Fallback expiry in ms when server doesn't provide expires_at. */
	fallbackExpiryMs: number;

	#cleanupInterval: ReturnType<typeof setInterval> | null = null;

	constructor(fallbackExpiryMs = 8_000) {
		this.fallbackExpiryMs = fallbackExpiryMs;
	}

	start() {
		if (!browser) return;
		this.#cleanupInterval = setInterval(() => this.#expireOld(), 1000);
	}

	stop() {
		if (this.#cleanupInterval) {
			clearInterval(this.#cleanupInterval);
			this.#cleanupInterval = null;
		}
	}

	addCaption(caption: Caption) {
		const now = Date.now();

		// Aggregation: if the last caption is from the same speaker within the time window, append text
		if (this.aggregateWindowMs > 0 && this.captions.length > 0) {
			const last = this.captions[this.captions.length - 1];
			if (
				last.speaker_name === caption.speaker_name &&
				now - last.receivedAt < this.aggregateWindowMs
			) {
				this.captions = this.captions.map((c, i) =>
					i === this.captions.length - 1
						? {
								...c,
								translated_text:
									(c.translated_text || c.text) + ' ' + (caption.translated_text || caption.text),
								original_text: c.original_text + ' ' + caption.original_text,
								expires_at: caption.expires_at || c.expires_at,
								receivedAt: now
							}
						: c
				);
				return;
			}
		}

		const enriched = { ...caption, receivedAt: now };
		this.captions = [...this.captions, enriched].slice(-this.maxCaptions);
	}

	updateCaption(caption: Caption) {
		this.captions = this.captions.map((c) =>
			c.id === caption.id ? { ...caption, receivedAt: c.receivedAt } : c
		);
	}

	removeCaption(captionId: string) {
		this.captions = this.captions.filter((c) => c.id !== captionId);
	}

	updateInterim(text: string) {
		this.interim = text;
	}

	clear() {
		this.captions = [];
		this.interim = '';
	}

	#expireOld() {
		const now = Date.now();
		this.captions = this.captions.filter((c) => {
			// Use server-provided expires_at if available, otherwise fall back
			if (c.expires_at) {
				return new Date(c.expires_at).getTime() > now;
			}
			return now - c.receivedAt < this.fallbackExpiryMs;
		});
	}
}

export const captionStore = new CaptionStore();
