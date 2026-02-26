import { browser } from '$app/environment';
import type { Caption } from '$lib/types';

export class CaptionStore {
	captions = $state<(Caption & { receivedAt: number })[]>([]);
	interim = $state('');
	maxCaptions = $state(50);

	#expiryMs: number;
	#cleanupInterval: ReturnType<typeof setInterval> | null = null;

	constructor(expiryMs = 10_000) {
		this.#expiryMs = expiryMs;
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
		const enriched = { ...caption, receivedAt: Date.now() };
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
		const cutoff = Date.now() - this.#expiryMs;
		this.captions = this.captions.filter((c) => c.receivedAt > cutoff);
	}
}

export const captionStore = new CaptionStore();
