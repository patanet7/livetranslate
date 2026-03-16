/**
 * Polls /api/health to keep STT/MT status indicators in sync with
 * actual backend service health. Config-driven, not hardcoded.
 */

export interface ServiceHealth {
	transcription: 'up' | 'down';
	translation: 'up' | 'down';
}

export interface HealthPollerHandle {
	stop: () => void;
}

type OnUpdate = (health: ServiceHealth) => void;

const POLL_INTERVAL_MS = 5_000;
const FETCH_TIMEOUT_MS = 3_000;

function mapStatus(raw: unknown): 'up' | 'down' {
	if (!raw) return 'down';
	// Handle both flat strings ("healthy") and objects ({ status: "healthy" })
	const statusStr = typeof raw === 'string' ? raw : (raw as Record<string, unknown>)?.status;
	if (typeof statusStr !== 'string') return 'down';
	const lower = statusStr.toLowerCase();
	return lower === 'healthy' || lower === 'up' || lower === 'active' || lower === 'ready'
		? 'up'
		: 'down';
}

export function startHealthPoller(onUpdate: OnUpdate): HealthPollerHandle {
	let stopped = false;

	async function poll() {
		if (stopped) return;
		try {
			const res = await fetch('/api/health', {
				signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
			});
			if (res.ok) {
				const data = await res.json();
				const services = data.services ?? {};
				onUpdate({
					// The backend uses "whisper" for the transcription service
					transcription: mapStatus(services.whisper ?? services.transcription),
					translation: mapStatus(services.translation),
				});
			} else {
				onUpdate({ transcription: 'down', translation: 'down' });
			}
		} catch {
			onUpdate({ transcription: 'down', translation: 'down' });
		}
	}

	// Poll immediately, then on interval
	poll();
	const intervalId = setInterval(poll, POLL_INTERVAL_MS);

	return {
		stop() {
			stopped = true;
			clearInterval(intervalId);
		},
	};
}
