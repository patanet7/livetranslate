import { ORCHESTRATION_URL } from '$env/static/private';

export class ApiError extends Error {
	constructor(
		public status: number,
		message: string
	) {
		super(message);
		this.name = 'ApiError';
	}
}

async function apiRequest<T>(
	fetch: typeof globalThis.fetch,
	path: string,
	options?: RequestInit
): Promise<T> {
	const url = `${ORCHESTRATION_URL}${path}`;
	const res = await fetch(url, {
		headers: { 'Content-Type': 'application/json', ...options?.headers },
		...options
	});

	if (!res.ok) {
		const text = await res.text().catch(() => 'Unknown error');
		throw new ApiError(
			res.status,
			`API ${options?.method ?? 'GET'} ${path}: ${res.status} — ${text}`
		);
	}

	if (res.status === 204) return undefined as T;
	return res.json() as Promise<T>;
}

export function createApi(fetch: typeof globalThis.fetch) {
	return {
		get: <T>(path: string) => apiRequest<T>(fetch, path),
		post: <T>(path: string, body?: unknown) =>
			apiRequest<T>(fetch, path, {
				method: 'POST',
				body: body ? JSON.stringify(body) : undefined
			}),
		put: <T>(path: string, body: unknown) =>
			apiRequest<T>(fetch, path, {
				method: 'PUT',
				body: JSON.stringify(body)
			}),
		patch: <T>(path: string, body: unknown) =>
			apiRequest<T>(fetch, path, {
				method: 'PATCH',
				body: JSON.stringify(body)
			}),
		del: <T>(path: string) => apiRequest<T>(fetch, path, { method: 'DELETE' })
	};
}
