import type { Handle } from '@sveltejs/kit';

export const handle: Handle = async ({ event, resolve }) => {
	const start = performance.now();
	const response = await resolve(event);
	const duration = Math.round(performance.now() - start);
	console.log(`${event.request.method} ${event.url.pathname} ${response.status} ${duration}ms`);
	return response;
};
