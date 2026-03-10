// modules/dashboard-service/src/lib/api/export.ts

/**
 * Export API — generates download URLs that point to our SvelteKit proxy routes.
 * These proxy routes forward to the orchestration service's export endpoints.
 */
export const exportApi = {
	transcriptUrl: (meetingId: string, format: string = 'srt') =>
		`/api/export/meetings/${meetingId}/transcript?format=${format}`,

	translationsUrl: (meetingId: string, lang: string, format: string = 'srt') =>
		`/api/export/meetings/${meetingId}/translations?lang=${lang}&format=${format}`,

	archiveUrl: (meetingId: string) => `/api/export/meetings/${meetingId}/archive`
};
