import { browser } from '$app/environment';
import { PUBLIC_WS_URL, PUBLIC_APP_NAME } from '$env/static/public';

export const WS_BASE = browser
	? (PUBLIC_WS_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`)
	: '';
export const APP_NAME = PUBLIC_APP_NAME || 'LiveTranslate';
