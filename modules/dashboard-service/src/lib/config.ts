import { browser } from '$app/environment';
import { PUBLIC_WS_URL, PUBLIC_APP_NAME } from '$env/static/public';

export const WS_BASE = browser ? (PUBLIC_WS_URL || 'ws://localhost:3000') : '';
export const APP_NAME = PUBLIC_APP_NAME || 'LiveTranslate';
