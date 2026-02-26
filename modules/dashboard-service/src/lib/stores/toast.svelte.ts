import { browser } from '$app/environment';

export interface Toast {
	id: string;
	message: string;
	type: 'success' | 'error' | 'info' | 'warning';
	duration?: number;
}

class ToastStore {
	toasts = $state<Toast[]>([]);

	add(message: string, type: Toast['type'] = 'info', duration = 5000) {
		const id = crypto.randomUUID();
		this.toasts = [...this.toasts, { id, message, type, duration }];

		if (duration > 0 && browser) {
			setTimeout(() => this.dismiss(id), duration);
		}

		return id;
	}

	success(message: string) {
		return this.add(message, 'success');
	}
	error(message: string) {
		return this.add(message, 'error', 8000);
	}
	warning(message: string) {
		return this.add(message, 'warning');
	}
	info(message: string) {
		return this.add(message, 'info');
	}

	dismiss(id: string) {
		this.toasts = this.toasts.filter((t) => t.id !== id);
	}
}

export const toastStore = new ToastStore();
