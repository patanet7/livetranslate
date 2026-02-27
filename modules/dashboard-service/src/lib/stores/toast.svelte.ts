import { toast } from 'svelte-sonner';

class ToastStore {
	add(message: string, type: 'success' | 'error' | 'info' | 'warning' = 'info', duration = 5000) {
		switch (type) {
			case 'success':
				toast.success(message, { duration });
				break;
			case 'error':
				toast.error(message, { duration });
				break;
			case 'warning':
				toast.warning(message, { duration });
				break;
			case 'info':
			default:
				toast.info(message, { duration });
				break;
		}
	}

	success(message: string) {
		this.add(message, 'success');
	}
	error(message: string) {
		this.add(message, 'error', 8000);
	}
	warning(message: string) {
		this.add(message, 'warning');
	}
	info(message: string) {
		this.add(message, 'info');
	}
}

export const toastStore = new ToastStore();
