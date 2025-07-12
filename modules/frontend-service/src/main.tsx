import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Global styles
import './styles/globals.css';

// Performance monitoring (optional)
if (process.env.NODE_ENV === 'development') {
  // Enable React DevTools profiler in development
  const root = document.getElementById('root');
  if (root) {
    root.dataset.reactProfiler = 'true';
  }
}

// Error handling for uncaught errors
window.addEventListener('error', (event) => {
  console.error('Uncaught error:', event.error);
  // Could send to error reporting service
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  // Could send to error reporting service
});

// Render the app
const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element not found');
}

const root = ReactDOM.createRoot(rootElement);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);