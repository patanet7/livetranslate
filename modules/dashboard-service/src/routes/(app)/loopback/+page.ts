/**
 * Disable SSR for the loopback page.
 *
 * This page is inherently browser-only: it uses AudioCapture (getUserMedia),
 * WebSocket, AudioContext, and navigator.mediaDevices. SSR renders the HTML
 * correctly on the server but the client-side hydration fails because
 * $effect blocks in child components write to the shared loopback store
 * during initialization, causing reactive cascades that differ from the
 * server-rendered output.
 *
 * With ssr=false the entire page tree (including layouts) renders client-side.
 * This is the correct trade-off: a brief flash before JS loads is acceptable
 * for a page that requires browser audio APIs to function.
 */
export const ssr = false;
