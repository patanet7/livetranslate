"""
Rate Limiting Utilities

Implements rate limiting for API endpoints and WebSocket connections.
"""

import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict, deque


class RateLimiter:
    """
    Simple in-memory rate limiter with sliding window
    """

    def __init__(self):
        self.requests = defaultdict(lambda: defaultdict(deque))
        self.lock = asyncio.Lock()

    async def is_allowed(
        self, client_id: str, endpoint: str, limit: int, window: int
    ) -> bool:
        """
        Check if request is allowed for client

        Args:
            client_id: Client identifier (IP address, user ID, etc.)
            endpoint: Endpoint identifier
            limit: Maximum requests allowed
            window: Time window in seconds

        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            current_time = time.time()
            client_requests = self.requests[client_id][endpoint]

            # Remove old requests outside the window
            while client_requests and client_requests[0] <= current_time - window:
                client_requests.popleft()

            # Check if limit exceeded
            if len(client_requests) >= limit:
                return False

            # Add current request
            client_requests.append(current_time)
            return True

    async def get_remaining(
        self, client_id: str, endpoint: str, limit: int, window: int
    ) -> int:
        """
        Get remaining requests for client

        Returns:
            Number of remaining requests
        """
        async with self.lock:
            current_time = time.time()
            client_requests = self.requests[client_id][endpoint]

            # Remove old requests outside the window
            while client_requests and client_requests[0] <= current_time - window:
                client_requests.popleft()

            return max(0, limit - len(client_requests))

    async def reset_client(self, client_id: str, endpoint: Optional[str] = None):
        """
        Reset rate limit for client

        Args:
            client_id: Client identifier
            endpoint: Optional endpoint (if None, reset all endpoints)
        """
        async with self.lock:
            if endpoint:
                if client_id in self.requests and endpoint in self.requests[client_id]:
                    self.requests[client_id][endpoint].clear()
            else:
                if client_id in self.requests:
                    del self.requests[client_id]

    async def cleanup_old_entries(self, max_age: int = 3600):
        """
        Cleanup old rate limit entries

        Args:
            max_age: Maximum age in seconds
        """
        async with self.lock:
            current_time = time.time()
            clients_to_remove = []

            for client_id, endpoints in self.requests.items():
                endpoints_to_remove = []

                for endpoint, requests in endpoints.items():
                    # Remove old requests
                    while requests and requests[0] <= current_time - max_age:
                        requests.popleft()

                    # Mark empty endpoints for removal
                    if not requests:
                        endpoints_to_remove.append(endpoint)

                # Remove empty endpoints
                for endpoint in endpoints_to_remove:
                    del endpoints[endpoint]

                # Mark empty clients for removal
                if not endpoints:
                    clients_to_remove.append(client_id)

            # Remove empty clients
            for client_id in clients_to_remove:
                del self.requests[client_id]

    async def get_stats(self) -> Dict[str, int]:
        """
        Get rate limiter statistics

        Returns:
            Dictionary with statistics
        """
        async with self.lock:
            total_clients = len(self.requests)
            total_endpoints = sum(
                len(endpoints) for endpoints in self.requests.values()
            )
            total_requests = sum(
                len(requests)
                for endpoints in self.requests.values()
                for requests in endpoints.values()
            )

            return {
                "total_clients": total_clients,
                "total_endpoints": total_endpoints,
                "total_requests": total_requests,
            }
