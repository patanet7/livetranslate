"""
Security Utilities

Provides security-related utilities for the orchestration service.
"""

import hashlib
import hmac
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class SecurityUtils:
    """
    Security utilities for authentication and authorization
    """

    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)

    def generate_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """
        Generate JWT token

        Args:
            payload: Token payload
            expires_in: Expiration time in seconds

        Returns:
            JWT token string
        """
        payload["exp"] = datetime.utcnow() + timedelta(seconds=expires_in)
        payload["iat"] = datetime.utcnow()

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token

        Args:
            token: JWT token string

        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify password against hash

        Args:
            password: Plain text password
            hashed: Hashed password

        Returns:
            True if password matches, False otherwise
        """
        return self.hash_password(password) == hashed

    def generate_api_key(self, length: int = 32) -> str:
        """
        Generate API key

        Args:
            length: Key length

        Returns:
            Random API key
        """
        return secrets.token_urlsafe(length)

    def verify_hmac_signature(self, message: str, signature: str, key: str) -> bool:
        """
        Verify HMAC signature

        Args:
            message: Original message
            signature: HMAC signature
            key: Secret key

        Returns:
            True if signature is valid, False otherwise
        """
        expected_signature = hmac.new(
            key.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for security

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove directory traversal attempts
        filename = filename.replace("..", "").replace("/", "").replace("\\", "")

        # Remove special characters
        allowed_chars = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
        )
        filename = "".join(c for c in filename if c in allowed_chars)

        # Ensure filename is not empty
        if not filename:
            filename = "file"

        return filename

    def validate_ip_address(self, ip: str) -> bool:
        """
        Validate IP address format

        Args:
            ip: IP address string

        Returns:
            True if valid IP, False otherwise
        """
        import ipaddress

        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def is_safe_url(self, url: str, allowed_hosts: list = None) -> bool:
        """
        Check if URL is safe for redirection

        Args:
            url: URL to check
            allowed_hosts: List of allowed hosts

        Returns:
            True if URL is safe, False otherwise
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)

            # Check for dangerous schemes
            if parsed.scheme not in ("http", "https", ""):
                return False

            # Check allowed hosts
            if allowed_hosts and parsed.netloc:
                if parsed.netloc not in allowed_hosts:
                    return False

            return True
        except Exception:
            return False
