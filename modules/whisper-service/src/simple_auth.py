#!/usr/bin/env python3
"""
Simple Authentication and Authorization

Basic token-based authentication system for WebSocket connections.
Designed to be simple and lightweight while providing basic security.
"""

import logging
import time
import hashlib
import secrets
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """Simple user roles"""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"

@dataclass
class AuthToken:
    """Simple authentication token"""
    token: str
    user_id: str
    role: UserRole
    created_at: datetime
    expires_at: datetime
    connection_id: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if token is valid"""
        return not self.is_expired()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "token": self.token,
            "user_id": self.user_id,
            "role": self.role.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "connection_id": self.connection_id
        }

class SimpleAuth:
    """Simple authentication and authorization system"""
    
    def __init__(self):
        # In-memory storage (in production, use a database)
        self.tokens: Dict[str, AuthToken] = {}
        self.users: Dict[str, Dict] = {}
        self.connection_tokens: Dict[str, str] = {}  # connection_id -> token
        
        # Simple configuration
        self.token_expiry_hours = 24  # Tokens expire after 24 hours
        self.guest_token_expiry_minutes = 60  # Guest tokens expire after 1 hour
        self.max_tokens_per_user = 5  # Max concurrent tokens per user
        
        # Create default admin user
        self._create_default_users()
        
        logger.info("Simple authentication system initialized")
    
    def _create_default_users(self):
        """Create default users for testing"""
        # Default admin user
        self.users["admin"] = {
            "user_id": "admin",
            "password_hash": self._hash_password("admin123"),
            "role": UserRole.ADMIN,
            "created_at": datetime.now()
        }
        
        # Default regular user
        self.users["user"] = {
            "user_id": "user",
            "password_hash": self._hash_password("user123"),
            "role": UserRole.USER,
            "created_at": datetime.now()
        }
        
        logger.info("Created default users: admin, user")
    
    def _hash_password(self, password: str) -> str:
        """Simple password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_token(self) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(32)
    
    def create_guest_token(self) -> AuthToken:
        """Create a guest token for anonymous access"""
        token = self._generate_token()
        user_id = f"guest_{int(time.time())}"
        
        auth_token = AuthToken(
            token=token,
            user_id=user_id,
            role=UserRole.GUEST,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=self.guest_token_expiry_minutes)
        )
        
        self.tokens[token] = auth_token
        logger.info(f"Created guest token for {user_id}")
        return auth_token
    
    def authenticate(self, user_id: str, password: str) -> Optional[AuthToken]:
        """Authenticate user with username/password"""
        user = self.users.get(user_id)
        if not user:
            logger.warning(f"Authentication failed: user {user_id} not found")
            return None
        
        password_hash = self._hash_password(password)
        if password_hash != user["password_hash"]:
            logger.warning(f"Authentication failed: invalid password for {user_id}")
            return None
        
        # Check token limit
        user_tokens = [t for t in self.tokens.values() if t.user_id == user_id and t.is_valid()]
        if len(user_tokens) >= self.max_tokens_per_user:
            # Remove oldest token
            oldest_token = min(user_tokens, key=lambda t: t.created_at)
            self.revoke_token(oldest_token.token)
        
        # Create new token
        token = self._generate_token()
        auth_token = AuthToken(
            token=token,
            user_id=user_id,
            role=user["role"],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=self.token_expiry_hours)
        )
        
        self.tokens[token] = auth_token
        logger.info(f"User {user_id} authenticated successfully")
        return auth_token
    
    def validate_token(self, token: str) -> Optional[AuthToken]:
        """Validate an authentication token"""
        auth_token = self.tokens.get(token)
        if not auth_token:
            return None
        
        if auth_token.is_expired():
            # Clean up expired token
            self.revoke_token(token)
            return None
        
        return auth_token
    
    def revoke_token(self, token: str) -> bool:
        """Revoke an authentication token"""
        if token in self.tokens:
            auth_token = self.tokens[token]
            
            # Remove connection association
            if auth_token.connection_id:
                self.connection_tokens.pop(auth_token.connection_id, None)
            
            # Remove token
            del self.tokens[token]
            logger.info(f"Token revoked for user {auth_token.user_id}")
            return True
        
        return False
    
    def associate_connection(self, token: str, connection_id: str) -> bool:
        """Associate a token with a WebSocket connection"""
        auth_token = self.validate_token(token)
        if not auth_token:
            return False
        
        # Remove any existing association for this connection
        old_token = self.connection_tokens.get(connection_id)
        if old_token and old_token in self.tokens:
            self.tokens[old_token].connection_id = None
        
        # Create new association
        auth_token.connection_id = connection_id
        self.connection_tokens[connection_id] = token
        
        logger.info(f"Associated token with connection {connection_id}")
        return True
    
    def get_connection_auth(self, connection_id: str) -> Optional[AuthToken]:
        """Get authentication info for a connection"""
        token = self.connection_tokens.get(connection_id)
        if not token:
            return None
        
        return self.validate_token(token)
    
    def disconnect_connection(self, connection_id: str):
        """Handle connection disconnection"""
        token = self.connection_tokens.pop(connection_id, None)
        if token and token in self.tokens:
            self.tokens[token].connection_id = None
            logger.info(f"Disconnected connection {connection_id}")
    
    def check_permission(self, token: str, required_role: UserRole) -> bool:
        """Check if token has required permission level"""
        auth_token = self.validate_token(token)
        if not auth_token:
            return False
        
        # Role hierarchy: ADMIN > USER > GUEST
        role_levels = {
            UserRole.GUEST: 0,
            UserRole.USER: 1,
            UserRole.ADMIN: 2
        }
        
        user_level = role_levels.get(auth_token.role, 0)
        required_level = role_levels.get(required_role, 0)
        
        return user_level >= required_level
    
    def cleanup_expired_tokens(self):
        """Clean up expired tokens"""
        expired_tokens = [
            token for token, auth_token in self.tokens.items()
            if auth_token.is_expired()
        ]
        
        for token in expired_tokens:
            self.revoke_token(token)
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
    
    def get_statistics(self) -> Dict:
        """Get authentication statistics"""
        valid_tokens = [t for t in self.tokens.values() if t.is_valid()]
        
        role_counts = {}
        for role in UserRole:
            role_counts[role.value] = len([t for t in valid_tokens if t.role == role])
        
        return {
            "total_users": len(self.users),
            "active_tokens": len(valid_tokens),
            "connected_tokens": len(self.connection_tokens),
            "role_distribution": role_counts,
            "token_expiry_hours": self.token_expiry_hours,
            "guest_token_expiry_minutes": self.guest_token_expiry_minutes,
            "max_tokens_per_user": self.max_tokens_per_user
        }

# Global authentication instance
simple_auth = SimpleAuth()

# Authentication middleware for message router
def auth_middleware(context):
    """Simple authentication middleware"""
    from message_router import RoutePermission
    
    # Skip auth for public routes
    if context.route_info and context.route_info.permission == RoutePermission.PUBLIC:
        return True
    
    # Get authentication info
    auth_token = simple_auth.get_connection_auth(context.connection_id)
    if not auth_token:
        logger.warning(f"Unauthenticated access attempt from {context.connection_id}")
        return False
    
    # Check role permissions
    if context.route_info:
        if context.route_info.permission == RoutePermission.AUTHENTICATED:
            required_role = UserRole.USER
        elif context.route_info.permission == RoutePermission.ADMIN:
            required_role = UserRole.ADMIN
        else:
            required_role = UserRole.GUEST
        
        if not simple_auth.check_permission(auth_token.token, required_role):
            logger.warning(f"Insufficient permissions for {context.connection_id} "
                         f"(has: {auth_token.role.value}, needs: {required_role.value})")
            return False
    
    # Add auth info to context
    context.middleware_data['auth'] = {
        'user_id': auth_token.user_id,
        'role': auth_token.role.value,
        'token': auth_token.token
    }
    
    return True 