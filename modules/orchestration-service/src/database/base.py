#!/usr/bin/env python3
"""
SQLAlchemy Base for Database Models

Defines the declarative base that all database models inherit from.
This ensures all models share the same metadata and can be properly
related to each other.

Usage:
    from database.base import Base

    class MyModel(Base):
        __tablename__ = "my_table"
        ...
"""

from sqlalchemy.orm import declarative_base

# Create the declarative base
# All database models should inherit from this Base
Base = declarative_base()

__all__ = ['Base']
