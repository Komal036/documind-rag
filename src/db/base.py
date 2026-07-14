"""
DocuMind Database Base
-----------------------
Shared declarative base for all SQLAlchemy models.
Kept in its own module so both models.py and connection.py can import it
without a circular dependency.
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass