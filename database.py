"""
Database Configuration Module for AI Assistant Pro
Provides robust database connection handling with retry logic,
connection pooling, and comprehensive error handling.
"""

import os
import sys
import time
import logging
from typing import Generator, Optional
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    Text,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Database Configuration ---
class DatabaseConfig:
    """Centralized database configuration management."""
    
    def __init__(self):
        self.host = os.getenv("DB_HOST", "db")
        self.port = os.getenv("DB_PORT", "3306")
        self.user = os.getenv("DB_USER", "myuser")
        self.password = os.getenv("DB_PASSWORD", "mypassword")
        self.name = os.getenv("DB_NAME", "asistenteia_db")
        self.echo = os.getenv("DB_ECHO", "false").lower() == "true"
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        
        # Retry settings
        self.max_retries = int(os.getenv("DB_MAX_RETRIES", "5"))
        self.retry_delay = int(os.getenv("DB_RETRY_DELAY", "5"))
    
    @property
    def database_url(self) -> str:
        """Construct database URL with proper escaping."""
        return (
            f"mysql+mysqlconnector://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )
    
    def get_engine_kwargs(self) -> dict:
        """Get SQLAlchemy engine configuration."""
        return {
            "poolclass": QueuePool,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": True,  # Verify connections before using
            "echo": self.echo,
            "connect_args": {
                "charset": "utf8mb4",
                "use_unicode": True,
            }
        }


# --- SQLAlchemy Base and Models ---
Base = declarative_base()


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps."""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )


class Company(Base, TimestampMixin):
    """Company model representing an organization."""
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    
    # Relationships
    users = relationship(
        "User",
        back_populates="company",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    documents = relationship(
        "Document",
        back_populates="company",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<Company(id={self.id}, name='{self.name}')>"


class User(Base, TimestampMixin):
    """User model representing system users."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Integer, default=1, nullable=False)  # Using Integer for boolean
    company_id = Column(
        Integer,
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Relationships
    company = relationship("Company", back_populates="users")
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}')>"


class Document(Base, TimestampMixin):
    """Document model for storing uploaded files metadata."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=True)  # Size in bytes
    mime_type = Column(String(100), nullable=True)
    company_id = Column(
        Integer,
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Relationships
    company = relationship("Company", back_populates="documents")
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}')>"


# --- Database Connection Management ---
class DatabaseManager:
    """Manages database connections and session lifecycle."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._is_connected = False
    
    def connect(self) -> bool:
        """
        Establish database connection with retry logic.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self._is_connected:
            logger.warning("Database is already connected")
            return True
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info(
                    f"Attempting to connect to MySQL at {self.config.host}:{self.config.port} "
                    f"(Attempt {attempt}/{self.config.max_retries})"
                )
                
                # Create engine with connection pooling
                self.engine = create_engine(
                    self.config.database_url,
                    **self.config.get_engine_kwargs()
                )
                
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(sqlalchemy.text("SELECT 1"))
                
                logger.info("✓ Database connection established successfully")
                
                # Create tables if they don't exist
                self._create_tables()
                
                # Initialize session factory
                self.SessionLocal = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self.engine
                )
                
                self._is_connected = True
                self._setup_event_listeners()
                
                return True
                
            except OperationalError as e:
                logger.warning(
                    f"✗ Database connection failed: {e}. "
                    f"Retrying in {self.config.retry_delay}s..."
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(
                        "✗ FATAL: Could not connect to database after "
                        f"{self.config.max_retries} attempts"
                    )
                    return False
                    
            except Exception as e:
                logger.error(f"✗ Unexpected error during database connection: {e}")
                return False
        
        return False
    
    def _create_tables(self):
        """Create all tables defined in models."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✓ Database tables created/verified successfully")
        except SQLAlchemyError as e:
            logger.error(f"✗ Error creating database tables: {e}")
            raise
    
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for connection management."""
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            logger.debug("Database connection acquired from pool")
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")
    
    def disconnect(self):
        """Disconnect from database and dispose of connection pool."""
        if self.engine:
            self.engine.dispose()
            logger.info("✓ Database connection closed")
            self._is_connected = False
    
    def get_session(self) -> Session:
        """
        Create a new database session.
        
        Returns:
            Session: SQLAlchemy session object
            
        Raises:
            RuntimeError: If database is not connected
        """
        if not self._is_connected or self.SessionLocal is None:
            raise RuntimeError(
                "Database is not connected. Call connect() first."
            )
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations.
        
        Usage:
            with db_manager.session_scope() as session:
                # perform database operations
                pass
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"✗ Transaction rolled back due to error: {e}")
            raise
        finally:
            session.close()
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._is_connected
    
    def health_check(self) -> bool:
        """
        Perform a health check on the database connection.
        
        Returns:
            bool: True if connection is healthy, False otherwise
        """
        if not self._is_connected:
            return False
        
        try:
            with self.session_scope() as session:
                session.execute(sqlalchemy.text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"✗ Database health check failed: {e}")
            return False


# --- Global Database Instance ---
db_manager = DatabaseManager()


# --- Dependency Injection for FastAPI ---
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    Usage:
        @app.get("/")
        def read_root(db: Session = Depends(get_db)):
            # use db session
            pass
    """
    if not db_manager.is_connected:
        raise RuntimeError(
            "Database is not connected. Ensure connect_to_db() is called on startup."
        )
    
    session = db_manager.get_session()
    try:
        yield session
    except SQLAlchemyError as e:
        logger.error(f"✗ Database session error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


# --- Convenience Functions ---
def connect_to_db(config: Optional[DatabaseConfig] = None) -> bool:
    """
    Initialize database connection (convenience wrapper).
    
    Args:
        config: Optional DatabaseConfig instance
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    if config:
        global db_manager
        db_manager = DatabaseManager(config)
    
    success = db_manager.connect()
    
    if not success:
        logger.critical("✗ Application cannot start without database connection")
        sys.exit(1)
    
    return success


def disconnect_from_db():
    """Disconnect from database (convenience wrapper)."""
    db_manager.disconnect()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager


# --- Startup/Shutdown Hooks for FastAPI ---
async def startup_event():
    """FastAPI startup event handler."""
    logger.info("🚀 Starting up application...")
    connect_to_db()


async def shutdown_event():
    """FastAPI shutdown event handler."""
    logger.info("🛑 Shutting down application...")
    disconnect_from_db()


if __name__ == "__main__":
    # Test database connection
    logger.info("Testing database connection...")
    if connect_to_db():
        logger.info("✓ Database test successful")
        
        # Test health check
        if db_manager.health_check():
            logger.info("✓ Database health check passed")
        
        # Test session creation
        with db_manager.session_scope() as session:
            result = session.execute(sqlalchemy.text("SELECT DATABASE()"))
            db_name = result.fetchone()[0]
            logger.info(f"✓ Connected to database: {db_name}")
        
        disconnect_from_db()
    else:
        logger.error("✗ Database test failed")
        sys.exit(1)