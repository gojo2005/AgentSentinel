from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config
from sqlalchemy import pool

from app.storage.models import Base
from core.config import DATABASE_URL

# Alembic configuration object.
config = context.config

# Configure logging using the Alembic configuration file.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Escape percent signs for ConfigParser interpolation.
# The original URL is passed directly to the migration context.
config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))

# Provide SQLAlchemy metadata for Alembic autogenerate support.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""

    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a live database connection."""

    # SQLite requires this argument for connections used across threads.
    connect_args = (
        {"check_same_thread": False}
        if DATABASE_URL.startswith("sqlite")
        else {}
    )

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
        hide_parameters=True,
    )

    try:
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
            )

            with context.begin_transaction():
                context.run_migrations()
    finally:
        # Release database resources even if migration execution fails.
        connectable.dispose()


# Select the appropriate migration mode.
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()