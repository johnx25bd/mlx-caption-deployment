FROM postgres:15

# Set environment variables
ENV POSTGRES_DB=user_logs
ENV POSTGRES_USER=logger
ENV POSTGRES_PASSWORD=secure_password

# Copy initialization SQL script
COPY init.sql /docker-entrypoint-initdb.d/init.sql

EXPOSE 5432