\echo 'Running initialization script...'

CREATE TABLE IF NOT EXISTS user_activity (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    action_type VARCHAR(50),
    action_details JSONB,
    ip_address VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finetuned BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS user_images (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_id TEXT NOT NULL,
    caption TEXT NOT NULL,
    finetuned BOOLEAN DEFAULT FALSE,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL
);

\echo 'Initialization complete.'
