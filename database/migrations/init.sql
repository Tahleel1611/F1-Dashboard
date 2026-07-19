-- Phase 3: Relational Database Schema Architecture for F1 Performance Analytics
-- Enforces 3NF Normalization and provides time-series indexing.

-- 1. Lookup Tables (Low Churn Data)
CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(50) PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL,
    team_color_hex CHAR(7) NOT NULL
);

CREATE TABLE IF NOT EXISTS drivers (
    driver_code CHAR(3) PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    permanent_number INT NOT NULL
);

-- 2. Context Calendars
CREATE TABLE IF NOT EXISTS races (
    race_id SERIAL PRIMARY KEY,
    season_year INT NOT NULL,
    round_number INT NOT NULL,
    circuit_name VARCHAR(100) NOT NULL,
    country VARCHAR(50) NOT NULL,
    CONSTRAINT unique_season_round UNIQUE (season_year, round_number)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id SERIAL PRIMARY KEY,
    race_id INT REFERENCES races(race_id) ON DELETE CASCADE,
    session_type VARCHAR(10) NOT NULL, -- 'FP1', 'FP2', 'FP3', 'Q', 'Sprint', 'Race'
    date_held DATE NOT NULL
);

-- 3. The Lap Layer (Aggregated Metric Context)
CREATE TABLE IF NOT EXISTS laps (
    lap_id SERIAL PRIMARY KEY,
    session_id INT REFERENCES sessions(session_id) ON DELETE CASCADE,
    driver_code CHAR(3) REFERENCES drivers(driver_code),
    team_id VARCHAR(50) REFERENCES teams(team_id),
    lap_number INT NOT NULL,
    lap_time_ms INT, -- Storing total duration in ms for fast numerical comparisons
    sector_1_ms INT,
    sector_2_ms INT,
    sector_3_ms INT,
    tyre_compound VARCHAR(15) NOT NULL, -- 'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'
    tyre_life_at_start INT NOT NULL,
    is_valid BOOLEAN DEFAULT TRUE
);

-- 4. High-Density Time-Series Telemetry Table
CREATE TABLE IF NOT EXISTS telemetry_samples (
    sample_id BIGSERIAL PRIMARY KEY,
    lap_id INT REFERENCES laps(lap_id) ON DELETE CASCADE,
    distance_meters REAL NOT NULL, -- REAL type handles 4-byte floating points, optimized for space
    timestamp_offset_ms INT NOT NULL,
    speed_kph SMALLINT NOT NULL,   -- SMALLINT bounds to 32,767; perfect for car speeds up to 360 km/h
    throttle_pct SMALLINT NOT NULL, -- Range 0-100
    brake_active BOOLEAN NOT NULL,
    engine_rpm SMALLINT NOT NULL,  -- F1 engines rev up to 15,000 RPM
    gear_engaged SMALLINT NOT NULL, -- Range 1-8
    drs_status SMALLINT NOT NULL,
    coordinate_x REAL NOT NULL,    -- Global track spatial layout positioning coordinates
    coordinate_y REAL NOT NULL
);

-- 5. High-Performance Optimization Performance Indexing
CREATE INDEX IF NOT EXISTS idx_laps_lookup 
ON laps(session_id, driver_code, lap_number);

-- CRITICAL: Composite index optimizing the synchronized multi-grid layout cursor search paths.
-- Speeds up retrieval of distance arrays when plotting the synchronized visual charts.
CREATE INDEX IF NOT EXISTS idx_telemetry_lap_distance 
ON telemetry_samples(lap_id, distance_meters);