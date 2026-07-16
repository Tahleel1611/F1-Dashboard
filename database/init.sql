CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(50) PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL,
    team_color_hex CHAR(7) NOT NULL
);

CREATE TABLE IF NOT EXISTS drivers (
    driver_code CHAR(3) PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    permanent_number SMALLINT NOT NULL
);

CREATE TABLE IF NOT EXISTS races (
    race_id SERIAL PRIMARY KEY,
    season_year SMALLINT NOT NULL,
    round_number SMALLINT NOT NULL,
    circuit_name VARCHAR(100) NOT NULL,
    country VARCHAR(50) NOT NULL,
    CONSTRAINT unique_season_round UNIQUE (season_year, round_number)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id SERIAL PRIMARY KEY,
    race_id INT NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
    session_type VARCHAR(10) NOT NULL,
    date_held DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS laps (
    lap_id SERIAL PRIMARY KEY,
    session_id INT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    driver_code CHAR(3) NOT NULL REFERENCES drivers(driver_code),
    team_id VARCHAR(50) NOT NULL REFERENCES teams(team_id),
    lap_number SMALLINT NOT NULL,
    lap_time_ms INT,
    sector_1_ms INT,
    sector_2_ms INT,
    sector_3_ms INT,
    tyre_compound VARCHAR(15) NOT NULL,
    tyre_life_at_start SMALLINT NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_samples (
    sample_id BIGSERIAL PRIMARY KEY,
    lap_id INT NOT NULL REFERENCES laps(lap_id) ON DELETE CASCADE,
    distance_meters REAL NOT NULL,
    timestamp_offset_ms INT NOT NULL,
    speed_kph SMALLINT NOT NULL,
    throttle_pct SMALLINT NOT NULL,
    brake_active BOOLEAN NOT NULL,
    engine_rpm SMALLINT NOT NULL,
    gear_engaged SMALLINT NOT NULL,
    drs_status SMALLINT NOT NULL,
    coordinate_x REAL NOT NULL,
    coordinate_y REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_telemetry_lap_distance
ON telemetry_samples(lap_id, distance_meters);