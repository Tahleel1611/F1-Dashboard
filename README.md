# F1 Performance Analytics

Formula 1 performance analytics dashboard for exploring race results, driver performance, and season trends.

## Overview

This project provides a simple starting point for analyzing F1 data and presenting key insights through a dashboard-style interface. It is intended to support comparisons across drivers, constructors, and races.

## Features

- Driver and constructor performance analysis
- Race-by-race trend review
- Comparison of results across seasons
- Data visualization for F1 insights

## Getting Started

1. Clone the repository.
2. Install the project dependencies.
3. From the `f1-perf-analytics/` folder, run the launcher script:

	```powershell
	.\run-dashboard.ps1
	```

	If you prefer to launch Streamlit directly, run:

	```powershell
	streamlit run frontend/dashboard.py
	```

	If you are in a different directory, use the absolute path:

	```powershell
	streamlit run c:\Users\ASUS\OneDrive\Documents\F1-Project\f1-perf-analytics\frontend\dashboard.py
	```

4. To run the full stack with Docker, use:

	```powershell
	docker compose up --build
	```

## Project Structure

- `data/` - datasets used by the project
- `src/` - application source code
- `assets/` - images and static resources
- `README.md` - project documentation

## Usage

Open the dashboard in your browser after Streamlit starts, then use the sidebar controls to inspect driver stats, team performance, and race trends.

## Data

The project is designed to work with Formula 1 race and season data such as results, standings, lap performance, and team metrics.

## Contributing

Contributions are welcome. Keep changes focused and document any new data sources or analysis steps.

