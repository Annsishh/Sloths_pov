# AI Product Investigator - Synthetic Data Generation Foundation

## Overview
This package is the data-generation, simulation, and consistency-validation engine for the **AI Product Investigator**. It generates synthetic, multi-table SaaS product data that models real-world business dynamics, release lifecycles, user interactions, and specific failure modes.

## Strict Separation of Investigator Data & Ground Truth
To ensure unbiased evaluation, the data is split into two isolated domains:
1. **Investigator Accessible Data (`investigator_data/`)**: Contains only observable logs, tables, metrics, and customer feedback that an AI agent or data analyst would have access to in a production data warehouse.
2. **Ground Truth & Evaluation Artifacts (`ground_truth/`)**: Contains scenario metadata, root-cause tags, injected failure parameters, affected cohorts, and expected metrics impact. The investigator agent must **never** be given access to this directory.

## Directory Structure
```
sloths_pov/
│
├── data/                               # Generated datasets
│   ├── baseline/                       # Healthy SaaS operating baseline
│   │   ├── investigator_data/          # Observable datasets (users, events, metrics, etc.)
│   │   └── ground_truth/               # Injected causes & evaluation metadata
│   └── <scenario_name>/                # Injected anomaly scenarios
│       ├── investigator_data/
│       └── ground_truth/
│
├── src/
│   └── investigator_data/
│       ├── __init__.py
│       ├── config.py                   # Global constants and default configurations
│       ├── schemas.py                  # Table definitions & column specifications
│       ├── scenarios.py                # Scenario definitions & parameterizations
│       ├── engine.py                   # Core synthetic data generator
│       └── validator.py                # Integrity, temporal & scenario consistency validator
│
├── tests/
│   ├── __init__.py
│   ├── test_reproducibility.py        # Seed reproducibility tests
│   ├── test_integrity.py              # Referential and temporal integrity tests
│   ├── test_scenarios.py              # Scenario observable effect verification tests
│   └── test_separation.py             # Ground-truth isolation tests
│
├── generate_data.py                    # CLI script to generate scenarios
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Requirements
Install the minimal dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Datasets
Run the CLI generator:

```bash
# Generate baseline (normal operations)
python generate_data.py --scenario baseline --output-dir data/baseline

# Generate a product regression (checkout payment failure on mobile v2.4.0)
python generate_data.py --scenario product_regression --output-dir data/product_regression

# Generate an acquisition mix shift (surge of low-converting social ad signups)
python generate_data.py --scenario acquisition_shift --output-dir data/acquisition_shift

# Generate an analytics tracking failure (missing checkout_success events on Safari)
python generate_data.py --scenario tracking_failure --output-dir data/tracking_failure
```

Or generate all supported scenarios in one command:
```bash
python generate_data.py --all --output-dir data
```

### 3. Run Validation Tests
Execute the automated test suite with pytest:
```bash
pytest tests/ -v
```

