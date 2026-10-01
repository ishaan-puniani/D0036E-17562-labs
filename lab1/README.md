# Lab 1 — California Housing Exploration

Explore `housing.csv` with custom Python functions to answer the lab research questions (counts, means/medians by `ocean_proximity`, histograms, quality and demographic comparisons).

## Prerequisites

- Python 3.9+
- pip

## Setup

From the project root:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### GitHub Codespaces

1. Open the repo in a Codespace (**Code → Codespaces → Create codespace**).
2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Run the exploration script (see below).

## Run

Quick CSV check:

```bash
python3 csv_read.py
```

Full research exploration (prints Q1–Q10 answers + interpretations, saves histograms):

```bash
python3 explore_housing.py
```

Histograms are written to `figures/histograms.png`.

## Project layout

| File | Description |
|------|-------------|
| `explore_housing.py` | Custom exploration functions + Q1–Q10 answers |
| `csv_read.py` | Minimal load / preview of `housing.csv` |
| `housing.csv` | California housing districts dataset |
| `requirements.txt` | Python dependencies |
| `figures/histograms.png` | Created after running `explore_housing.py` |

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Load and aggregate the CSV |
| `matplotlib` | Histograms for selected features |

```bash
python3 -m pip install -r requirements.txt
```

## Research questions covered

1. Number of geographical units  
2. Overall mean house value  
3. Mean house value by `ocean_proximity`  
4. Mean vs median by category  
5–6. Histograms and distribution notes (including capping)  
7. Expensive/large vs cheap/small categories  
8. House quality (age, rooms) by category  
9. Demographics (population, households, income) by category  
10. Extra observations and follow-up questions  
