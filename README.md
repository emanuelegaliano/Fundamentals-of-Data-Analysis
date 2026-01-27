![Course Logo](logo.png)

# Fundamentals of Data Analysis (Fondamenti di Analisi dei Dati)

This repository collects notes, code, and the final project developed for the course **“Fondamenti di Analisi dei Dati”** taught by **Prof. Furnari** at the **University of Catania**.

Most materials are in **Italian** (notes/figures), while several code artifacts and the project notebook are written in **English**.

---

## Repository structure

```
.
├── codes/        # Standalone Python scripts (plots, statistics, small demos)
├── notes/        # LaTeX notes + compiled PDF + images
└── project/      # End-to-end data analysis project (notebook + src + outputs)
```

---

## `notes/` — Course notes (LaTeX + PDF)

- `notes/Fondamenti di Analisi Dati.tex` — main LaTeX source
- `notes/Fondamenti di Analisi Dati.pdf` — compiled PDF
- `notes/chapters/` — chapter files
- `notes/images/` — figures used throughout the notes

---

## `codes/` — Python snippets and plotting examples

A collection of small, self-contained scripts used to illustrate concepts during the course (e.g., histograms, boxplots, density plots, confidence intervals, basic hypothesis testing, metrics).

---

## `project/` — Practical project (Pizza menu dataset)

The `project/` folder contains a complete, reproducible workflow in **`assignment.ipynb`**, supported by reusable modules in `project/src/` and intermediate outputs saved in `project/results/`.

### 🍕 Dataset: *Pizza Restaurants and the Pizza They Sell*

The **Pizza Restaurants and the Pizza They Sell** dataset contains information about pizza restaurants and the pizzas they offer across different geographic locations. It includes restaurant-level metadata such as restaurant name, category, address, city, state, and country, along with menu-level details like pizza names, descriptions, and prices.

The dataset comprises several thousand pizza entries collected from multiple restaurants, making it suitable for analyzing menu composition, price distributions, naming patterns, and variability across locations. It is commonly used for exploratory data analysis, text cleaning and normalization, and basic statistical analysis of real-world commercial data.

### What the notebook does

The notebook is structured as a multi-part analysis:

- **Data understanding**
  - Load and inspect the dataset structure and types
  - Identify missing values, duplicates, and inconsistencies across business/menu fields

- **Data cleaning & preparation**
  - Handle missing values and duplicates
  - Normalize/standardize fields and formats (with special focus on menu item names)
  - Detect and treat outliers for geo coordinates and pricing-related fields

- **Exploratory data analysis**
  - Univariate analysis (numeric + categorical distributions)
  - Multivariate analysis (relationships between price, location, and business attributes)
  - Geographic labeling (e.g., *coastal vs. inland* cities) to support downstream analysis

- **Statistical inference**
  - Hypothesis-driven comparisons (e.g., coastal vs inland pricing)
  - Category-level comparisons (ANOVA and assumption checks)
  - Consistency checks between business-level price ranges and observed menu prices (including comparisons by state/province)
  - Distribution comparisons of pizza categories across administrative areas

- **Statistical modelling & machine learning**
  - Linear modelling of menu prices with interpretation
  - Supervised classification with:
    - **Logistic Regression** (feature engineering, threshold tuning, class weighting, regularization)
    - **k-NN** (evaluation + decision boundary visualization on a 2D projection)

- **Unsupervised learning**
  - **K-Means** clustering and discussion of limitations
  - **PCA** for dimensionality reduction and interpretability
  - **Gaussian Mixture Models (GMM)** on PCA space and cluster interpretation

- **Extras**
  - A small section on generating *realistic fake menu items* via controlled perturbations

### Project presentation slides

For the project presentation, see: **`project/slides.pdf`** (slides used during the final presentation).

### Datasets and outputs

- `project/datasets/` contains the raw dataset(s)
- `project/results/` contains exported CSV summaries produced during the workflow (e.g., cleaned-name summaries and epoch-based “top menu names” files)

---

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open the notebook:
   ```bash
   jupyter notebook project/assignment.ipynb
   ```