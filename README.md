# DataValidator

A Python tool for validating and cleaning CSV datasets. It performs basic data quality checks such as missing values, duplicate rows, and can be extended with custom validation rules. Generates a JSON report summarizing the findings.

---

## Features

- Load any CSV dataset.  
- Check for missing values in all columns.  
- Detect duplicate rows.  
- Generate a structured JSON report.  
- Optional: save the report to a specified file path.  
- Easily extensible for additional validation rules (e.g., outlier detection, categorical value checks, visualizations).

---

## Dataset

This tool can work with **any CSV dataset**. For demonstration purposes, it has been tested with a dataset from **Kaggle**:  
[Salary Insights by Job Role](https://www.kaggle.com/datasets/zahranusrat/salary)

---

## Installation

Make sure you have Python 3 installed and the required packages:

```bash
pip install pandas
