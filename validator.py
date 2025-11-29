import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import os
import argparse
import json
import datetime

class ColumnValidator:
    """
    Validates columns based on configurable rules.

    Supported rule types:
    - numeric: validates numeric range (min, max) and detects outliers (IQR method)
    - categorical: validates against allowed values
    - string: validates string length (min_length, max_length) and pattern (regex)

    Example config:
        rules = {
            "salary_in_usd": {
                "type": "numeric",
                "min": 0,
                "max": 1000000,
                "detect_outliers": True
            },
            "experience_level": {
                "type": "categorical",
                "allowed": ["EN", "MI", "SE", "EX"]
            },
            "job_title": {
                "type": "string",
                "min_length": 2,
                "max_length": 100
            }
        }
    """

    def __init__(self, df, rules):
        self.df = df
        self.rules = rules
        self.results = {}

    def validate(self):
        """Run all validations and return results."""
        for column, rule in self.rules.items():
            if column not in self.df.columns:
                self.results[column] = {"error": f"Column '{column}' not found"}
                continue

            col_type = rule.get("type")
            if col_type == "numeric":
                self.results[column] = self._validate_numeric(column, rule)
            elif col_type == "categorical":
                self.results[column] = self._validate_categorical(column, rule)
            elif col_type == "string":
                self.results[column] = self._validate_string(column, rule)
            else:
                self.results[column] = {"error": f"Unknown type '{col_type}'"}

        return self.results

    def _validate_numeric(self, column, rule):
        """Validate numeric column for range and outliers."""
        result = {"type": "numeric", "invalid_rows": []}
        data = self.df[column]

        min_val = rule.get("min")
        max_val = rule.get("max")

        # Check range violations
        if min_val is not None:
            below_min = self.df[data < min_val].index.tolist()
            result["below_min"] = {"count": len(below_min), "rows": below_min[:10]}

        if max_val is not None:
            above_max = self.df[data > max_val].index.tolist()
            result["above_max"] = {"count": len(above_max), "rows": above_max[:10]}

        # Detect outliers
        if rule.get("detect_outliers", False):
            method = rule.get("outlier_method", "iqr")

            if method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                # If min is set, use it as floor for lower bound
                if min_val is not None:
                    lower = max(lower, min_val)

                outlier_mask = (data < lower) | (data > upper)
                result["outliers"] = {
                    "method": "iqr",
                    "count": int(outlier_mask.sum()),
                    "bounds": {"lower": float(lower), "upper": float(upper)},
                    "rows": self.df[outlier_mask].index.tolist()[:10]
                }

            elif method == "modified_zscore":
                # Modified Z-score uses median and MAD (Median Absolute Deviation)
                # More robust to existing outliers than mean/std
                median = data.median()
                mad = (data - median).abs().median()

                # 0.6745 is the scaling factor to make MAD comparable to std for normal distribution
                if mad == 0:
                    # If MAD is 0, fall back to using deviation from median
                    modified_z = (data - median).abs()
                    threshold = rule.get("zscore_threshold", 3.5)
                    outlier_mask = modified_z > (threshold * data.std())
                else:
                    modified_z = 0.6745 * (data - median) / mad
                    threshold = rule.get("zscore_threshold", 3.5)
                    outlier_mask = modified_z.abs() > threshold

                # Calculate equivalent bounds for display
                if mad > 0:
                    lower = median - (threshold * mad / 0.6745)
                    upper = median + (threshold * mad / 0.6745)
                else:
                    lower = median - (threshold * data.std())
                    upper = median + (threshold * data.std())

                # Apply min constraint
                if min_val is not None:
                    lower = max(lower, min_val)

                result["outliers"] = {
                    "method": "modified_zscore",
                    "threshold": threshold,
                    "median": float(median),
                    "mad": float(mad),
                    "count": int(outlier_mask.sum()),
                    "bounds": {"lower": float(lower), "upper": float(upper)},
                    "rows": self.df[outlier_mask].index.tolist()[:10]
                }

        return result

    def _validate_categorical(self, column, rule):
        """Validate categorical column against allowed values."""
        result = {"type": "categorical"}
        data = self.df[column]

        allowed = rule.get("allowed", [])
        if allowed:
            invalid_mask = ~data.isin(allowed)
            invalid_rows = self.df[invalid_mask].index.tolist()
            invalid_values = data[invalid_mask].unique().tolist()
            result["invalid"] = {
                "count": len(invalid_rows),
                "values": invalid_values[:10],
                "rows": invalid_rows[:10]
            }

        result["value_counts"] = data.value_counts().to_dict()
        return result

    def _validate_string(self, column, rule):
        """Validate string column for length and pattern."""
        import re
        result = {"type": "string"}
        data = self.df[column].astype(str)

        min_len = rule.get("min_length")
        max_len = rule.get("max_length")
        pattern = rule.get("pattern")

        if min_len is not None:
            too_short = self.df[data.str.len() < min_len].index.tolist()
            result["too_short"] = {"count": len(too_short), "rows": too_short[:10]}

        if max_len is not None:
            too_long = self.df[data.str.len() > max_len].index.tolist()
            result["too_long"] = {"count": len(too_long), "rows": too_long[:10]}

        if pattern is not None:
            no_match = self.df[~data.str.match(pattern, na=False)].index.tolist()
            result["pattern_mismatch"] = {"count": len(no_match), "rows": no_match[:10]}

        return result

    def get_invalid_rows(self, column):
        """Get DataFrame rows that failed validation for a column."""
        if column not in self.results:
            return None

        result = self.results[column]
        invalid_indices = set()

        for key in ["below_min", "above_max", "outliers", "invalid", "too_short", "too_long", "pattern_mismatch"]:
            if key in result and "rows" in result[key]:
                invalid_indices.update(result[key]["rows"])

        return self.df.loc[list(invalid_indices)] if invalid_indices else pd.DataFrame()


class DataValidator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.report = {}
    

    def load_data(self):
        self.df = pd.read_csv(self.filepath)

    def validate_columns(self):
        self.report["missing_values"] = self.df.isna().sum().to_dict()
        #print(self.df.head())
        #print(self.df.isna().sum())

        missing = self.df.isna().sum()
        missing = missing[missing > 0]
        self.report["missing_values"] = {k: int(v) for k, v in missing.to_dict().items()}
    
    def duplicates_rows(self):
    
        self.report["duplicates_rows"] = int(self.df.duplicated().sum())

    def save_report(self, filepath):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
        filename = f"report_{timestamp}.json"
        full_path = os.path.join(filepath,filename)

        with open(full_path, 'w') as f:
            json.dump(self.report, f, indent = 4)
        print(f"Report saved as {filename}")

    
    def run(self, report_path = None):

        self.load_data()
        self.validate_columns()
        self.duplicates_rows()
        
        if report_path is not None:
            self.save_report(report_path)

        return self.report

class DataVisualizer:
    def __init__(self, df, columns_to_plot = None):
        self.df = df
        self.columns_to_plot = columns_to_plot or df.columns.tolist()
    
    def plot_missing_heatmap(self):
        #fetch missing values
        missing_matrix = self.df.isna()

        plt.figure(figsize=(10, 6))
        sns.heatmap(missing_matrix, cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
    
    def plot_distribution(self, column):
        
        if column not in self.df.columns:
            print(f"This column does not exist")
            return
        
        #exclude missing values
        data = self.df[column].dropna() 

        plt.figure(figsize=(8, 5))
        sns.histplot(data, kde=True, bins=20)
        plt.title(f"Distribution of {column}")
        plt.show()
    
    def plot_correlation_matrix(self):
        #select only numeric columns
        numeric_df = self.df[self.columns_to_plot].select_dtypes(include='number')

        if numeric_df.empty:
            print("No numeric columns to plot correlation matrix")
            return

        #calculate correlation
        corr = numeric_df.corr()

        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()
    
    def plot_categorical_distribution(self,column):
        if column not in self.df.columns:
            print(f"This column does not exist")
            return

        #exclude missing values
        data = self.df[column].dropna()

        #calculate each category
        counts = data.value_counts()

        plt.figure(figsize=(8, 5))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f"Categorical Distribution of {column}")
        plt.xticks(rotation=45)
        plt.show()

    def plot_outliers(self, column, lower_bound, upper_bound):
        """
        Visualize outliers using multiple plot types.

        Args:
            column: Column name to plot
            lower_bound: Lower threshold for outliers
            upper_bound: Upper threshold for outliers
        """
        if column not in self.df.columns:
            print(f"Column '{column}' does not exist")
            return

        data = self.df[column].dropna()
        is_outlier = (data < lower_bound) | (data > upper_bound)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Outlier Visualization: {column}", fontsize=14)

        # 1. Box plot
        ax1 = axes[0, 0]
        sns.boxplot(x=data, ax=ax1, color='skyblue')
        ax1.axvline(lower_bound, color='red', linestyle='--', label=f'Lower: {lower_bound:,.0f}')
        ax1.axvline(upper_bound, color='red', linestyle='--', label=f'Upper: {upper_bound:,.0f}')
        ax1.set_title("Box Plot")
        ax1.legend()

        # 2. Scatter plot with outliers highlighted
        ax2 = axes[0, 1]
        colors = ['red' if outlier else 'steelblue' for outlier in is_outlier]
        ax2.scatter(range(len(data)), data.values, c=colors, alpha=0.5, s=10)
        ax2.axhline(lower_bound, color='red', linestyle='--', label=f'Lower: {lower_bound:,.0f}')
        ax2.axhline(upper_bound, color='red', linestyle='--', label=f'Upper: {upper_bound:,.0f}')
        ax2.set_title("Scatter Plot (red = outliers)")
        ax2.set_xlabel("Row Index")
        ax2.set_ylabel(column)
        ax2.legend()

        # 3. Violin + strip plot
        ax3 = axes[1, 0]
        sns.violinplot(x=data, ax=ax3, color='lightblue', inner=None)
        sns.stripplot(x=data[~is_outlier], ax=ax3, color='steelblue', alpha=0.3, size=3)
        sns.stripplot(x=data[is_outlier], ax=ax3, color='red', alpha=0.8, size=5)
        ax3.axvline(lower_bound, color='red', linestyle='--')
        ax3.axvline(upper_bound, color='red', linestyle='--')
        ax3.set_title("Violin + Strip Plot (red = outliers)")

        # 4. Histogram with outlier regions shaded
        ax4 = axes[1, 1]
        sns.histplot(data, bins=50, ax=ax4, color='steelblue', alpha=0.7)
        ax4.axvline(lower_bound, color='red', linestyle='--', label=f'Lower: {lower_bound:,.0f}')
        ax4.axvline(upper_bound, color='red', linestyle='--', label=f'Upper: {upper_bound:,.0f}')
        ax4.axvspan(data.min(), lower_bound, alpha=0.2, color='red')
        ax4.axvspan(upper_bound, data.max(), alpha=0.2, color='red')
        ax4.set_title("Histogram with Outlier Regions")
        ax4.legend()

        plt.tight_layout()
        plt.show()

        # Print summary
        outlier_count = is_outlier.sum()
        print(f"\nOutlier Summary for '{column}':")
        print(f"  Total rows: {len(data)}")
        print(f"  Outliers: {outlier_count} ({outlier_count/len(data)*100:.2f}%)")
        print(f"  Bounds: [{lower_bound:,.2f}, {upper_bound:,.2f}]")


if __name__ == "__main__":
    #create parser
    parser = argparse.ArgumentParser(description="DataValidator: validate CSV datasets.")

    parser.add_argument(
        "--file",
        "-f",
        required = True,
        help = "Path to the CSV file you want to validate."
    )

    parser.add_argument("--report", "-r", default=".", help="Folder path to save the report.")

    parser.add_argument(
        "--num_columns",
        help = "Comma-separated list of numerical columns to plot"
    )

    parser.add_argument(
        "--cat_columns",
        help= "Comma-separated list of categorical columns to plot"
    )


    args = parser.parse_args()

    validator = DataValidator(args.file)

    report = validator.run(args.report)

    num_columns = args.num_columns.split(",") if args.num_columns else []
    cat_columns = args.cat_columns.split(",") if args.cat_columns else []

    visualizer = DataVisualizer(validator.df)
    visualizer.plot_missing_heatmap()
    visualizer.plot_correlation_matrix()

    for col in num_columns:
        visualizer.plot_distribution(col)

    for col in cat_columns:
        visualizer.plot_categorical_distribution(col)

    print(report)

    # Column validation rules for salary dataset
    salary_rules = {
        "work_year": {
            "type": "numeric",
            "min": 2020,
            "max": 2024
        },
        "experience_level": {
            "type": "categorical",
            "allowed": ["EN", "MI", "SE", "EX"]
        },
        "employment_type": {
            "type": "categorical",
            "allowed": ["FT", "PT", "CT", "FL"]
        },
        "job_title": {
            "type": "string",
            "min_length": 2,
            "max_length": 100
        },
        "salary_in_usd": {
            "type": "numeric",
            "min": 15000,
            "max": 1000000,
            "detect_outliers": True,
            "outlier_method": "modified_zscore",
            "zscore_threshold": 3.5
        },
        "remote_ratio": {
            "type": "categorical",
            "allowed": [0, 50, 100]
        },
        "company_size": {
            "type": "categorical",
            "allowed": ["S", "M", "L"]
        }
    }

    col_validator = ColumnValidator(validator.df, salary_rules)
    col_results = col_validator.validate()

    print("\n--- Column Validation Results ---")
    for col, result in col_results.items():
        print(f"\n{col}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        elif result["type"] == "numeric":
            if "below_min" in result:
                print(f"  Below min: {result['below_min']['count']}")
            if "above_max" in result:
                print(f"  Above max: {result['above_max']['count']}")
            if "outliers" in result:
                print(f"  Outliers: {result['outliers']['count']} (bounds: {result['outliers']['bounds']})")
                if result['outliers']['count'] > 0:
                    outlier_rows = col_validator.get_invalid_rows(col)
                    print(f"\n  Outlier rows:")
                    print(outlier_rows[['job_title', 'experience_level', col]].sort_values(col, ascending=False).to_string(index=False))

                    # Plot outliers
                    bounds = result['outliers']['bounds']
                    visualizer.plot_outliers(col, bounds['lower'], bounds['upper'])
        elif result["type"] == "categorical":
            if "invalid" in result:
                print(f"  Invalid values: {result['invalid']['count']} - {result['invalid']['values']}")
            else:
                print("  All values valid")
        elif result["type"] == "string":
            if "too_short" in result:
                print(f"  Too short: {result['too_short']['count']}")
            if "too_long" in result:
                print(f"  Too long: {result['too_long']['count']}")




        
        
        




        

