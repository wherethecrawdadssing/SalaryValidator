import argparse
import datetime
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class ColumnValidator:
    """
    Rule-based column validation with support for:
    - numeric: range checks + outlier detection (IQR or modified z-score)
    - categorical: allowed values check
    - string: length and regex pattern checks
    - multivariate: Mahalanobis distance for correlated columns
    """

    def __init__(self, df, rules):
        self.df = df
        self.rules = rules
        self.results = {}

    def validate(self):
        for name, rule in self.rules.items():
            rule_type = rule.get("type")

            if rule_type == "multivariate":
                self.results[name] = self._validate_multivariate(name, rule)
            elif name not in self.df.columns:
                self.results[name] = {"error": f"Column '{name}' not found"}
            elif rule_type == "numeric":
                self.results[name] = self._validate_numeric(name, rule)
            elif rule_type == "categorical":
                self.results[name] = self._validate_categorical(name, rule)
            elif rule_type == "string":
                self.results[name] = self._validate_string(name, rule)
            else:
                self.results[name] = {"error": f"Unknown type '{rule_type}'"}

        return self.results

    def _detect_outliers(self, data, method, threshold=3.5, lower_pct=None):
        """Run outlier detection using IQR or modified z-score."""
        if method == "iqr":
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            if lower_pct is not None:
                lower = data.quantile(lower_pct / 100.0)

            mask = (data < lower) | (data > upper)
            return {
                "method": "iqr",
                "count": int(mask.sum()),
                "bounds": {"lower": float(lower), "upper": float(upper)},
                "lower_percentile": lower_pct,
                "rows": data[mask].index.tolist(),
                "mask": mask
            }

        elif method == "modified_zscore":
            median = data.median()
            mad = (data - median).abs().median()

            # MAD-based z-score (0.6745 scales MAD to be comparable with std dev)
            if mad == 0:
                z = (data - median).abs()
                mask = z > (threshold * data.std())
                lower = median - threshold * data.std()
                upper = median + threshold * data.std()
            else:
                z = 0.6745 * (data - median) / mad
                mask = z.abs() > threshold
                lower = median - threshold * mad / 0.6745
                upper = median + threshold * mad / 0.6745

            if lower_pct is not None:
                lower = data.quantile(lower_pct / 100.0)
                mask = (data < lower) | (data > upper)

            return {
                "method": "modified_zscore",
                "threshold": threshold,
                "median": float(median),
                "mad": float(mad),
                "count": int(mask.sum()),
                "bounds": {"lower": float(lower), "upper": float(upper)},
                "lower_percentile": lower_pct,
                "rows": data[mask].index.tolist(),
                "mask": mask
            }

        return {"error": f"Unknown method: {method}"}

    def _validate_numeric(self, col, rule):
        result = {"type": "numeric", "invalid_rows": []}
        data = self.df[col]

        min_val, max_val = rule.get("min"), rule.get("max")

        if rule.get("display_scale"):
            result["display_scale"] = rule["display_scale"]
        if rule.get("display_unit"):
            result["display_unit"] = rule["display_unit"]

        if min_val is not None:
            bad = self.df[data < min_val].index.tolist()
            result["below_min"] = {"count": len(bad), "rows": bad[:10]}

        if max_val is not None:
            bad = self.df[data > max_val].index.tolist()
            result["above_max"] = {"count": len(bad), "rows": bad[:10]}

        if rule.get("detect_outliers"):
            method = rule.get("outlier_method", "iqr")
            threshold = rule.get("zscore_threshold", 3.5)
            lower_pct = rule.get("lower_percentile")
            stratify = rule.get("stratify_by")

            if stratify:
                if stratify not in self.df.columns:
                    result["outliers"] = {"error": f"Column '{stratify}' not found"}
                else:
                    result["stratify_by"] = stratify
                    result["outliers_by_group"] = {}
                    all_rows = []

                    for grp, grp_df in self.df.groupby(stratify):
                        grp_data = grp_df[col].dropna()
                        if len(grp_data) < 4:
                            result["outliers_by_group"][grp] = {"error": "Too few points", "count": 0}
                            continue

                        info = self._detect_outliers(grp_data, method, threshold, lower_pct)
                        info["rows"] = info["rows"][:10]
                        info.pop("mask", None)
                        result["outliers_by_group"][grp] = info
                        all_rows.extend(info["rows"])

                    total = sum(g.get("count", 0) for g in result["outliers_by_group"].values())
                    result["outliers"] = {"method": method, "stratified": True, "count": total, "rows": all_rows[:10]}
            else:
                info = self._detect_outliers(data.dropna(), method, threshold, lower_pct)
                info["rows"] = info["rows"][:10]
                info.pop("mask", None)
                result["outliers"] = info

        return result

    def _validate_categorical(self, col, rule):
        result = {"type": "categorical"}
        data = self.df[col]

        allowed = rule.get("allowed", [])
        if allowed:
            bad_mask = ~data.isin(allowed)
            bad_rows = self.df[bad_mask].index.tolist()
            bad_vals = data[bad_mask].unique().tolist()
            result["invalid"] = {"count": len(bad_rows), "values": bad_vals[:10], "rows": bad_rows[:10]}

        result["value_counts"] = data.value_counts().to_dict()
        return result

    def _validate_string(self, col, rule):
        result = {"type": "string"}
        data = self.df[col].astype(str)

        min_len, max_len = rule.get("min_length"), rule.get("max_length")
        pattern = rule.get("pattern")

        if min_len is not None:
            bad = self.df[data.str.len() < min_len].index.tolist()
            result["too_short"] = {"count": len(bad), "rows": bad[:10]}

        if max_len is not None:
            bad = self.df[data.str.len() > max_len].index.tolist()
            result["too_long"] = {"count": len(bad), "rows": bad[:10]}

        if pattern is not None:
            bad = self.df[~data.str.match(pattern, na=False)].index.tolist()
            result["pattern_mismatch"] = {"count": len(bad), "rows": bad[:10]}

        return result

    def _validate_multivariate(self, name, rule):
        """Mahalanobis distance outlier detection for correlated columns."""
        cols = rule.get("columns", [])
        result = {"type": "multivariate", "columns": cols}

        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            result["error"] = f"Columns not found: {missing}"
            return result

        if len(cols) < 2:
            result["error"] = "Need at least 2 columns"
            return result

        data = self.df[cols].dropna()
        if len(data) < len(cols) + 1:
            result["error"] = "Not enough rows"
            return result

        mean = data.mean().values
        cov = data.cov().values

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        diff = data.values - mean
        mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)

        p_val = rule.get("p_value", 0.001)
        thresh = stats.chi2.ppf(1 - p_val, df=len(cols))

        outliers = mahal_sq > thresh
        result["outliers"] = {
            "count": int(outliers.sum()),
            "p_value": p_val,
            "chi2_threshold": float(thresh),
            "rows": data.index[outliers].tolist()[:10]
        }
        result["mahalanobis_distances"] = {
            "max": float(mahal_sq.max()),
            "mean": float(mahal_sq.mean()),
            "median": float(np.median(mahal_sq))
        }
        result["correlations"] = data.corr().round(3).to_dict()

        return result

    def get_invalid_rows(self, name):
        """Get rows that failed validation for a given rule."""
        if name not in self.results:
            return None

        result = self.results[name]
        indices = set()

        if result.get("type") == "multivariate":
            if "outliers" in result and "rows" in result["outliers"]:
                indices.update(result["outliers"]["rows"])
            if indices:
                cols = result.get("columns", [])
                return self.df.loc[list(indices), cols] if cols else self.df.loc[list(indices)]
            return pd.DataFrame()

        for key in ["below_min", "above_max", "outliers", "invalid", "too_short", "too_long", "pattern_mismatch"]:
            if key in result and "rows" in result[key]:
                indices.update(result[key]["rows"])

        return self.df.loc[list(indices)] if indices else pd.DataFrame()


class DataValidator:
    """CSV validation: missing values, duplicates, and column-level rules."""

    def __init__(self, filepath, config_path=None):
        self.filepath = filepath
        self.config_path = config_path
        self.df = None
        self.report = {}
        self.column_validator = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)

    def load_config(self):
        if not self.config_path:
            return None
        with open(self.config_path) as f:
            return json.load(f)

    def check_missing(self):
        missing = self.df.isna().sum()
        missing = missing[missing > 0]
        self.report["missing_values"] = {k: int(v) for k, v in missing.to_dict().items()}

    def check_duplicates(self):
        self.report["duplicate_rows"] = int(self.df.duplicated().sum())

    def validate_columns(self, rules):
        self.column_validator = ColumnValidator(self.df, rules)
        self.report["column_validation"] = self.column_validator.validate()

    def save_report(self, output_dir):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{ts}.json"
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.report, f, indent=4)
        print(f"Report saved as {filename}")

    def run(self, report_path=None):
        self.load_data()
        self.check_missing()
        self.check_duplicates()

        rules = self.load_config()
        if rules:
            self.validate_columns(rules)

        if report_path:
            self.save_report(report_path)

        return self.report


class DataVisualizer:
    """Plotting utilities for data exploration."""

    def __init__(self, df, columns=None):
        self.df = df
        self.columns = columns or df.columns.tolist()

    def plot_missing_heatmap(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isna(), cbar=False, cmap="viridis")
        plt.title("Missing Values")
        plt.show()

    def plot_distribution(self, col):
        if col not in self.df.columns:
            print(f"Column '{col}' not found")
            return
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[col].dropna(), kde=True, bins=20)
        plt.title(f"Distribution: {col}")
        plt.show()

    def plot_correlation_matrix(self):
        numeric = self.df[self.columns].select_dtypes(include='number')
        if numeric.empty:
            print("No numeric columns")
            return
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

    def plot_categorical(self, col):
        if col not in self.df.columns:
            print(f"Column '{col}' not found")
            return
        counts = self.df[col].dropna().value_counts()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f"Distribution: {col}")
        plt.xticks(rotation=45)
        plt.show()

    def plot_outliers(self, col, lower, upper, scale=1, unit=""):
        if col not in self.df.columns:
            print(f"Column '{col}' not found")
            return

        data = self.df[col].dropna() * scale
        lo, hi = lower * scale, upper * scale
        outliers = (data < lo) | (data > hi)

        label = f"{col} ({unit})" if unit else col

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Outliers: {label}", fontsize=14)

        # Box plot
        ax = axes[0, 0]
        sns.boxplot(x=data, ax=ax, color='skyblue')
        ax.axvline(lo, color='red', linestyle='--', label=f'Lower: {lo:,.0f}')
        ax.axvline(hi, color='red', linestyle='--', label=f'Upper: {hi:,.0f}')
        ax.set_title("Box Plot")
        ax.legend()

        # Scatter
        ax = axes[0, 1]
        colors = ['red' if o else 'steelblue' for o in outliers]
        ax.scatter(range(len(data)), data.values, c=colors, alpha=0.5, s=10)
        ax.axhline(lo, color='red', linestyle='--')
        ax.axhline(hi, color='red', linestyle='--')
        ax.set_title("Scatter (red = outliers)")
        ax.set_xlabel("Index")
        ax.set_ylabel(label)

        # Violin
        ax = axes[1, 0]
        sns.violinplot(x=data, ax=ax, color='lightblue', inner=None)
        sns.stripplot(x=data[~outliers], ax=ax, color='steelblue', alpha=0.3, size=3)
        sns.stripplot(x=data[outliers], ax=ax, color='red', alpha=0.8, size=5)
        ax.axvline(lo, color='red', linestyle='--')
        ax.axvline(hi, color='red', linestyle='--')
        ax.set_title("Violin + Strip")

        # Histogram
        ax = axes[1, 1]
        sns.histplot(data, bins=50, ax=ax, color='steelblue', alpha=0.7)
        ax.axvline(lo, color='red', linestyle='--')
        ax.axvline(hi, color='red', linestyle='--')
        ax.axvspan(data.min(), lo, alpha=0.2, color='red')
        ax.axvspan(hi, data.max(), alpha=0.2, color='red')
        ax.set_title("Histogram")

        plt.tight_layout()
        plt.show()

        n_out = outliers.sum()
        suffix = f" {unit}" if unit else ""
        print(f"\nOutlier Summary for '{col}':")
        print(f"  Total: {len(data)}, Outliers: {n_out} ({n_out/len(data)*100:.2f}%)")
        print(f"  Bounds: [{lo:,.0f}, {hi:,.0f}]{suffix}")

    def plot_stratified_outliers(self, col, by_group, strat_col, scale=1, unit=""):
        if col not in self.df.columns or strat_col not in self.df.columns:
            print("Column not found")
            return

        groups = [g for g in by_group if "error" not in by_group[g]]
        if not groups:
            print("No valid groups")
            return

        label = f"{col} ({unit})" if unit else col
        n = len(groups)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        fig.suptitle(f"Stratified Outliers: {label} by {strat_col}", fontsize=14)

        axes = [axes] if n == 1 else axes.flatten()

        for i, grp in enumerate(groups):
            ax = axes[i]
            info = by_group[grp]
            grp_data = self.df[self.df[strat_col] == grp][col].dropna() * scale
            lo, hi = info['bounds']['lower'] * scale, info['bounds']['upper'] * scale

            sns.histplot(grp_data, bins=30, ax=ax, color='steelblue', alpha=0.7)
            ax.axvline(lo, color='red', linestyle='--', linewidth=2)
            ax.axvline(hi, color='red', linestyle='--', linewidth=2)
            ax.axvspan(grp_data.min(), lo, alpha=0.2, color='red')
            ax.axvspan(hi, grp_data.max(), alpha=0.2, color='red')
            ax.set_title(f"{grp}\n{info['count']} outliers [{lo:,.0f}, {hi:,.0f}]", fontsize=10)
            ax.set_xlabel(label)

        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

        total = sum(g.get('count', 0) for g in by_group.values())
        print(f"\nStratified Summary for '{col}' by {strat_col}:")
        print(f"  Total outliers: {total}")
        for grp, info in by_group.items():
            if "error" not in info:
                cnt = self.df[self.df[strat_col] == grp][col].count()
                pct = info['count'] / cnt * 100 if cnt else 0
                print(f"    {grp}: {info['count']} ({pct:.2f}%)")

    def plot_multivariate_outliers(self, cols, outlier_idx, title="Multivariate Outliers"):
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            print(f"Columns not found: {missing}")
            return

        data = self.df[cols].copy()
        data['_outlier'] = self.df.index.isin(outlier_idx)

        # Pairplot
        n = len(cols)
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"{title} (red = outliers)", fontsize=14)

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                ax = fig.add_subplot(n, n, i*n + j + 1)
                if i == j:
                    ax.hist(data.loc[~data['_outlier'], c1], bins=30, alpha=0.7, color='steelblue')
                    ax.hist(data.loc[data['_outlier'], c1], bins=30, alpha=0.7, color='red')
                else:
                    ax.scatter(data.loc[~data['_outlier'], c2], data.loc[~data['_outlier'], c1],
                              alpha=0.3, s=5, c='steelblue')
                    ax.scatter(data.loc[data['_outlier'], c2], data.loc[data['_outlier'], c1],
                              alpha=0.7, s=15, c='red')
                if j == 0:
                    ax.set_ylabel(c1[:12], fontsize=8)
                if i == n-1:
                    ax.set_xlabel(c2[:12], fontsize=8)
                ax.tick_params(labelsize=6)

        plt.tight_layout()
        plt.show()

        # Parallel coordinates
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"{title} - Parallel Coordinates", fontsize=14)

        normed = data[cols].copy()
        for c in cols:
            cmin, cmax = normed[c].min(), normed[c].max()
            if cmax > cmin:
                normed[c] = (normed[c] - cmin) / (cmax - cmin)

        normal = normed[~data['_outlier']]
        if len(normal) > 500:
            normal = normal.sample(500, random_state=42)

        for idx in normal.index:
            ax.plot(range(n), normal.loc[idx, cols].values, color='steelblue', alpha=0.1, linewidth=0.5)

        for idx in normed[data['_outlier']].index:
            ax.plot(range(n), normed.loc[idx, cols].values, color='red', alpha=0.5, linewidth=1)

        ax.set_xticks(range(n))
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_ylabel('Normalized')

        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color='steelblue', alpha=0.5, label='Normal'),
            Line2D([0], [0], color='red', alpha=0.8, label='Outlier')
        ])

        plt.tight_layout()
        plt.show()

        n_out = data['_outlier'].sum()
        print(f"\nMultivariate Summary: {len(data)} rows, {n_out} outliers ({n_out/len(data)*100:.2f}%)")


def print_results(results, df, validator=None, viz=None):
    """Print validation results to console."""
    print("\n--- Validation Results ---")

    for col, res in results.items():
        print(f"\n{col}:")

        if "error" in res:
            print(f"  Error: {res['error']}")
            continue

        if res["type"] == "numeric":
            scale = res.get("display_scale", 1)
            unit = res.get("display_unit", "")
            suffix = f" {unit}" if unit else ""

            if "below_min" in res:
                print(f"  Below min: {res['below_min']['count']}")
            if "above_max" in res:
                print(f"  Above max: {res['above_max']['count']}")

            if "outliers" in res:
                info = res['outliers']

                if res.get("stratify_by"):
                    strat = res["stratify_by"]
                    print(f"  Stratified by: {strat}")
                    print(f"  Total outliers: {info['count']}")
                    print(f"\n  By group:")

                    for grp, grp_info in res.get("outliers_by_group", {}).items():
                        if "error" in grp_info:
                            print(f"    {grp}: {grp_info['error']}")
                        else:
                            lo = grp_info['bounds']['lower'] * scale
                            hi = grp_info['bounds']['upper'] * scale
                            print(f"    {grp}: {grp_info['count']} outliers ({lo:,.0f} - {hi:,.0f}{suffix})")

                    if info['count'] > 0 and validator:
                        rows = validator.get_invalid_rows(col).copy()
                        print(f"\n  Sample outliers:")
                        show = [strat] + [c for c in df.columns[:2] if c not in [col, strat]] + [col]
                        show = [c for c in show if c in rows.columns]
                        if scale != 1:
                            rows[col] = rows[col] * scale
                        print(rows[show].head(10).sort_values(col, ascending=False).to_string(index=False))

                        if viz:
                            viz.plot_stratified_outliers(col, res["outliers_by_group"], strat, scale, unit)
                else:
                    lo = info['bounds']['lower'] * scale
                    hi = info['bounds']['upper'] * scale
                    print(f"  Outliers: {info['count']} ({lo:,.0f} - {hi:,.0f}{suffix})")

                    if info['count'] > 0 and validator:
                        rows = validator.get_invalid_rows(col).copy()
                        print(f"\n  Sample outliers:")
                        show = [c for c in df.columns[:3] if c != col] + [col]
                        show = [c for c in show if c in rows.columns]
                        if scale != 1:
                            rows[col] = rows[col] * scale
                        print(rows[show].head(10).sort_values(col, ascending=False).to_string(index=False))

                        if viz:
                            viz.plot_outliers(col, info['bounds']['lower'], info['bounds']['upper'], scale, unit)

        elif res["type"] == "categorical":
            if "invalid" in res and res['invalid']['count'] > 0:
                print(f"  Invalid: {res['invalid']['count']} - {res['invalid']['values']}")
            else:
                print("  All valid")

        elif res["type"] == "string":
            if "too_short" in res:
                print(f"  Too short: {res['too_short']['count']}")
            if "too_long" in res:
                print(f"  Too long: {res['too_long']['count']}")
            if "pattern_mismatch" in res:
                print(f"  Pattern mismatch: {res['pattern_mismatch']['count']}")

        elif res["type"] == "multivariate":
            cols = res.get("columns", [])
            print(f"  Columns: {cols}")

            info = res.get("outliers", {})
            if info:
                print(f"  Outliers: {info['count']} (p={info['p_value']}, threshold={info['chi2_threshold']:.2f})")

            if "mahalanobis_distances" in res:
                d = res["mahalanobis_distances"]
                print(f"  Mahalanobis: mean={d['mean']:.2f}, median={d['median']:.2f}, max={d['max']:.2f}")

            if "correlations" in res:
                print("  Correlations:")
                corr = res["correlations"]
                for c in cols:
                    vals = [f"{corr[c].get(c2, 0):.2f}" for c2 in cols]
                    print(f"    {c}: {vals}")

            if info.get('count', 0) > 0 and validator:
                rows = validator.get_invalid_rows(col)
                print(f"\n  Sample outliers:")
                print(rows.head(10).to_string(index=False))

                if viz:
                    viz.plot_multivariate_outliers(cols, info.get('rows', []), f"Multivariate: {col}")


def main():
    parser = argparse.ArgumentParser(description="CSV data validator")
    parser.add_argument("-f", "--file", required=True, help="CSV file path")
    parser.add_argument("-r", "--report", default=".", help="Report output directory")
    parser.add_argument("-c", "--config", help="JSON config file")
    parser.add_argument("--num_columns", help="Numeric columns to plot (comma-separated)")
    parser.add_argument("--cat_columns", help="Categorical columns to plot (comma-separated)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plots")
    args = parser.parse_args()

    validator = DataValidator(args.file, config_path=args.config)
    report = validator.run(args.report)

    print(f"\nMissing: {report.get('missing_values', {})}")
    print(f"Duplicates: {report.get('duplicate_rows', 0)}")

    viz = None
    if not args.no_plots:
        viz = DataVisualizer(validator.df)
        viz.plot_missing_heatmap()
        viz.plot_correlation_matrix()

        if args.num_columns:
            for c in args.num_columns.split(","):
                viz.plot_distribution(c.strip())

        if args.cat_columns:
            for c in args.cat_columns.split(","):
                viz.plot_categorical(c.strip())

    if "column_validation" in report:
        print_results(report["column_validation"], validator.df, validator.column_validator, viz)


if __name__ == "__main__":
    main()
