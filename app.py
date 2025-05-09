# EDA visualization
import openai
from openai import OpenAI
from flask import session
import hashlib
import traceback
import textwrap
import pandas as pd
import base64
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import re  
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
from jinja2 import Undefined

# Plots
import matplotlib.pyplot as plt
import json
import plotly.express as px
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
from sklearn.tree import export_text
from dtreeviz import dtreeviz
from dtreeviz import model

# Metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Snorkel
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import MajorityLabelVoter
from sklearn.metrics import accuracy_score

# EDA visualization
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go

# app/routes_typologies.py
from flask import Blueprint, jsonify
import inspect
import os
from dotenv import load_dotenv

ABSTAIN = -1
POSITIVE = 1
NEGATIVE = 0

load_dotenv()  # Load environment variables 

latest_matched_indices = [] # Global store for latest rule match results
latest_snorkel_metrics = {}
cached_importances = None

# Falsk initiation
app = Flask(__name__)
MODEL = "gpt-4o-mini"

# Load your datasets
df = pd.read_csv("train_sample.csv")
df = df.loc[:, ~df.columns.duplicated()].copy()
test_df = pd.read_csv("test_sample.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["DayOfWeek"] = df["Timestamp"].dt.day_name()
df["HourOfDay"] = df["Timestamp"].dt.hour

# Initialize label function library
label_function_library = []

# Ensure Match Type column
if "Match Type" not in df.columns:
    df["Match Type"] = "Unknown"


def readable_bin_label(interval):
    if pd.isna(interval):
        return "NA"
    if np.isinf(interval.right):
        return f">{int(interval.left / 1000)}k"
    left = int(interval.left)
    right = int(interval.right)
    if right <= 1000:
        return "0–1k"
    elif right < 1_000_000:
        return f"{left // 1000}k–{right // 1000}k"
    else:
        return f"{left // 1_000_000}M–{right // 1_000_000}M"
# --- Helper Functions for Typologies  ---
import sys
sys.path.append('label_functions')



def sanitize_and_validate_python(code_str):
    """Clean and verify a code string is a valid Python label_function."""
    try:
        # Remove backslash-escaped characters like \\n and \\t
        code_str = code_str.encode().decode('unicode_escape')

        # Ensure it's a function definition starting with def label_function
        if not code_str.strip().startswith("def label_function"):
            raise SyntaxError("Code must start with 'def label_function'")

        # Optional: Strip trailing whitespace
        code_str = re.sub(r"[ \t]+(\n|$)", r"\1", code_str)

        # Attempt to parse into AST to check validity
        ast.parse(code_str)

        return code_str  # Return cleaned-up version if all checks pass
    except Exception as e:
        raise SyntaxError(f"Sanitization failed: {str(e)}")


def apply_rule(df, rule_func):
    results = []
    for _, row in df.iterrows():
        try:
            label = rule_func(row)
        except:
            label = -1
        results.append(label)
    return pd.Series(results, index=df.index)  # Ensure proper alignment

def evaluate_rules(df):
    summary = []
    for name, fn in RULES.items():
        print(f"Evaluating {name}...")
        labels = apply_rule(df, fn)
        ground_truth = df["Is Laundering"].tolist()

        tp = sum((l == 1 and gt == 1) for l, gt in zip(labels, ground_truth))
        fp = sum((l == 1 and gt == 0) for l, gt in zip(labels, ground_truth))
        fn = sum((l == 0 and gt == 1) for l, gt in zip(labels, ground_truth))
        tn = sum((l == 0 and gt == 0) for l, gt in zip(labels, ground_truth))
        covered = sum(l != -1 for l in labels)

        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-5)
        coverage = covered / len(df)

        summary.append({
            "name": name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "coverage": round(coverage, 4),
        })

    return pd.DataFrame(summary)

@app.route("/apply-rule", methods=["POST"])
def apply_rule():
    try:
        code = request.form.get("label_function")
        name = request.form.get("name", "Manual Rule")

        exec_globals = {
            "get_transactions_to": get_transactions_to,
            "get_incoming_count": get_incoming_count,
            "sum_incoming": sum_incoming,
            "get_outgoing_count": get_outgoing_count,
            "get_distinct_receivers": get_distinct_receivers,
            "get_distinct_senders": get_distinct_senders,
            "has_transaction": has_transaction,
            "get_transactions_from": get_transactions_from,
            "sum_outgoing": sum_outgoing,
            "time_diff": time_diff,
            "df": test_df,
            "test_df": test_df,
            "ABSTAIN": -1,
            "POSITIVE": 1,
            "NEGATIVE": 0
        }

        exec(code, exec_globals)
        func = exec_globals.get("label_function")

        if not func:
            raise ValueError("Function not defined")

        def safe_apply(row):
            try:
                return int(func(row, test_df))  # Ensure it gets both row and df
            except:
                return -1

        test_df["Human Label"] = test_df.apply(safe_apply, axis=1)

        return jsonify({"success": True})
    except Exception as e:
        print("❌ Failed to apply rule:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/get-columns")
def get_columns():
    cols = [col for col in test_df.columns if col not in ["Human Label", "Match Type", "_bin"]]
    return jsonify(sorted(cols))


@app.route("/distinct/<column>")
def distinct_column_values(column):
    if column in test_df.columns:
        values = test_df[column].dropna().unique().tolist()
        return jsonify(sorted(values))
    else:
        return jsonify([])


# Rename for clarity
# utils/helpers.py
import pandas as pd
from datetime import timedelta

SENDER = "Account"
RECEIVER = "Account.1"

def get_outgoing_count(account, df, window_days=1, current_time=None):
    df = df[df[SENDER].eq(account)]
    if current_time:
        start = current_time - timedelta(days=window_days)
        df = df[df[TIME] >= start]
    return len(df)

def get_incoming_count(account, df, window_days=1, current_time=None):
    df = df[df[RECEIVER].eq(account)]
    if current_time:
        start = current_time - timedelta(days=window_days)
        df = df[df[TIME] >= start]
    return len(df)

def get_distinct_receivers(account, df, window_days=1, current_time=None):
    df = df[df[SENDER].eq(account)]
    if current_time:
        start = current_time - timedelta(days=window_days)
        df = df[df[TIME] >= start]
    return df[RECEIVER].unique().tolist()

def get_distinct_senders(account, df, window_days=1, current_time=None):
    df = df[df[RECEIVER].eq(account)]
    if current_time:
        start = current_time - timedelta(days=window_days)
        df = df[df[TIME] >= start]
    return df[SENDER].unique().tolist()

def has_transaction(sender, receiver, df):
    return not df[df[SENDER].eq(sender) & df[RECEIVER].eq(receiver)].empty

def get_transactions_from(account, df):
    return df[df[SENDER].eq(account)].to_dict("records")

def get_transactions_to(account, df):
    return df[df[RECEIVER].eq(account)].to_dict("records")

def sum_incoming(account, df, window_days=1, current_time=None):
    df = df[df[RECEIVER].eq(account)]
    if current_time:
        start = current_time - timedelta(days=window_days)
        df = df[df[TIME] >= start]
    return df[RECEIVED].sum()

def sum_outgoing(account, df, window_days=1, current_time=None):
    df = df[df[SENDER].eq(account)]
    if current_time:
        start = current_time - timedelta(days=window_days)
        df = df[df[TIME] >= start]
    return df[PAID].sum()

def time_diff(account, direction, df):
    if direction == "in_to_out":
        in_times = df[df[RECEIVER].eq(account)][TIME].sort_values()
        out_times = df[df[SENDER].eq(account)][TIME].sort_values()
        if not in_times.empty and not out_times.empty:
            return (out_times.iloc[0] - in_times.iloc[-1]).total_seconds()
    return float("inf")

# --- Helper Functions for Calculation ---
def get_column_distributions(df):
    plots = []
    for col in df.columns:
        try:
            if df[col].dtype == 'object' or df[col].nunique() < 30:
                # Categorical
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, "Count"]
                fig = px.bar(vc, x=col, y="Count", title=col)
            else:
                # Numeric
                fig = px.histogram(df, x=col, nbins=40, title=col)
            fig_json = pio.to_json(fig)
            plots.append({"name": col, "figJSON": fig_json})
        except Exception as e:
            print(f"⚠️ Skipping column {col} due to error:", e)
    return plots
    
def apply_label_function(code, df):
    try:
        exec_globals = {}
        exec(code, {}, exec_globals)  # Execute the code in a fresh global scope
        func = exec_globals.get("label_function")  # Get the label function

        if func:
            # Apply the label function to each row of the DataFrame
            return [idx for idx, row in df.iterrows() if func(row) == 1]
        else:
            print("❌ label_function not found")
            return []
    except Exception as e:
        print(f"Function error: {e}")
        return []


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": f"{accuracy_score(y_true, y_pred):.2%}",
        "precision": f"{precision_score(y_true, y_pred, zero_division=0):.2%}",
        "recall": f"{recall_score(y_true, y_pred, zero_division=0):.2%}",
        "f1_score": f"{f1_score(y_true, y_pred, zero_division=0):.2%}"
    }

def compute_coverage(y_pred):
    return f"{(y_pred != ABSTAIN).sum() / len(y_pred):.2%}"

def compute_conflict_rate(df, label_functions):
    """
    Computes % of rows flagged by multiple label functions (i.e., conflict).
    """
    total_conflicts = 0

    for idx, row in df.iterrows():
        active = 0
        for lf in label_functions:
            try:
                exec_globals = {}
                exec(lf["code"], {}, exec_globals)
                func = exec_globals.get("label_function")
                if func:
                    # Ensure that both row and df are passed
                    if func(row, df) == 1:  # Pass both row and the DataFrame
                        active += 1
            except Exception:
                continue
        if active > 1:
            total_conflicts += 1

    return round(total_conflicts / len(df), 4) if len(df) else 0.0

def create_snorkel_metrics_plot(metrics):
    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(metrics.keys()),
        y=[float(m.strip('%')) for m in metrics.values()],
        text=list(metrics.values()),
        textposition='auto',
        marker_color='rgba(100, 200, 102, 0.7)'
    ))
    fig.update_layout(title="Snorkel LabelModel Evaluation", yaxis_title="Percentage")
    return fig.to_html(full_html=False)

def apply_lfs_to_data(snorkel_lfs, df):
    return PandasLFApplier(snorkel_lfs).apply(df=df)

def snorkel_predict(lfs, df):
    applier = PandasLFApplier(lfs)
    L = applier.apply(df=df)
    model = LabelModel(cardinality=2, verbose=True)
    model.fit(L_train=L, n_epochs=500, log_freq=100, seed=42)
    preds = model.predict(L=L)
    return preds, L

    
# --- Add a function to apply all LFs to all rows---
def apply_all_label_functions(df, library):
    label_matrix = []

    for idx, row in df.iterrows():
        row_votes = []
        for lf in library:
            try:
                exec_globals = {}
                exec(lf["code"], {}, exec_globals)
                func = exec_globals.get("label_function")
                result = func(row)
                if result == 1:
                    row_votes.append(1)
                elif result == 0:
                    row_votes.append(0)
                else:
                    row_votes.append(None)
            except:
                row_votes.append(None)
        label_matrix.append(row_votes)

    return label_matrix

from inspect import signature
from snorkel.labeling import labeling_function
ABSTAIN = -1
POSITIVE = 1
NEGATIVE = 0

def convert_to_snorkel_lfs(library, df):
    snorkel_lfs = []
    valid_count = 0

    for lf in library:
        try:
            # Setup execution environment
            exec_globals = {
                "pd": pd,
                "np": np,
                "ABSTAIN": ABSTAIN,
                "POSITIVE": POSITIVE,
                "NEGATIVE": NEGATIVE,
                "get_outgoing_count": get_outgoing_count,
                "get_distinct_receivers": get_distinct_receivers,
                "get_incoming_count": get_incoming_count,
                "get_distinct_senders": get_distinct_senders,
                "has_transaction": has_transaction,
                "sum_incoming": sum_incoming,
                "sum_outgoing": sum_outgoing,
                "get_transactions_from": get_transactions_from,
                "get_transactions_to": get_transactions_to,
                "time_diff": time_diff
            }

            code = lf["code"]
            exec(code, exec_globals)
            func = exec_globals.get("label_function") or next(
                (v for k, v in exec_globals.items() if callable(v)), None
            )

            if not func:
                print(f"❌ Skipping '{lf['name']}': No callable function found")
                continue

            # Inspect function signature (1 or 2 args)
            from inspect import signature
            sig = signature(func)
            param_count = len(sig.parameters)

            if param_count not in [1, 2]:
                print(f"❌ Skipping '{lf['name']}': function must take 1 or 2 parameters, got {param_count}")
                continue

            @labeling_function(name=lf["name"])
            def wrapped(x, f=func):
                try:
                    if param_count == 1:
                        return POSITIVE if f(x) == 1 else NEGATIVE if f(x) == 0 else ABSTAIN
                    else:
                        return POSITIVE if f(x, df) == 1 else NEGATIVE if f(x, df) == 0 else ABSTAIN
                except Exception as e:
                    print(f"⚠️ '{lf['name']}' failed on row: {e}")
                    return ABSTAIN

            snorkel_lfs.append(wrapped)
            valid_count += 1

        except Exception as e:
            print(f"❌ Error parsing LF '{lf.get('name', 'Unnamed')}': {e}")

    print(f"✅ {valid_count} label function(s) compiled into Snorkel format")
    return snorkel_lfs


def majority_vote(label_matrix):
    """
    Takes a label matrix (n_samples x n_LFs) and returns final predicted labels.
    Ignores None and uses majority rule.
    """
    final_labels = []
    for row in label_matrix:
        votes = [label for label in row if label is not None]
        if not votes:
            final_labels.append(0)  # or None
        else:
            final_labels.append(int(sum(votes) >= len(votes)/2))
    return final_labels


# --- Graph in index page---
def compute_evaluation_metrics(matched_indices, df):
    y_true = df["Is Laundering"].values
    y_pred = np.zeros(len(df))
    for idx in matched_indices:
        if idx < len(y_pred):
            y_pred[idx] = 1

    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    coverage = compute_coverage_rate(matched_indices, df)

    return {
        "accuracy": f"{accuracy:.2%}",
        "precision": f"{precision:.2%}",
        "recall": f"{recall:.2%}",
        "f1": f"{f1:.2%}",
        "coverage": f"{coverage:.2%}"
    }



def create_label_distribution_graph(df, label_col="Is Laundering"):
    """
    Creates a bar chart displaying the distribution of labels.
    
    Parameters:
    - df: DataFrame containing the data
    - label_col: The column representing the labels (default is "Is Laundering")
    
    Returns:
    - A base64 encoded PNG image for embedding in HTML
    """
    try:
        # Count the occurrences of each label (e.g., 0 for legitimate, 1 for laundering)
        label_counts = df[label_col].value_counts().sort_index()

        # Prepare the labels and values for the chart
        labels = ["Legitimate", "Laundering"]
        values = [label_counts.get(0, 0), label_counts.get(1, 0)]  # Get counts for legitimate (0) and laundering (1)

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(6, 4))  # Adjust size to fit the layout
        ax.bar(labels, values, color=['blue', 'red'])

        # Adding title and labels
        ax.set_title(f"Label Distribution: {label_col}", fontsize=16)
        ax.set_xlabel("Label", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_ylim(0, max(values) * 1.1)  # Add a little padding to the top

        # Convert the plot to a PNG image and encode it in base64
        img_stream = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_stream, format='png')
        plt.close(fig)

        img_stream.seek(0)
        img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')
        
        # Return the base64-encoded image string
        return img_base64

    except Exception as e:
        print(f"Error generating label distribution graph: {str(e)}")
        return ""


def annotate_match_type(df, matched):
    df = df.copy()
    df["Match Type"] = "True Negative"
    for idx in df.index:
        actual = df.at[idx, "Is Laundering"]
        predicted = 1 if idx in matched else 0
        if predicted == 1 and actual == 1:
            df.at[idx, "Match Type"] = "True Positive"
        elif predicted == 1 and actual == 0:
            df.at[idx, "Match Type"] = "False Positive"
        elif predicted == 0 and actual == 1:
            df.at[idx, "Match Type"] = "False Negative"
    return df


def extract_label_function_block(raw):
    # Remove markdown backticks if present
    raw = re.sub(r"```(python)?", "", raw)
    raw = re.sub(r"```", "", raw)

    # Extract only the function (everything starting from `def label_function`)
    match = re.search(r"(def label_function\(.*?:\n(?:\s+.+\n?)+)", raw)
    if match:
        return match.group(1).strip()
    return ""
    
# --- Coverage performance calculation ---
def compute_coverage_rate(matched_indices, df):
    """
    Compute coverage rate = fraction of positive cases captured by the label function.
    """
    actual_positives = set(df[df["Is Laundering"] == 1].index)
    matched_set = set(matched_indices)

    if not actual_positives:
        return 0.0

    true_positives = matched_set.intersection(actual_positives)
    coverage_rate = len(true_positives) / len(actual_positives)
    return coverage_rate

def compute_accuracy(matched_indices, df):
    y_true = df["Is Laundering"].values
    y_pred = np.zeros(len(df))
    for idx in matched_indices:
        if idx < len(y_pred):
            y_pred[idx] = 1
    return (y_true == y_pred).sum() / len(y_true)

def compute_precision_recall_f1(matched_indices, df):
    y_true = df["Is Laundering"].values
    y_pred = np.zeros(len(df))
    for idx in matched_indices:
        if idx < len(y_pred):
            y_pred[idx] = 1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1

def generate_confusion_bar_chart(tp, fp, tn, fn):
    """
    Generate a horizontal bar chart showing counts of TP, FP, TN, FN.
    Returns base64-encoded PNG string for HTML embedding.
    """
    labels = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
    values = [tp, fp, tn, fn]
    colors = ['green', 'orange', 'gray', 'red']

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Count")
    ax.set_title("Label Match Type Breakdown")
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()

# --- Routes ---
DISPLAY_ROWS = 20
SCATTER_ROWS = 1000

compiled_functions = {}
import io
def compute_feature_graphs(df):
    feature_graphs = []

    df_copy = df.copy(deep=True)  # ✅ Only copy once
    if "Human Label" not in df_copy.columns or df_copy["Human Label"].isnull().all():
        print("⚠️ Warning: Human Label is missing or empty!")

    for col in cached_importances:
        try:
            col_data = df_copy[col]

            # Define bins and labels
            if pd.api.types.is_numeric_dtype(col_data) and col in ["Amount Received", "Amount Paid"]:
                bins = [0, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 50000000, float("inf")]
                labels = [f"{int(a//1000)}k–{int(b//1000)}k" if b != float("inf") else f">{int(a//1000)}k"
                          for a, b in zip(bins[:-1], bins[1:])]
                df_copy["_bin"] = pd.cut(col_data, bins=bins, right=False, labels=labels)
            elif col == "HourOfDay":
                labels = ["Late Night (0–5)", "Early (5–8)", "Morning (8–12)", "Afternoon (12–17)", "Evening (17–21)", "Night (21–24)"]
                df_copy["_bin"] = pd.cut(df_copy[col], bins=[-1, 5, 8, 12, 17, 21, 24], right=False, labels=labels)
            elif pd.api.types.is_numeric_dtype(col_data):
                try:
                    binned = pd.qcut(col_data, q=5, duplicates='drop')
                    labels = [str(interval) for interval in binned.cat.categories]
                    df_copy["_bin"] = binned.astype(str)
                except:
                    df_copy["_bin"] = "Unknown"
                    labels = ["Unknown"]
            else:
                df_copy["_bin"] = col_data.astype(str)
                labels = sorted(df_copy["_bin"].dropna().unique().tolist())

            # Compute before/after counts
            before = df_copy.groupby(["_bin", "Is Laundering"], observed=False).size().unstack(fill_value=0)
            after = df_copy.groupby(["_bin", "Human Label"], observed=False).size().unstack(fill_value=0)

            # Compute total shifts (✅ keep only these)
            total_legit_before = before.get(0, pd.Series()).sum()
            total_legit_after = after.get(0, pd.Series()).sum()
            total_laundering_before = before.get(1, pd.Series()).sum()
            total_laundering_after = after.get(1, pd.Series()).sum()

            delta_legit = round((total_legit_after - total_legit_before) / (total_legit_before + 1e-5) * 100, 1)
            delta_laundering = round((total_laundering_after - total_laundering_before) / (total_laundering_before + 1e-5) * 100, 1)

            def count_or_zero(gb, l, val):
                return int(gb.loc[l].get(val, 0)) if l in gb.index else 0

            legit_before = [count_or_zero(before, l, 0) for l in labels]
            laundering_before = [count_or_zero(before, l, 1) for l in labels]
            legit_after = [count_or_zero(after, l, 0) for l in labels]
            laundering_after = [count_or_zero(after, l, 1) for l in labels]

            feature_graphs.append({
                "feature": col,
                "importance": float(cached_importances[col]),
                "labels": [str(l) for l in labels],
                "legit": legit_before,
                "laundering": laundering_before,
                "after": {
                    "labels": [str(l) for l in labels],
                    "legit": legit_after,
                    "laundering": laundering_after
                },
                "delta_legit": delta_legit,
                "delta_laundering": delta_laundering,
                "before_legit": int(sum(legit_before)),
                "before_laundering": int(sum(laundering_before)),
                "after_legit": int(sum(legit_after)),
                "after_laundering": int(sum(laundering_after))
            })

        except Exception as e:
            print(f"⚠️ Feature error [{col}]:", e)

    return sorted(feature_graphs, key=lambda x: x["importance"], reverse=True)



@app.route("/get-feature-comparison")
def get_feature_comparison():
    try:
        features = compute_feature_graphs(test_df)
        return jsonify(features)  # Should now be safe
    except Exception as e:
        print(f"❌ Feature comparison error: {e}")
        return jsonify([]), 500


@app.route("/", methods=["GET", "POST"])
def index():
    global cached_importances

    # Ensure "Human Label" exists
    if "Human Label" not in test_df.columns:
        test_df["Human Label"] = -1

    # Ensure time-derived features
    if "HourOfDay" not in test_df or "DayOfWeek" not in test_df:
        test_df["Timestamp"] = pd.to_datetime(test_df["Timestamp"], errors="coerce")
        test_df["HourOfDay"] = test_df["Timestamp"].dt.hour
        test_df["DayOfWeek"] = test_df["Timestamp"].dt.day_name()

    # Default feature importances if not loaded
    if cached_importances is None:
        cached_importances = {
            "Amount Received": 10.5,
            "Amount Paid": 8.93,
            "Payment Currency": 1.74,
            "Receiving Currency": 1.72,
            "Payment Format": 47.48,
            "From Bank": 4.97,
            "To Bank": 4.63,
            "Account": 7.01,
            "Account.1": 5.8,
            "HourOfDay": 4.41,
            "DayOfWeek": 2.82
        }

    # If POST, apply a new label function
    code = ""
    name = "Manual Rule"
    if request.method == "POST":
        code = request.form.get("label_function", "")
        name = request.form.get("name", "Manual Rule")

        try:
            exec_globals = {}
            exec(code, {}, exec_globals)
            func = exec_globals.get("label_function")
            if not func:
                raise ValueError("Function `label_function` not defined.")

            def safe_apply(row):
                try:
                    return int(func(row, test_df))
                except:
                    return -1

            test_df["Human Label"] = test_df.apply(safe_apply, axis=1)

        except Exception as e:
            print("❌ Failed to apply rule:", e)
            test_df["Human Label"] = -1  # fallback if it breaks

    # Compute evaluation metrics
    y_true = test_df["Is Laundering"].values
    y_pred = test_df["Human Label"].values
    mask = y_pred != -1

    if mask.any():
        evaluation = {
            "accuracy": f"{accuracy_score(y_true[mask], y_pred[mask]):.2%}",
            "precision": f"{precision_score(y_true[mask], y_pred[mask], zero_division=0):.2%}",
            "recall": f"{recall_score(y_true[mask], y_pred[mask], zero_division=0):.2%}",
            "f1": f"{f1_score(y_true[mask], y_pred[mask], zero_division=0):.2%}",
            "coverage": f"{mask.mean():.2%}"
        }
    else:
        evaluation = {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "f1": "N/A",
            "coverage": "0.00%"
        }

    # Generate feature comparison graphs using new function
    feature_graphs = compute_feature_graphs(test_df)

    return render_template("index.html",
                           evaluation=evaluation,
                           feature_graphs=feature_graphs,
                           label_function_code=code,
                           label_name=name)



# ✅ Updated index() route with correct indentation for safe_execute()
from flask import request, jsonify

from flask import request, jsonify

def try_cast(val, typ):
    try:
        if val is None:
            return None
        return typ(val)
    except (TypeError, ValueError):
        print(f"⚠️ Failed to cast: {val} to {typ}")
        return None

@app.route("/generate-rule-code", methods=["POST"])
def generate_rule_code():
    data = request.get_json(force=True)
    print("📦 Raw request data:", data)

    conditions = data.get("conditions", [])
    label = try_cast(data.get("label"), int)

    # Typology values
    min_tx = try_cast(data.get("min_transactions"), int)
    max_amt = try_cast(data.get("max_individual_amount"), float)
    total_amt = try_cast(data.get("total_threshold"), float)
    unique_senders = try_cast(data.get("min_unique_senders"), int)
    unique_recipients = try_cast(data.get("min_unique_recipients"), int)

    print("🔍 Parsed values:", {
        "conditions": conditions,
        "min_tx": min_tx,
        "max_amt": max_amt,
        "total_amt": total_amt,
        "unique_senders": unique_senders,
        "unique_recipients": unique_recipients
    })

    if not conditions and all(v is None for v in [min_tx, max_amt, total_amt, unique_senders, unique_recipients]):
        return jsonify({
            "code": "def label_function(row, df):\n    return -1  # no conditions defined"
        })

    lines = ["def label_function(row, df):"]

    # --- Structuring Pattern ---
    if all(v is not None for v in [min_tx, max_amt, total_amt]):
        lines.append("    recipient = row['Account.1']")
        lines.append("    transactions = df[df['Account.1'] == recipient]")
        lines.append(f"    small_deposits = transactions[transactions['Amount Received'] <= {max_amt}]")
        lines.append(f"    if len(small_deposits) < {min_tx}: return 0")
        lines.append(f"    if small_deposits['Amount Received'].sum() <= {total_amt}: return 0")

    # --- Fan-In Pattern ---
    if unique_senders is not None:
        lines.append("    recipient = row['Account.1']")
        lines.append("    incoming = df[df['Account.1'] == recipient]")
        lines.append("    senders = incoming['Account'].nunique()")
        lines.append(f"    if senders < {unique_senders}: return 0")

    # --- Fan-Out Pattern ---
    if unique_recipients is not None:
        lines.append("    sender = row['Account']")
        lines.append("    outgoing = df[df['Account'] == sender]")
        lines.append("    receivers = outgoing['Account.1'].nunique()")
        lines.append(f"    if receivers < {unique_recipients}: return 0")

    # --- Manual Conditions ---
    # --- Manual Conditions ---
    lines.append("    # Manual conditions")
    for cond in conditions:
        field = cond.get("field")
        operator = cond.get("operator")
        value = cond.get("value")

        if not field or not operator or value is None:
            continue

        if operator.lower() == "between":
            # Value should be two numbers separated by a dash or comma
            parts = [v.strip() for v in str(value).replace(",", "-").split("-")]
            if len(parts) == 2:
                try:
                    low = float(parts[0])
                    high = float(parts[1])
                    lines.append(f"    if not ({low} <= row['{field}'] <= {high}): return 0")
                except ValueError:
                    lines.append(f"    # Skipped invalid between condition: {value}")
            else:
                lines.append(f"    # Skipped malformed between value: {value}")
            continue

        # Handle normal comparisons
        op_map = {
            "is": "==",
            "is not": "!=",
            "greater than": ">",
            "less than": "<"
        }
        py_op = op_map.get(operator.lower(), "==")

        # Quote strings if value is not a number
        try:
            float_val = float(value)
            formatted_value = value
        except ValueError:
            formatted_value = f"'{value}'"

        lines.append(f"    if not (row['{field}'] {py_op} {formatted_value}): return 0")


    # --- Final output ---
    lines.append(f"    return {label}")
    return jsonify({"code": "\n".join(lines)})


# Function to get counts for before and after
def get_counts(gb, labels):
    legit, laundering = [], []
    for label in labels:
        row = gb.loc[label] if label in gb.index else {}
        legit.append(int(row.get(0, 0)))
        laundering.append(int(row.get(1, 0)))
    return legit, laundering


def evaluate_label_function(func, df):
    """
    Applies a label function and evaluates it against 'Is Laundering'.
    Returns metrics dict and matched indices.
    """
    try:
        matched = df.apply(lambda row: func(row, df), axis=1)
        matched = matched.reindex(df.index).fillna(0).astype(int)
        y_pred = matched.values
        y_true = df["Is Laundering"].values

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        coverage = (y_pred == 1).sum() / len(y_pred)

        return {
            "accuracy": f"{accuracy:.2%}",
            "precision": f"{precision:.2%}",
            "recall": f"{recall:.2%}",
            "f1_score": f"{f1:.2%}",
            "coverage": f"{coverage:.2%}",
            "matched_indices": df.index[y_pred == 1].tolist()
        }

    except Exception as e:
        print("❌ Evaluation error:", e)
        return {
            "accuracy": "0.00%", "precision": "0.00%", "recall": "0.00%",
            "f1_score": "0.00%", "coverage": "0.00%", "matched_indices": []
        }

@app.route("/apply-rule", methods=["POST"])
def apply_manual_rule():
    rule_code = request.form.get("label_function", "")
    if not rule_code:
        return jsonify({"error": "No rule code provided"}), 400

    try:
        local_env = {
            "get_transactions_to": get_transactions_to,
            "get_incoming_count": get_incoming_count,
            "sum_incoming": sum_incoming,
            # Add others if needed
        }
        exec(rule_code, local_env)
        rule_func = local_env["label_function"]
        test_df["Human Label"] = apply_rule(test_df, rule_func)  # ✅ <- key fix here
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-evaluation")
def get_evaluation():
    try:
        if "Human Label" not in test_df:
            return jsonify({
                "accuracy": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
                "coverage": "0.00%"
            })

        y_pred = test_df["Human Label"].fillna(-1).astype(int).values
        y_true = test_df["Is Laundering"].values

        # Only evaluate on labeled examples
        mask = y_pred != -1
        if not mask.any():
            return jsonify({
                "accuracy": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
                "coverage": "0.00%"
            })

        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]

        accuracy = accuracy_score(y_true_masked, y_pred_masked)
        precision = precision_score(y_true_masked, y_pred_masked, zero_division=0)
        recall = recall_score(y_true_masked, y_pred_masked, zero_division=0)
        f1 = f1_score(y_true_masked, y_pred_masked, zero_division=0)
        coverage = mask.mean()

        return jsonify({
            "accuracy": f"{accuracy:.2%}",
            "precision": f"{precision:.2%}",
            "recall": f"{recall:.2%}",
            "f1": f"{f1:.2%}",
            "coverage": f"{coverage:.2%}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Updated index() route with all requested fixes


def evaluate_weak_supervision():
    label_matrix = apply_all_label_functions(test_df, label_function_library)
    y_pred = majority_vote(label_matrix)
    y_true = test_df["Is Laundering"].values

    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return render_template("flask-library.html", metrics={
        "accuracy": f"{accuracy:.2%}",
        "precision": f"{precision:.2%}",
        "recall": f"{recall:.2%}",
        "f1_score": f"{f1:.2%}",
    })


@app.route("/library")
def library():
    global latest_snorkel_metrics  # Declare as global here
    # Ensure latest_snorkel_metrics exists, even if empty (fallback values)
    if not latest_snorkel_metrics:
        latest_snorkel_metrics = {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "coverage": "N/A"
        }

    conflict_rate = compute_conflict_rate(test_df, label_function_library)

    return render_template("flask-library.html",
                           labels=label_function_library,
                           snorkel_metrics=latest_snorkel_metrics,  # Pass the metrics
                           conflict_rate=conflict_rate)


# ✅ Full /add-to-library Route (Fixed)
# ✅ Full /add-to-library Route (Fully Fixed)
@app.route("/add-to-library", methods=["POST"])
def add_to_library():
    global latest_snorkel_metrics

    label_function_code = request.form.get('label_function')
    source = request.form.get('source', 'Human')
    name = request.form.get('name', 'Untitled Rule')
    rule_type = request.form.get('rule_type')

    try:
        # Clean and validate the code
        cleaned_code = sanitize_and_validate_python(label_function_code)

        # Prepare execution environment (includes helper functions)
        exec_globals = {
            "get_outgoing_count": get_outgoing_count,
            "get_distinct_receivers": get_distinct_receivers,
            "get_incoming_count": get_incoming_count,
            "get_distinct_senders": get_distinct_senders,
            "has_transaction": has_transaction,
            "sum_incoming": sum_incoming,
            "sum_outgoing": sum_outgoing,
            "get_transactions_from": get_transactions_from,
            "get_transactions_to": get_transactions_to,
            "time_diff": time_diff,
            "test_df": test_df,
            "df": test_df,
            "ABSTAIN": -1,
            "POSITIVE": 1,
            "NEGATIVE": 0
        }

        # Compile label function
        exec(cleaned_code, exec_globals)
        func = exec_globals.get("label_function")
        if not func:
            raise ValueError("label_function not found")

        # Signature inspection
        from inspect import signature
        sig = signature(func)

        def safe_apply(row):
            try:
                if len(sig.parameters) == 2:
                    return func(row, test_df)
                else:
                    return func(row)
            except Exception as e:
                print(f"⚠️ Error applying function to row: {e}")
                return ABSTAIN

        matched = test_df.apply(safe_apply, axis=1)
        matched = pd.Series(matched).fillna(0).clip(0, 1).astype(int)  # ensure only 0 or 1
        y_true = (test_df["Is Laundering"] == 1).astype(int)  # enforce binary
        y_pred = matched.values

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        coverage = (y_pred == 1).sum() / len(y_true)

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            chart = generate_confusion_bar_chart(tp, fp, tn, fn)
        except:
            chart = None

        label_function_library.append({
            "name": name,
            "code": cleaned_code,
            "explanation": request.form.get("explanation", ""),
            "creator": source,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "matched_indices": list(np.where(y_pred == 1)[0]),
            "coverage_rate": f"{coverage:.2%}",
            "accuracy": f"{accuracy:.2%}",
            "precision": f"{precision:.2%}",
            "recall": f"{recall:.2%}",
            "f1_score": f"{f1:.2%}",
            "metric_chart": chart,
            "rule_type": rule_type
        })

        retrain_snorkel_label_model()
        return redirect(url_for("library"))

    except SyntaxError as e:
        print(f"❌ Code Error: {str(e)}")
        return f"❌ Code Error: {str(e)}", 400
    except Exception as e:
        print(f"❌ Top-level exception in add_to_library: {str(e)}")
        traceback.print_exc()
        return f"❌ Failed to apply label function: {str(e)}", 500


# ✅ Define retrain_snorkel_label_model function
def retrain_snorkel_label_model():
    global latest_snorkel_metrics
    if len(label_function_library) < 3:
        print("ℹ️ Not enough label functions to train Snorkel LabelModel")
        return

    snorkel_lfs = convert_to_snorkel_lfs(label_function_library, test_df)
    applier = PandasLFApplier(snorkel_lfs)
    L_train = applier.apply(df=test_df)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)

    preds = label_model.predict(L=L_train)
    y_true = test_df["Is Laundering"].values
    valid = preds != ABSTAIN
    y_pred = preds[valid]
    y_true_valid = y_true[valid]

    snorkel_metrics = {
        "accuracy": f"{accuracy_score(y_true_valid, y_pred):.2%}",
        "precision": f"{precision_score(y_true_valid, y_pred, zero_division=0):.2%}",
        "recall": f"{recall_score(y_true_valid, y_pred, zero_division=0):.2%}",
        "f1_score": f"{f1_score(y_true_valid, y_pred, zero_division=0):.2%}",
        "coverage": f"{valid.mean():.2%}"
    }

    print("✅ Snorkel LabelModel retrained")
    latest_snorkel_metrics = snorkel_metrics



def build_snorkel_lfs(label_function_library):
    # This function would convert the label functions into Snorkel format.
    # Example placeholder logic
    snorkel_lfs = []
    for lf in label_function_library:
        def wrapped(row):
            try:
                exec_globals = {}
                exec(lf['code'], {}, exec_globals)
                func = exec_globals.get('label_function')
                return func(row) if func else -1
            except Exception as e:
                print(f"Error executing label function {lf['name']}: {e}")
                return -1
        snorkel_lfs.append(wrapped)
    return snorkel_lfs

# --- Add an endpoint to evaluate weak supervision ---

@app.route("/combine-weak-supervision")
def combine_weak_supervision():
    if len(label_function_library) < 1:
        return "❌ No label functions available", 400

    try:
        # Convert to Snorkel-style LFs
        snorkel_lfs = convert_to_snorkel_lfs(label_function_library, test_df)
        applier = PandasLFApplier(snorkel_lfs)
        L_train = applier.apply(df=test_df)

        # Train Snorkel LabelModel
        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)

        # Predictions
        preds = label_model.predict(L=L_train)
        y_true = test_df["Is Laundering"].values

        # Metrics Calculation
        snorkel_metrics = compute_metrics(y_true, preds)
        snorkel_metrics["coverage"] = f"{(preds != ABSTAIN).sum() / len(preds):.2%}"

        # Conflict Rate Calculation
        conflict_rate = compute_conflict_rate(test_df, label_function_library)

        return render_template(
            "flask-library.html",
            labels=label_function_library,
            snorkel_metrics=snorkel_metrics,
            conflict_rate=conflict_rate
        )

    except Exception as e:
        print(f"❌ Weak supervision error: {e}")
        return f"❌ Error combining weak supervision rules: {str(e)}", 500



# --- Action in the library: Provide LLM feedback, Delete label, Edit ---
@app.route('/delete-label/<int:idx>', methods=['POST'])
def delete_label(idx):
    if 0 <= idx < len(label_function_library):
        label_function_library.pop(idx)
    return redirect(url_for('library'))

@app.route('/edit-label/<int:idx>', methods=['GET'])
def edit_label(idx):
    label = label_function_library[idx]
    matched = label.get("matched_indices", [])

    before_img = create_label_distribution_graph(df)
    after_img = create_label_distribution_graph(df)

    
    return render_template("index.html",
                           label_function_code=label["code"],
                           label_name=label["name"],
                           explanation=label.get("explanation", ""),
                           label_nl="",
                           matched_count=len(matched),
                           table=df.head(20).to_html(classes='data', index=False),
                           labels=label_function_library,
                           before_img=before_img,
                           after_img=after_img,
                           evaluation=None,
                           suggestions=None,
                           editing_idx=idx)  # Send index of label being edited


# --- Generate suggestion in index page ---
def to_slug(name):
    return name.lower().replace(" ", "_")
        
recent_feature_delta = []
recent_dayofweek_delta = {"labels": [], "legit": [], "laundering": []}
recent_timeofday_delta = {"labels": [], "legit": [], "laundering": []}

def format_for_llm(delta_list):
    """Convert list of dicts to aligned label/legit/laundering lists."""
    labels = []
    legit = []
    laundering = []
    for d in delta_list:
        labels.append(d.get("HourOfDay") or d.get("DayOfWeek") or "Unknown")
        if d["Is Laundering"] == 0:
            legit.append(d["percent_change"])
        else:
            laundering.append(d["percent_change"])
    return {"labels": labels, "legit": legit, "laundering": laundering}

import ast
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1

@app.route("/train-labelmodel", methods=["GET", "POST"])
def train_labelmodel():
    try:
        if len(label_function_library) < 1:
            return jsonify({"error": "Need at least 3 labeling functions to train."}), 400

        # Convert to Snorkel LFs and apply them to the dataframe
        snorkel_lfs = convert_to_snorkel_lfs(label_function_library, test_df)
        applier = PandasLFApplier(snorkel_lfs)
        L_train = applier.apply(df=test_df)

        # Train the model
        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)

        # Get predictions
        preds = label_model.predict(L=L_train)
        y_true = test_df["Is Laundering"].values

        # Remove abstentions
        valid = preds != ABSTAIN
        y_pred = preds[valid]
        y_true_valid = y_true[valid]

        # Calculate metrics
        snorkel_metrics = {
            "accuracy": f"{accuracy_score(y_true_valid, y_pred):.2%}",
            "precision": f"{precision_score(y_true_valid, y_pred, zero_division=0):.2%}",
            "recall": f"{recall_score(y_true_valid, y_pred, zero_division=0):.2%}",
            "f1_score": f"{f1_score(y_true_valid, y_pred, zero_division=0):.2%}",
            "coverage": f"{valid.mean():.2%}"
        }

        print("✅ Snorkel Metrics Calculated:", snorkel_metrics)

        # Send the metrics as JSON response
        return jsonify({"metrics": snorkel_metrics})

    except Exception as e:
        print("❌ Error in /train-labelmodel:", e)
        return jsonify({"error": str(e)}), 500


# --- EDA ---

@app.route("/tutorial")
def tutorial():
    return render_template("eda.html")


# ✅ Updated index() route with correct indentation for safe_execute()
@app.route("/sensitivity-analysis", methods=["POST"])
def eda_sensitivity():
    try:
        df = pd.read_csv("train_sample.csv")
        df = df.dropna(subset=["Is Laundering"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["HourOfDay"] = df["Timestamp"].dt.hour
        df["DayOfWeek"] = df["Timestamp"].dt.day_name()

        # Filter by HourOfDay if specified
        hour = request.form.get("hour", "").strip()
        if hour != "":
            try:
                hour = int(hour)
                df = df[df["HourOfDay"] == hour]
            except ValueError:
                pass  # Invalid input will be ignored

        # Filter by DayOfWeek if specified
        day_index = request.form.get("dayOfWeek", "").strip()
        if day_index != "":
            try:
                day_index = int(day_index)
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                if 0 <= day_index < len(day_names):
                    df = df[df["DayOfWeek"] == day_names[day_index]]
            except ValueError:
                pass

        # Label encode categorical features
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        feature_cols = [col for col in df.columns if col not in ["Is Laundering", "Timestamp", "Date"]]
        X = df[feature_cols]
        y = df["Is Laundering"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        raw_payload = {k.replace("_", " ").title(): v for k, v in request.get_json(force=True).items()}
        print("Normalized payload keys:", list(raw_payload.keys()))

        for key in ["Amount Paid", "Amount Received"]:
            min_val = float(raw_payload.get(f"{key} Min", 0))
            max_val = float(raw_payload.get(f"{key} Max", 0))
            if min_val > max_val:
                return jsonify({"status": "error", "message": f"{key} Min cannot be greater than Max"}), 400

        sample_input = {}
        for col in feature_cols:
            if col in raw_payload:
                val = raw_payload[col]
                if col in label_encoders:
                    if val not in label_encoders[col].classes_:
                        return jsonify({"status": "error", "message": f"Invalid value for {col}: {val}"}), 400
                    val = label_encoders[col].transform([val])[0]
                sample_input[col] = float(val)
            else:
                sample_input[col] = X[col].mode()[0] if col in categorical_cols else X[col].mean()

        sample = pd.DataFrame([sample_input])
        prob_original = model.predict_proba(X)[:, 1].mean()
        prob_sample = model.predict_proba(sample)[0][1]
        uplift = (prob_sample - prob_original) * 100

        return jsonify({
            "original": round(prob_original, 4),
            "perturbed": round(prob_sample, 4),
            "uplift": round(uplift, 2)
        })

    except Exception as e:
        print("❌ Sensitivity analysis error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route("/eda/available-currencies")
def available_currencies():
    df = pd.read_csv("train_sample.csv")
    df.columns = df.columns.str.strip()

    currencies = set()
    if "Payment Currency" in df.columns:
        currencies.update(df["Payment Currency"].dropna().unique())
    if "Receiving Currency" in df.columns:
        currencies.update(df["Receiving Currency"].dropna().unique())

    return jsonify(sorted(list(currencies)))


@app.route("/eda/payment-formats")
def payment_formats():
    df = pd.read_csv("train_sample.csv")
    formats = df["Payment Format"].dropna().unique().tolist()
    return jsonify(formats)




from urllib.parse import quote

@app.route("/timeofday-data")
def timeofday_data():
    try:
        before = df.copy()
        after = df.copy()
        if "Human Label" in after.columns:
            after["Is Laundering"] = after["Human Label"]

        def summarize(dfset):
            dfset["HourOfDay"] = dfset["Timestamp"].dt.hour
            return dfset.groupby(["HourOfDay", "Is Laundering"]).size().reset_index(name="count")

        def format_deltas(before_df, after_df):
            results = {"labels": [], "legit": [], "laundering": []}
            for hour in range(24):
                for label in [0, 1]:
                    b = before_df.query("HourOfDay == @hour and Is Laundering == @label")["count"].sum()
                    a = after_df.query("HourOfDay == @hour and Is Laundering == @label")["count"].sum()
                    delta = round(100 * (a - b) / (b + 1e-5), 2)
                    results["labels"].append(hour)
                    if label == 0:
                        results["legit"].append(delta)
                    else:
                        results["laundering"].append(delta)
            return results

        b = summarize(before)
        a = summarize(after)
        global recent_timeofday_delta
        recent_timeofday_delta = format_deltas(b, a)

        return jsonify({"before": b.to_dict("records"), "after": a.to_dict("records"), "delta": recent_timeofday_delta})

    except Exception as e:
        print("❌ /timeofday-data failed:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/dayofweek-data")
def dayofweek_data():
    try:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        before = df.copy()
        after = df.copy()
        if "Human Label" in after.columns:
            after["Is Laundering"] = after["Human Label"]

        def summarize(dfset):
            dfset["DayOfWeek"] = pd.Categorical(dfset["Timestamp"].dt.day_name(), categories=days, ordered=True)
            return dfset.groupby(["DayOfWeek", "Is Laundering"]).size().reset_index(name="count")

        def format_deltas(before_df, after_df):
            results = {"labels": days, "legit": [], "laundering": []}
            for day in days:
                for label in [0, 1]:
                    b = before_df.query("DayOfWeek == @day and Is Laundering == @label")["count"].sum()
                    a = after_df.query("DayOfWeek == @day and Is Laundering == @label")["count"].sum()
                    delta = round(100 * (a - b) / (b + 1e-5), 2)
                    if label == 0:
                        results["legit"].append(delta)
                    else:
                        results["laundering"].append(delta)
            return results

        b = summarize(before)
        a = summarize(after)
        global recent_dayofweek_delta
        recent_dayofweek_delta = format_deltas(b, a)

        return jsonify({"before": b.to_dict("records"), "after": a.to_dict("records"), "delta": recent_dayofweek_delta})

    except Exception as e:
        print("❌ /dayofweek-data failed:", e)
        return jsonify({"error": str(e)}), 500

def inject_defensive_checks(code: str) -> str:
    if "row['Timestamp'].hour" in code:
        code = code.replace(
            "row['Timestamp'].hour",
            "(row['Timestamp'].hour if isinstance(row['Timestamp'], pd.Timestamp) else -1)"
        )
    return code



import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Boolean Series key will be reindexed")

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))  # Render sets this
#     app.run(host='0.0.0.0', port=port, use_reloader=False)