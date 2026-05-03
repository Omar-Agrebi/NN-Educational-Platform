"""
Custom dataset ingestion: CSV / XLSX upload, profiling, auto-detection, encoding.
Pure NumPy + stdlib — no pandas dependency at runtime (pandas used only if available).
"""
import numpy as np
import io
import csv
from typing import Dict, Any, List, Tuple, Optional


# ── File parsing ──────────────────────────────────────────────────────────────

def parse_csv(content: bytes) -> Tuple[List[str], List[List[str]]]:
    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV file is empty")
    headers = [h.strip() for h in rows[0]]
    data_rows = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    return headers, data_rows


def parse_xlsx(content: bytes) -> Tuple[List[str], List[List[str]]]:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        rows = [[str(cell.value) if cell.value is not None else "" for cell in row] for row in ws.iter_rows()]
        wb.close()
    except ImportError:
        # fallback: try xlrd
        try:
            import xlrd
            wb = xlrd.open_workbook(file_contents=content)
            ws = wb.sheet_by_index(0)
            rows = [[str(ws.cell_value(r, c)) for c in range(ws.ncols)] for r in range(ws.nrows)]
        except ImportError:
            raise ImportError("Install openpyxl or xlrd to read XLSX files: pip install openpyxl")
    if not rows:
        raise ValueError("XLSX file is empty")
    headers = [str(h).strip() for h in rows[0]]
    data_rows = rows[1:]
    return headers, data_rows


def try_float(val: str) -> Optional[float]:
    try:
        return float(val.replace(",", "").strip())
    except Exception:
        return None


# ── Column profiling ──────────────────────────────────────────────────────────

def profile_columns(headers: List[str], rows: List[List[str]]) -> List[Dict[str, Any]]:
    n = len(rows)
    profiles = []
    for ci, col in enumerate(headers):
        values = [row[ci] if ci < len(row) else "" for row in rows]
        missing = sum(1 for v in values if v.strip() in ("", "None", "nan", "NaN", "null", "NULL", "NA"))
        numeric_vals = [try_float(v) for v in values if v.strip() not in ("", "None", "nan", "NaN", "null", "NULL", "NA")]
        numeric_vals = [v for v in numeric_vals if v is not None]
        unique_vals = list(set(v for v in values if v.strip()))
        is_numeric = len(numeric_vals) >= (n - missing) * 0.7
        profiles.append({
            "name": col,
            "index": ci,
            "n_total": n,
            "n_missing": missing,
            "missing_pct": round(missing / max(n, 1) * 100, 1),
            "is_numeric": is_numeric,
            "n_unique": len(unique_vals),
            "unique_values": unique_vals[:20],  # sample
            "numeric_min": round(min(numeric_vals), 4) if numeric_vals else None,
            "numeric_max": round(max(numeric_vals), 4) if numeric_vals else None,
            "numeric_mean": round(float(np.mean(numeric_vals)), 4) if numeric_vals else None,
            "likely_id": col.lower() in ("id", "index", "idx", "rowid", "row_id", "unnamed: 0")
                         or (is_numeric and len(unique_vals) == n),
        })
    return profiles


# ── Problem auto-detection ────────────────────────────────────────────────────

def detect_problem_type(target_profile: Dict[str, Any]) -> Dict[str, Any]:
    n_unique = target_profile["n_unique"]
    is_numeric = target_profile["is_numeric"]

    if n_unique == 2:
        return {"type": "binary_classification", "confidence": 0.97,
                "reason": f"Target has exactly 2 unique values: {target_profile['unique_values'][:2]}"}
    elif not is_numeric and n_unique <= 20:
        return {"type": "multiclass_classification", "confidence": 0.92,
                "reason": f"Target is categorical with {n_unique} classes"}
    elif is_numeric and n_unique <= 15:
        return {"type": "multiclass_classification", "confidence": 0.82,
                "reason": f"Target is numeric integer with {n_unique} unique values — treating as classes"}
    elif is_numeric and n_unique > 15:
        return {"type": "regression", "confidence": 0.90,
                "reason": f"Target is continuous numeric with {n_unique} unique values"}
    else:
        return {"type": "binary_classification", "confidence": 0.50,
                "reason": "Could not determine — defaulting to binary classification"}


# ── Feature encoding & cleaning ───────────────────────────────────────────────

def encode_features(
    headers: List[str],
    rows: List[List[str]],
    feature_cols: List[int],
    target_col: int,
    problem_type: str,
) -> Dict[str, Any]:
    n = len(rows)

    # Extract raw columns
    raw_features = {ci: [rows[r][ci] if ci < len(rows[r]) else "" for r in range(n)] for ci in feature_cols}
    raw_target = [rows[r][target_col] if target_col < len(rows[r]) else "" for r in range(n)]

    # Build feature matrix
    feature_arrays = []
    feature_names_out = []
    encoder_info = {}

    for ci in feature_cols:
        col_name = headers[ci]
        vals = raw_features[ci]
        numeric_vals = [try_float(v) for v in vals]
        is_num = sum(v is not None for v in numeric_vals) >= n * 0.7

        if is_num:
            # Impute missing with mean
            known = [v for v in numeric_vals if v is not None]
            mean_val = float(np.mean(known)) if known else 0.0
            imputed = [v if v is not None else mean_val for v in numeric_vals]
            feature_arrays.append(np.array(imputed).reshape(-1, 1))
            feature_names_out.append(col_name)
            encoder_info[col_name] = {"type": "numeric", "impute_mean": mean_val}
        else:
            # One-hot encode
            unique_cats = sorted(set(v for v in vals if v.strip()))
            if len(unique_cats) > 20:
                # Too many categories — skip or label encode
                label_map = {c: i for i, c in enumerate(unique_cats)}
                encoded = np.array([label_map.get(v, 0) for v in vals], dtype=float).reshape(-1, 1)
                feature_arrays.append(encoded)
                feature_names_out.append(col_name + "_label")
                encoder_info[col_name] = {"type": "label", "map": label_map}
            else:
                for cat in unique_cats:
                    encoded = np.array([1.0 if v == cat else 0.0 for v in vals]).reshape(-1, 1)
                    feature_arrays.append(encoded)
                    feature_names_out.append(f"{col_name}_{cat}")
                encoder_info[col_name] = {"type": "onehot", "categories": unique_cats}

    X = np.hstack(feature_arrays) if feature_arrays else np.zeros((n, 1))

    # Encode target
    if problem_type in ("binary_classification", "multiclass_classification"):
        target_num = [try_float(v) for v in raw_target]
        is_num_target = sum(t is not None for t in target_num) >= n * 0.9

        if is_num_target:
            y_raw = np.array([t if t is not None else 0.0 for t in target_num])
            classes = sorted(set(int(v) for v in y_raw))
            class_map = {c: i for i, c in enumerate(classes)}
            y = np.array([class_map[int(v)] for v in y_raw], dtype=float)
        else:
            classes = sorted(set(v.strip() for v in raw_target if v.strip()))
            class_map = {c: i for i, c in enumerate(classes)}
            y = np.array([class_map.get(v.strip(), 0) for v in raw_target], dtype=float)

        n_classes = len(classes)
        target_info = {"classes": classes, "class_map": class_map, "n_classes": n_classes}
    else:
        # Regression
        target_num = [try_float(v) for v in raw_target]
        known = [t for t in target_num if t is not None]
        mean_t = float(np.mean(known)) if known else 0.0
        y = np.array([t if t is not None else mean_t for t in target_num], dtype=float)
        target_info = {"n_classes": 1, "mean": mean_t}

    # Remove rows with all-zero or NaN features
    valid = ~np.any(np.isnan(X), axis=1)
    X = X[valid]
    y = y[valid]

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names_out,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "target_info": target_info,
        "encoder_info": encoder_info,
        "problem_type": problem_type,
    }


# ── Baseline model ────────────────────────────────────────────────────────────

def baseline_score(y_train: np.ndarray, y_test: np.ndarray, problem_type: str) -> Dict[str, float]:
    if problem_type == "binary_classification":
        majority = 1.0 if y_train.mean() >= 0.5 else 0.0
        preds = np.full_like(y_test, majority)
        acc = float(np.mean(preds == y_test))
        return {"accuracy": round(acc, 4), "strategy": f"always predict {int(majority)}"}

    elif problem_type == "multiclass_classification":
        from collections import Counter
        majority = Counter(y_train.astype(int).tolist()).most_common(1)[0][0]
        preds = np.full_like(y_test, majority)
        acc = float(np.mean(preds == y_test))
        return {"accuracy": round(acc, 4), "strategy": f"always predict class {majority}"}

    else:  # regression
        mean_pred = float(y_train.mean())
        mse = float(np.mean((y_test - mean_pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        r2 = 1 - float(np.sum((y_test - mean_pred) ** 2)) / (ss_tot + 1e-10)
        return {"mse": round(mse, 4), "r2": round(r2, 4), "rmse": round(mse ** 0.5, 4),
                "strategy": f"always predict mean ({mean_pred:.3f})"}
