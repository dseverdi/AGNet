#!/usr/bin/env python3
import pandas as pd
import sys
from pathlib import Path

def csv_to_tabular(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    # Column alignment: left for first column (usually Method), right for others
    align = "l" + "r" * (len(df.columns) - 1)
    latex = "\\begin{tabular}{" + align + "}\n    \\toprule\n"
    # Process columns to replace Greek letters
    columns = [col.replace("μ", "\\mu").replace("σ", "\\sigma").replace("±", "\\pm") for col in df.columns]
    # Wrap in $ if contains math
    wrapped_columns = []
    for col in columns:
        if "\\mu" in col or "\\pm" in col:
            wrapped_columns.append("$" + col + "$")
        else:
            wrapped_columns.append(col)
    latex += " & ".join(wrapped_columns) + " \\\\\n    \\midrule\n"
    for _, row in df.iterrows():
        vals = []
        for i, val in enumerate(row.values):
            sval = str(val)
            # escape underscores in text fields
            if i == 0:
                sval = sval.replace("_", "\\_")
            # replace Greek letters
            sval = sval.replace("μ", "\\mu")
            sval = sval.replace("σ", "\\sigma").replace("±", "\\pm")
            # wrap in $ if contains math
            if "\\pm" in sval:
                sval = "$" + sval + "$"
            vals.append(sval)
        latex += " & ".join(vals) + " \\\\\n"
    latex += "    \\bottomrule\n\\end{tabular}\n"
    return latex

if __name__ == "__main__":
    # Usage: python csv_to_table.py <csv_path> <out_tex_path>
    if len(sys.argv) != 3:
        print("Usage: python csv_to_table.py <csv_path> <out_tex_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = csv_to_tabular(csv_path)
    out_path.write_text(content)
    print(f"Wrote LaTeX tabular to {out_path}")