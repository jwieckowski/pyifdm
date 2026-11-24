# Copyright (c) 2026 Jakub Więckowski & contributors
"""
Step logger for Intuitionistic Fuzzy MCDA method execution traces.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

__all__ = ["StepLogger"]

_VALID_FORMATS = ("console", "json", "csv", "excel")


def _try_tabulate(table, headers, floatfmt=".4f"):
    try:
        from tabulate import tabulate

        return tabulate(
            table,
            headers=headers,
            tablefmt="rounded_outline",
            floatfmt=floatfmt,
        )
    except ImportError:
        # Minimal plain-text fallback
        lines = []
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-+-".join("-" * max(4, len(str(h))) for h in headers))
        for row in table:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)


class StepLogger:
    """
    Collects named intermediate results from an IFS MCDA computation
    and renders them in one or more formats.

    Formats
    -------
    console
        Prints aligned tables to stdout.
    json
        Writes a structured JSON file.
    csv
        Writes one CSV file with all steps separated by blank rows.
    excel
        Writes an .xlsx workbook with one sheet per step.
        Requires openpyxl: ``pip install openpyxl``.

    Parameters
    ----------
    output :
        One or more of ``'console'``, ``'json'``, ``'csv'``, ``'excel'``.
    path :
        Base path (without extension) for file outputs. Ignored for console.
        Defaults to ``pyifdm_steps_<timestamp>``.
    alternative_names :
        Row labels for matrix / vector steps.
    criterion_names :
        Column labels for matrix steps.
    float_fmt :
        Format string for floating-point values (default ``'.4f'``).

    Examples
    --------
    >>> logger = StepLogger(output="console")
    >>> logger = StepLogger(output=["json", "csv"], path="results/topsis")
    >>> method = ifTOPSIS(logger=logger)
    >>> prefs = method(matrix, weights, types)
    >>> logger.save()   # write file-based outputs (also called by the method)
    >>> logger.print()  # force console output
    """

    def __init__(
        self,
        output: Union[str, Sequence[str]] = "console",
        path: Optional[Union[str, Path]] = None,
        alternative_names: Optional[Sequence[str]] = None,
        criterion_names: Optional[Sequence[str]] = None,
        float_fmt: str = ".4f",
    ) -> None:
        self.output = self._parse_output(output)
        self.path = (
            Path(path)
            if path
            else Path(f"pyifdm_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
        self.alt_names = list(alternative_names) if alternative_names else None
        self.crit_names = list(criterion_names) if criterion_names else None
        self.float_fmt = float_fmt

        self._steps: list[dict[str, Any]] = []
        self._method_name: str = ""

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def log(self, name: str, data: Any, description: str = "") -> None:
        """
        Record one intermediate result.

        Parameters
        ----------
        name :
            Human-readable step name, e.g. ``'Normalised matrix'``.
        data :
            Intermediate value: ndarray (ndim 1–3), scalar, or dict.
        description :
            Optional one-line explanation shown in headers.
        """
        self._steps.append(
            {
                "name": name,
                "description": description,
                "data": data,
            }
        )

    def _set_method(self, name: str) -> None:
        self._method_name = name

    def clear(self) -> None:
        """Remove all recorded steps (useful for re-running a method)."""
        self._steps.clear()

    @property
    def steps(self) -> list:
        """Read-only list of recorded step dicts."""
        return list(self._steps)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    def _resolve_labels(self, arr: np.ndarray):
        """
        Return ``(row_labels, col_labels)`` for *arr*.

        Falls back to positional defaults when stored names do not match
        the array dimensions (e.g. ideal solutions indexed by criteria).
        """
        if arr.ndim == 1:
            m = arr.shape[0]
            rows = (
                self.alt_names
                if (self.alt_names and len(self.alt_names) == m)
                else [f"A{i + 1}" for i in range(m)]
            )
            return rows, None
        if arr.ndim >= 2:
            m, n = arr.shape[0], arr.shape[1]
            rows = (
                self.alt_names
                if (self.alt_names and len(self.alt_names) == m)
                else [f"R{i + 1}" for i in range(m)]
            )
            cols = (
                self.crit_names
                if (self.crit_names and len(self.crit_names) == n)
                else [f"C{j + 1}" for j in range(n)]
            )
            return rows, cols
        return [], []

    @staticmethod
    def _format_ifs_cell(values: np.ndarray, float_fmt: str) -> str:
        """Format one IFS cell as (μ, ν) or (μ, ν, π)."""
        parts = [f"{float(v):{float_fmt}}" for v in values]
        return "(" + ", ".join(parts) + ")"

    # ------------------------------------------------------------------
    # Public output
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Write all file-based outputs (json / csv / excel)."""
        if "json" in self.output:
            self._write_json()
        if "csv" in self.output:
            self._write_csv()
        if "excel" in self.output:
            self._write_excel()

    def print(self) -> None:
        """Print all steps to the console as formatted tables."""
        self._write_console()

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------
    def _write_console(self) -> None:
        width = 72
        header = f"  {self._method_name} — Intermediate Results  "
        print("\n" + "═" * width)
        print(header.center(width))
        print("═" * width)

        for idx, step in enumerate(self._steps, 1):
            name = step["name"]
            desc = step["description"]
            data = step["data"]

            print(f"\n  [{idx}] {name}")
            if desc:
                print(f"      {desc}")
            print("  " + "─" * (width - 2))
            self._print_data(data)

        print("\n" + "═" * width + "\n")

    def _print_data(self, data: Any) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"\n  {key}")
                self._print_data(value)
            return

        if not isinstance(data, np.ndarray):
            print(f"  {data}")
            return

        arr = np.asarray(data)
        rows, cols = self._resolve_labels(arr)
        fmt = self.float_fmt

        if arr.ndim == 1:
            table = [[label, float(val)] for label, val in zip(rows, arr)]
            print(
                _try_tabulate(
                    table,
                    headers=["", "value"],
                    floatfmt=fmt,
                )
            )
            return

        if arr.ndim == 2:
            table = []
            for label, row in zip(rows, arr):
                table.append([label] + [float(v) for v in row.tolist()])
            print(
                _try_tabulate(
                    table,
                    headers=[""] + list(cols),
                    floatfmt=fmt,
                )
            )
            return

        if arr.ndim == 3:
            # IFS matrix / vector of IFS: last axis is 2 (μ,ν) or 3 (μ,ν,π)
            table = []
            for rlabel, row in zip(rows, arr):
                cells = [
                    self._format_ifs_cell(cell, fmt) for cell in row
                ]
                table.append([rlabel] + cells)
            print(
                _try_tabulate(
                    table,
                    headers=[""] + list(cols),
                    floatfmt=fmt,
                )
            )
            return

        print(arr)

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    def _write_json(self) -> None:
        path = self.path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "method": self._method_name,
            "generated": datetime.now().isoformat(),
            "steps": [],
        }
        for step in self._steps:
            payload["steps"].append(
                {
                    "name": step["name"],
                    "description": step["description"],
                    "data": self._to_json_serialisable(step["data"]),
                }
            )

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"[StepLogger] JSON saved → {path}")

    def _to_json_serialisable(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            arr = np.asarray(data, dtype=float)
            rows, cols = self._resolve_labels(arr)
            if arr.ndim == 1:
                return {
                    "shape": list(arr.shape),
                    "labels": rows,
                    "values": arr.tolist(),
                }
            if arr.ndim == 2:
                return {
                    "shape": list(arr.shape),
                    "row_labels": rows,
                    "col_labels": cols,
                    "values": arr.tolist(),
                }
            if arr.ndim == 3:
                m, n, k = arr.shape
                row_labels = rows or [f"A{i + 1}" for i in range(m)]
                col_labels = cols or [f"C{j + 1}" for j in range(n)]
                component = ["mu", "nu", "pi"][:k]
                ifs_data: dict[str, Any] = {}
                for i, rl in enumerate(row_labels):
                    ifs_data[rl] = {}
                    for j, cl in enumerate(col_labels):
                        ifs_data[rl][cl] = {
                            component[t]: float(arr[i, j, t]) for t in range(k)
                        }
                return {
                    "shape": list(arr.shape),
                    "components": component,
                    "ifs": ifs_data,
                }
            return {"values": arr.tolist()}

        if isinstance(data, dict):
            return {k: self._to_json_serialisable(v) for k, v in data.items()}
        if isinstance(data, (float, int, np.floating, np.integer)):
            return float(data)
        if isinstance(data, (list, tuple)):
            return [self._to_json_serialisable(v) for v in data]
        return str(data)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    def _write_csv(self) -> None:
        path = self.path.with_suffix(".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = f"{{:{self.float_fmt}}}"

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"# {self._method_name} — Intermediate Steps"])
            writer.writerow([f"# Generated: {datetime.now().isoformat()}"])
            writer.writerow([])

            for idx, step in enumerate(self._steps, 1):
                writer.writerow([f'[{idx}] {step["name"]}'])
                if step["description"]:
                    writer.writerow([f'# {step["description"]}'])
                self._write_csv_data(writer, step["data"], fmt)
                writer.writerow([])

        print(f"[StepLogger] CSV saved → {path}")

    def _write_csv_data(self, writer, data: Any, fmt: str) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                writer.writerow([f"## {key}"])
                self._write_csv_data(writer, value, fmt)
            return

        if not isinstance(data, np.ndarray):
            writer.writerow([str(data)])
            return

        arr = np.asarray(data, dtype=float)
        rows, cols = self._resolve_labels(arr)

        if arr.ndim == 1:
            writer.writerow(rows or [f"A{i + 1}" for i in range(arr.shape[0])])
            writer.writerow([fmt.format(v) for v in arr])
            return

        if arr.ndim == 2:
            col_labels = cols or [f"C{j + 1}" for j in range(arr.shape[1])]
            row_labels = rows or [f"A{i + 1}" for i in range(arr.shape[0])]
            writer.writerow([""] + list(col_labels))
            for i, rl in enumerate(row_labels):
                writer.writerow([rl] + [fmt.format(v) for v in arr[i]])
            return

        if arr.ndim == 3:
            m, n, k = arr.shape
            row_labels = rows or [f"A{i + 1}" for i in range(m)]
            col_labels = cols or [f"C{j + 1}" for j in range(n)]
            suffixes = ["mu", "nu", "pi"][:k]
            header = [""]
            for col in col_labels:
                header += [f"{col}_{s}" for s in suffixes]
            writer.writerow(header)
            for i, rl in enumerate(row_labels):
                row = [rl]
                for j in range(n):
                    row += [fmt.format(v) for v in arr[i, j]]
                writer.writerow(row)

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------
    def _write_excel(self) -> None:
        try:
            import openpyxl
            from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        except ImportError as exc:
            raise ImportError(
                "Excel output requires openpyxl: pip install openpyxl"
            ) from exc

        path = self.path.with_suffix(".xlsx")
        path.parent.mkdir(parents=True, exist_ok=True)

        blue, lblue, gray, white = "FF1F4E79", "FFD6E4F0", "FFF2F2F2", "FFFFFFFF"
        thin = Side(style="thin", color="FFCCCCCC")

        def bdr():
            return Border(left=thin, right=thin, top=thin, bottom=thin)

        def hfill():
            return PatternFill("solid", fgColor=blue)

        def sfill():
            return PatternFill("solid", fgColor=lblue)

        def hfont():
            return Font(bold=True, color="FFFFFFFF", name="Arial", size=10)

        def nfont():
            return Font(name="Arial", size=10)

        def bfont():
            return Font(bold=True, name="Arial", size=10)

        fmt = f"{{:{self.float_fmt}}}"
        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        ws_ov = wb.create_sheet("Overview", 0)
        ws_ov.column_dimensions["A"].width = 6
        ws_ov.column_dimensions["B"].width = 30
        ws_ov.column_dimensions["C"].width = 50

        ws_ov.merge_cells("A1:C1")
        title = ws_ov["A1"]
        title.value = f"{self._method_name} — Intermediate Computation Steps"
        title.fill = hfill()
        title.font = hfont()
        title.alignment = Alignment(horizontal="center")

        ws_ov["A2"], ws_ov["B2"], ws_ov["C2"] = "#", "Step name", "Description"
        for col in "ABC":
            ws_ov[f"{col}2"].fill = sfill()
            ws_ov[f"{col}2"].font = bfont()
            ws_ov[f"{col}2"].border = bdr()

        for idx, step in enumerate(self._steps, 1):
            ws_ov[f"A{idx + 2}"] = idx
            ws_ov[f"B{idx + 2}"] = step["name"]
            ws_ov[f"C{idx + 2}"] = step["description"]
            fill = PatternFill("solid", fgColor=gray if idx % 2 == 0 else white)
            for col in "ABC":
                cell = ws_ov[f"{col}{idx + 2}"]
                cell.fill = fill
                cell.font = nfont()
                cell.border = bdr()

        for idx, step in enumerate(self._steps, 1):
            sheet_name = f'{idx}. {step["name"]}'[:31]
            ws = wb.create_sheet(title=sheet_name)
            self._write_excel_step(
                ws,
                step,
                hfill,
                sfill,
                hfont,
                nfont,
                bfont,
                bdr,
                gray,
                white,
                fmt,
                openpyxl,
            )

        wb.save(path)
        print(f"[StepLogger] Excel saved → {path}")

    def _write_excel_step(
        self,
        ws,
        step,
        hfill,
        sfill,
        hfont,
        nfont,
        bfont,
        bdr,
        gray,
        white,
        fmt,
        openpyxl,
    ) -> None:
        from openpyxl.styles import PatternFill

        data = step["data"]
        if not isinstance(data, np.ndarray):
            if isinstance(data, dict):
                row = 1
                for key, value in data.items():
                    ws.cell(row=row, column=1, value=str(key)).font = bfont()
                    if isinstance(value, np.ndarray):
                        row = self._excel_write_array(
                            ws,
                            value,
                            row + 1,
                            fmt,
                            sfill,
                            nfont,
                            bfont,
                            bdr,
                            gray,
                            white,
                            openpyxl,
                        )
                    else:
                        ws.cell(row=row + 1, column=1, value=str(value)).font = nfont()
                        row += 2
            else:
                ws.cell(row=1, column=1, value=str(data)).font = nfont()
            return

        self._excel_write_array(
            ws,
            np.asarray(data, dtype=float),
            1,
            fmt,
            sfill,
            nfont,
            bfont,
            bdr,
            gray,
            white,
            openpyxl,
        )

    def _excel_write_array(
        self,
        ws,
        arr: np.ndarray,
        start_row: int,
        fmt: str,
        sfill,
        nfont,
        bfont,
        bdr,
        gray,
        white,
        openpyxl,
    ) -> int:
        from openpyxl.styles import Alignment, PatternFill

        rows, cols = self._resolve_labels(arr)

        if arr.ndim == 1:
            row_labels = rows or [f"A{i + 1}" for i in range(arr.shape[0])]
            for j, rl in enumerate(row_labels):
                cell = ws.cell(row=start_row, column=j + 2, value=rl)
                cell.fill = sfill()
                cell.font = bfont()
                cell.border = bdr()
                cell.alignment = Alignment(horizontal="center")
            for j, value in enumerate(arr):
                cell = ws.cell(row=start_row + 1, column=j + 2, value=float(value))
                cell.font = nfont()
                cell.border = bdr()
                cell.number_format = "0.0000"
                cell.alignment = Alignment(horizontal="center")
            ws.column_dimensions[openpyxl.utils.get_column_letter(1)].width = 4
            return start_row + 3

        if arr.ndim == 2:
            m, n = arr.shape
            row_labels = rows or [f"A{i + 1}" for i in range(m)]
            col_labels = cols or [f"C{j + 1}" for j in range(n)]
            ws.cell(row=start_row, column=1, value="").border = bdr()
            for j, cl in enumerate(col_labels):
                cell = ws.cell(row=start_row, column=j + 2, value=cl)
                cell.fill = sfill()
                cell.font = bfont()
                cell.border = bdr()
                cell.alignment = Alignment(horizontal="center")
            for i, rl in enumerate(row_labels):
                fill = PatternFill("solid", fgColor=gray if i % 2 == 1 else white)
                cell = ws.cell(row=start_row + i + 1, column=1, value=rl)
                cell.fill = fill
                cell.font = bfont()
                cell.border = bdr()
                for j, value in enumerate(arr[i]):
                    cell = ws.cell(
                        row=start_row + i + 1, column=j + 2, value=float(value)
                    )
                    cell.fill = fill
                    cell.font = nfont()
                    cell.border = bdr()
                    cell.number_format = "0.0000"
                    cell.alignment = Alignment(horizontal="center")
            ws.column_dimensions[openpyxl.utils.get_column_letter(1)].width = 14
            for j in range(n):
                ws.column_dimensions[
                    openpyxl.utils.get_column_letter(j + 2)
                ].width = 12
            return start_row + m + 2

        if arr.ndim == 3:
            m, n, k = arr.shape
            row_labels = rows or [f"A{i + 1}" for i in range(m)]
            col_labels = cols or [f"C{j + 1}" for j in range(n)]
            suffixes = ["μ", "ν", "π"][:k]
            ws.cell(row=start_row, column=1, value="").border = bdr()
            for j, cl in enumerate(col_labels):
                col_start = 2 + j * k
                if k > 1:
                    ws.merge_cells(
                        start_row=start_row,
                        start_column=col_start,
                        end_row=start_row,
                        end_column=col_start + k - 1,
                    )
                cell = ws.cell(row=start_row, column=col_start, value=cl)
                cell.fill = sfill()
                cell.font = bfont()
                cell.border = bdr()
                cell.alignment = Alignment(horizontal="center")
                for t, sub in enumerate(suffixes):
                    cell2 = ws.cell(
                        row=start_row + 1, column=col_start + t, value=sub
                    )
                    cell2.fill = PatternFill("solid", fgColor="FFBDD7EE")
                    cell2.font = bfont()
                    cell2.border = bdr()
                    cell2.alignment = Alignment(horizontal="center")
            ws.cell(row=start_row + 1, column=1, value="").border = bdr()
            for i, rl in enumerate(row_labels):
                fill = PatternFill("solid", fgColor=gray if i % 2 == 1 else white)
                cell = ws.cell(row=start_row + i + 2, column=1, value=rl)
                cell.fill = fill
                cell.font = bfont()
                cell.border = bdr()
                for j in range(n):
                    for t in range(k):
                        cell2 = ws.cell(
                            row=start_row + i + 2,
                            column=2 + j * k + t,
                            value=float(arr[i, j, t]),
                        )
                        cell2.fill = fill
                        cell2.font = nfont()
                        cell2.border = bdr()
                        cell2.number_format = "0.0000"
                        cell2.alignment = Alignment(horizontal="center")
            ws.column_dimensions[openpyxl.utils.get_column_letter(1)].width = 14
            for j in range(n):
                for t in range(k):
                    letter = openpyxl.utils.get_column_letter(2 + j * k + t)
                    ws.column_dimensions[letter].width = 8
            return start_row + m + 3

        return start_row + 2

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_output(output: Union[str, Sequence[str]]) -> set:
        if isinstance(output, str):
            output = [output]
        result = set()
        for fmt in output:
            fmt = str(fmt).strip().lower()
            if fmt not in _VALID_FORMATS:
                raise ValueError(
                    f"Invalid output format '{fmt}'. "
                    f"Valid options: {_VALID_FORMATS}."
                )
            result.add(fmt)
        return result

    def __repr__(self) -> str:
        return (
            f"StepLogger(method={self._method_name!r}, "
            f"steps={len(self._steps)}, output={sorted(self.output)})"
        )
