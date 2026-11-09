"""Console + CSV logging. The CSV is what the figures are built from."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: str | Path, resume: bool = False) -> None:
        self.csv_path = Path(run_dir) / "log.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        self._fields: list[str] | None = None
        if resume and self.csv_path.exists():  # keep the epochs already logged
            with self.csv_path.open(newline="", encoding="utf-8") as fh:
                self._fields = next(csv.reader(fh), None)

    def drop_from(self, epoch: int) -> None:
        """Forget epochs at or after `epoch`. A crash after the last checkpoint leaves rows the
        resumed run is about to redo, and duplicates would corrupt the per-epoch figures."""
        if not self.csv_path.exists():
            return
        with self.csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, self._fields)
            writer.writeheader()
            writer.writerows(r for r in rows if int(r["epoch"]) < epoch)

    def log_epoch(self, row: dict[str, Any]) -> None:
        row = {"wall_s": round(time.time() - self._t0, 1), **row}
        if self._fields is None:
            self._fields = list(row)
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, self._fields).writeheader()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, self._fields, extrasaction="ignore").writerow(row)

    @staticmethod
    def info(msg: str) -> None:
        print(msg, flush=True)

    @staticmethod
    def progress(msg: str) -> None:
        sys.stdout.write("\r" + msg + " " * 8)
        sys.stdout.flush()


def fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"
