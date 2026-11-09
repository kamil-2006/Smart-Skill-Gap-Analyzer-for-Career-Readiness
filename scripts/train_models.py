from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skillgap_analyzer.models import train_and_save


def main() -> None:
    train_and_save()
    print("Saved models to ./models")


if __name__ == "__main__":
    main()
