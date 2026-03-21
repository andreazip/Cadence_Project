"""Compatibility wrapper for the modular DTC runner in dtc/.

Keeps legacy commands like `python run_dtc.py` working.
"""

from dtc.run_dtc import main


if __name__ == "__main__":
    main()
