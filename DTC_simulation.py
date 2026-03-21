"""Compatibility entry point for the modular DTC simulation flow.

This wrapper keeps older commands working while delegating all logic to run_dtc.py.
"""

from run_dtc import main


if __name__ == "__main__":
    main()
