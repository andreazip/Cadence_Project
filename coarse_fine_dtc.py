"""Compatibility wrapper for coarse_fine package.

Imports core APIs and runs the package runner when executed as a script.
"""

from coarse_fine.coarse_fine_core import *  # noqa: F401,F403
from coarse_fine.run_coarse_fine import main


if __name__ == "__main__":
    main()
