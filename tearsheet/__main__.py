"""Enable ``python -m tearsheet``."""

import sys

from tearsheet.cli import main

if __name__ == "__main__":
    sys.exit(main())
