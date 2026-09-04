#!/usr/bin/env python3
"""Compatibility wrapper for the unified synthetic graph generator."""

import sys
from generate_synthetic import main


if __name__ == "__main__":
    sys.argv.insert(1, "erdos-renyi")
    if "--directed" not in sys.argv and "--undirected" not in sys.argv:
        sys.argv.insert(2, "--directed")
    main()
