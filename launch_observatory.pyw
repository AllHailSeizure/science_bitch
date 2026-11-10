"""Windowed executable entry point; no command shell is needed."""
from simulations.star_system.__main__ import main

if __name__ == '__main__':
    raise SystemExit(main())
