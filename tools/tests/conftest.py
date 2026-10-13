import pathlib
import sys

# Make tools/*.py importable as top-level modules from tools/tests/*.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
