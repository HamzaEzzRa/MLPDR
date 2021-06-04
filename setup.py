import sys
from cx_Freeze import setup, Executable

base = None
if sys.platform == "win32":
    base = "Win32GUI"

options = {"build_exe": {"includes": "atexit", "include_files": ["./weights", "./tmp", "./classes-detection.names", "./classes-ocr.names"]}}

executables = [Executable("main.py", base=base, target_name="mlpdr.exe")]

setup (
    name="mlpdr",
    version="1.0",
    description="A moroccan license plate detector & reader",
    options=options,
    executables=executables,
)
