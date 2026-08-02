import os
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.mkdtemp(prefix="ofuscador-tests-"))
os.environ["OFUSCADOR_DATA_DIR"] = str(TEST_ROOT / "data")
os.environ["OFUSCADOR_TEMP_DIR"] = str(TEST_ROOT / "temp")
