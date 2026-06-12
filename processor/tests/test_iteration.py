from pathlib import Path

from process import Processor

class IterTestProcessor(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_handlers = {
            ".mhd": str,
            ".mha": str,
        }

    def process_item(self, *args):
        return args

    def format_output(self, data):
        pass


TEST_DATA_DIR = Path(__file__).resolve().parent / "data"

def test_iterate_default():
    mhds_dir = TEST_DATA_DIR / "mhds"
    processor = IterTestProcessor(mhds_dir, mhds_dir)

