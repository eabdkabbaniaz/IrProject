from services.Proccessing.data_loader import DataLoaderService
from services.Proccessing.text_cleaner import TextCleaningService
from services.Proccessing.data_saver import DataSaverService

import logging

logging.basicConfig(
    level=logging.DEBUG,                   # أظهر كل شيء من DEBUG فأعلى
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class ServiceProccessing:
    def __init__(self):
        self.loader = DataLoaderService()
        self.cleaner = TextCleaningService()
        self.saver = DataSaverService()

    def clean_data(self, input_path: str, output_path: str):
        logging.debug("clean_data() => start")

        df = self.loader.load(input_path)
        processed_rows = self.cleaner.clean(df)

        logging.debug(f"Number of processed rows: {len(processed_rows)}")
        self.saver.save(processed_rows, output_path)

        logging.info(f"تم الحفظ في {output_path}")

        return processed_rows
        
