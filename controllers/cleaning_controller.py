from services.data_loader import DataLoaderService
from services.text_cleaner import TextCleaningService
from services.data_saver import DataSaverService

class CleaningController:
    def __init__(self):
        self.loader = DataLoaderService()
        self.cleaner = TextCleaningService()
        self.saver = DataSaverService()

    def clean_data(self, input_path: str, output_path: str):
        df = self.loader.load(input_path)
        processed_rows = self.cleaner.clean(df)
        self.saver.save(processed_rows, output_path)
        return {"message": f"تم تنظيف البيانات وحفظها في: {output_path}"}
