from services.tfIdfProccessingInvertedIndex.ViewtfIdfMatrix import viewMatrix

class viewTfIdfMatrixController:
    def preview(self, input_path: str, limit: int = 100):
        return viewMatrix(input_path, limit)
