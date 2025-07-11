from services.Proccessing.ServiceProccessing import ServiceProccessing

service = ServiceProccessing()

class ProcessingController:
    def execute(self,input, output):
        return service.clean_data(input, output)
