from services.QueryRefinementService.AutoCompleteQuery import autocomplete

class AutoCompletController:
    def execute(self, query):
        return autocomplete(query)
