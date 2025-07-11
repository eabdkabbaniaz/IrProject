from services.QueryRefinementService.CorrectQuery import full_correct_query

class CorrectQueryController:
    def execute(self, query):
        return full_correct_query(query)
