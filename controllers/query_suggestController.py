from services.QueryRefinementService.QuerySuggestion import suggest_similar_queries

class QuerySuggestController:
    def execute(self, query):
        return suggest_similar_queries(query)
