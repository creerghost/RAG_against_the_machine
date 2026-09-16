from .minimal_search_results import MinimalSearchResults


class MinimalAnswer(MinimalSearchResults):
    """Store ranked source locations and a generated answer."""

    answer: str
