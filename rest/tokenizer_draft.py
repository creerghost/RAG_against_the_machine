import re


class Tokenizer:
    camel_regex = re.compile(
        r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')

    def __init__(self, stopwords: frozenset[str]) -> None:
        self.stopwords = stopwords

    def tokenize(self, text: str) -> list[str]:
        result = []
        for match in re.finditer(r'\w+', text):
            token = match.group()
            parts = [c for p in token.split("_")
                     for c in self.camel_regex.split(p) if c]
            if parts != [token]:  # also "__init__" -> "init"
                result.extend(p.lower() for p in parts)
            result.append(token.lower())
        return [t for t in result if self._keep(t)]

    def _keep(self, token: str) -> bool:
        return (len(token) > 1 and token not in self.stopwords
                and any(c.isalnum() for c in token))


if __name__ == "__main__":
    tokenizer = Tokenizer({"the", "is", "in", "a", "does", "how", "of"})
    res = tokenizer.tokenize("what is 2 + 2? and how to find a find_me  ___"
                             "      HelloMyBeautifulWorld")
    print(res)
