import re


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*")
_CAMEL_BOUNDARY_PATTERN = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z][a-z])"
)


def tokenize(text: str) -> list[str]:
    """Return normalized lexical terms, including identifier components."""
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0).replace("-", "_")
        normalized_token = token.lower()
        tokens.append(normalized_token)
        underscore_components = [
            component for component in token.split("_") if component
        ]
        if len(underscore_components) > 1:
            tokens.extend(
                component.lower() for component in underscore_components
            )
        elif underscore_components:
            camel_components = _CAMEL_BOUNDARY_PATTERN.sub(
                " ", underscore_components[0]
            ).split()
            if len(camel_components) > 1:
                tokens.extend(component.lower() for component in camel_components)
    return tokens
