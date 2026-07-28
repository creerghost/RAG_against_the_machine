# Project Title

This is a short introductory paragraph for the test readme. It should be easily captured by the first chunk.

## Installation

To install this project, you need to run the following commands in your terminal:

```bash
pip install -r requirements.txt
python setup.py install
```

This section is also relatively short and tests standard markdown code blocks.

## Long Section

This is a section that could theoretically contain a very long paragraph. For testing the fallback logic in your chunker, you might want to paste a massive wall of text here that exceeds the 2000 character limit without any headers intervening. 

For now, this is just a standard paragraph.

Another paragraph in the same section, separated by a double newline (`\n\n`), which tests your `_split_by_paragraph` grouping logic!

### Sub-section

Even deeper nesting to ensure your `#+` regex catches H3s correctly!
