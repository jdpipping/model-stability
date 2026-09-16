"""Create a Reveal print edition without changing the live presentation."""

import argparse
from html.parser import HTMLParser
from pathlib import Path


class RevealGroups(HTMLParser):
    def __init__(self):
        super().__init__()
        self.slides = []
        self.current = None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        classes = attrs.get("class", "").split()
        if tag == "section" and "slide" in classes and "level2" in classes:
            self.current = {"id": attrs.get("id"), "indexed": set(), "unindexed": 0}
            self.slides.append(self.current)
        if "fragment" in classes and self.current is not None:
            if "data-fragment-index" in attrs:
                self.current["indexed"].add(attrs["data-fragment-index"])
            else:
                self.current["unindexed"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root = Path(__file__).resolve().parents[3] / "presentations" / "cassis2026"
    source = (root / "index.html").read_text()
    assert source.count("'pdfSeparateFragments': false") == 1
    assert source.count("Reveal.initialize({") == 1
    print_html = source.replace("'pdfSeparateFragments': false", "'pdfSeparateFragments': true")
    print_html = print_html.replace("Reveal.initialize({", "Reveal.initialize({\n        view: 'print',", 1)
    (root / "PRINT-reveals.html").write_text(print_html)

    groups = RevealGroups()
    groups.feed(source)
    reveals = sum(len(slide["indexed"]) + slide["unindexed"] for slide in groups.slides)
    print(f"Print edition prepared: {len(groups.slides)} slides + {reveals} reveal states = {len(groups.slides) + reveals} expected pages.")
    print("Open PRINT-reveals.html in a browser, then Print > Save as PDF.")


if __name__ == "__main__":
    main()
