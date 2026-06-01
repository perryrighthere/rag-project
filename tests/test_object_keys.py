from rag_project.storage import (
    build_http_object_url,
    build_parsed_image_key,
    build_parsed_json_key,
    build_parsed_markdown_key,
    build_raw_object_key,
    rewrite_relative_image_paths,
)


def test_object_key_conventions() -> None:
    assert build_raw_object_key("kb x", "doc/1", "../合同.pdf") == "raw/kb_x/doc_1/合同.pdf"
    assert build_parsed_markdown_key("kb", "doc", "合同.pdf") == "parsed/kb/doc/markdown/合同.md"
    assert build_parsed_image_key("kb", "doc", "figures/a b.png") == "parsed/kb/doc/images/figures/a_b.png"
    assert build_parsed_json_key("kb", "doc", "合同.pdf", "middle") == "parsed/kb/doc/json/合同_middle.json"


def test_http_url_quotes_path_segments() -> None:
    url = build_http_object_url("localhost:9000", "rag bucket", "parsed/kb/doc/images/a b.png")
    assert url == "http://localhost:9000/rag%20bucket/parsed/kb/doc/images/a%20b.png"


def test_rewrite_relative_image_paths() -> None:
    text = """![](images/a.png)
![](./images/nested/b.png)
{"img_path": "images/c.png"}
<img src='images/d.png'>
![](https://example.com/images/e.png)
"""

    rewritten = rewrite_relative_image_paths(text, lambda name: f"https://minio.local/{name}")

    assert "![](https://minio.local/a.png)" in rewritten
    assert "![](https://minio.local/nested/b.png)" in rewritten
    assert '"img_path": "https://minio.local/c.png"' in rewritten
    assert "<img src='https://minio.local/d.png'>" in rewritten
    assert "![](https://example.com/images/e.png)" in rewritten

