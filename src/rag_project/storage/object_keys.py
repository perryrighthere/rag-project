import re
from pathlib import PurePosixPath
from urllib.parse import quote


_UNSAFE_OBJECT_CHARS = re.compile(r"[^\w._=-]+", re.UNICODE)


def sanitize_object_part(value: str) -> str:
    """Make one object-key path segment stable without flattening extensions."""

    parts = [part for part in PurePosixPath(value.strip()).parts if part not in {"", ".", ".."}]
    cleaned = "_".join(parts)
    cleaned = _UNSAFE_OBJECT_CHARS.sub("_", cleaned).strip("._")
    return cleaned or "file"


def build_raw_object_key(kb_id: str, document_id: str, filename: str) -> str:
    return "/".join(["raw", sanitize_object_part(kb_id), sanitize_object_part(document_id), sanitize_object_part(filename)])


def build_parsed_markdown_key(kb_id: str, document_id: str, filename: str) -> str:
    stem = PurePosixPath(sanitize_object_part(filename)).stem or "document"
    return "/".join(["parsed", sanitize_object_part(kb_id), sanitize_object_part(document_id), "markdown", f"{stem}.md"])


def build_parsed_image_key(kb_id: str, document_id: str, image_name: str) -> str:
    image_path = PurePosixPath(image_name)
    parts = [sanitize_object_part(part) for part in image_path.parts if part not in {"", "."}]
    return "/".join(["parsed", sanitize_object_part(kb_id), sanitize_object_part(document_id), "images", *parts])


def build_parsed_json_key(kb_id: str, document_id: str, filename: str, kind: str) -> str:
    stem = PurePosixPath(sanitize_object_part(filename)).stem or "document"
    kind_part = sanitize_object_part(kind)
    return "/".join(["parsed", sanitize_object_part(kb_id), sanitize_object_part(document_id), "json", f"{stem}_{kind_part}.json"])


def normalize_http_endpoint(endpoint: str, secure: bool | None = None) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    scheme = "https" if secure else "http"
    return f"{scheme}://{endpoint}"


def build_http_object_url(endpoint: str, bucket: str, object_key: str, *, secure: bool | None = None) -> str:
    quoted_key = "/".join(quote(part, safe="") for part in object_key.strip("/").split("/"))
    return f"{normalize_http_endpoint(endpoint, secure)}/{quote(bucket, safe='')}/{quoted_key}"


_RELATIVE_IMAGE_PATTERN = re.compile(
    r"""(?P<prefix>["'(])(?:\./)?images/(?P<name>[^"')]+)""",
)


def rewrite_relative_image_paths(text: str, image_url_for_name) -> str:
    """Rewrite Markdown or JSON relative image references to object URLs.

    Handles references produced by MinerU such as `![](images/a.png)`,
    `![](./images/a.png)`, `"img_path": "images/a.png"` and HTML-ish
    `src='images/a.png'`. Absolute URLs are intentionally left untouched.
    """

    def replace(match: re.Match[str]) -> str:
        image_name = match.group("name").lstrip("/")
        return f"{match.group('prefix')}{image_url_for_name(image_name)}"

    return _RELATIVE_IMAGE_PATTERN.sub(replace, text)
