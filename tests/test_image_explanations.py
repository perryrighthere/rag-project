import asyncio

import pytest

from rag_project.parsers.image_explanations import (
    ImageAsset,
    ImageExplanationConfig,
    ImageExplanationGenerator,
)


class StubImageExplanationGenerator(ImageExplanationGenerator):
    def _create_client(self):
        return object()

    async def _request_image_explanation(self, client, *, prompt: str, asset: ImageAsset) -> str:
        return f"这张图片展示了 {asset.object_key}"


@pytest.mark.asyncio
async def test_image_explanation_writes_back_markdown_and_builds_chunks() -> None:
    generator = StubImageExplanationGenerator(
        ImageExplanationConfig(enabled=True, base_url="http://vlm.local/v1", model="vlm-model")
    )
    asset = ImageAsset(
        image_url="http://minio/rag/parsed/kb/doc/images/a.png",
        object_key="parsed/kb/doc/images/a.png",
        content=b"image-bytes",
    )

    result = await generator.enrich_markdown(
        "上文\n![](http://minio/rag/parsed/kb/doc/images/a.png)\n下文\n",
        kb_id="kb",
        document_id="doc",
        image_assets=[asset],
        language="ch",
    )

    assert "> 图片解释：这张图片展示了 parsed/kb/doc/images/a.png" in result.markdown_text
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.chunk_index == 0
    assert chunk.text == "这张图片展示了 parsed/kb/doc/images/a.png"
    assert chunk.image_object_key == "parsed/kb/doc/images/a.png"
    assert chunk.metadata["chunk_type"] == "image_explanation"
    assert chunk.metadata["kb_id"] == "kb"
    assert chunk.metadata["document_id"] == "doc"


@pytest.mark.asyncio
async def test_existing_image_explanation_is_not_duplicated() -> None:
    generator = StubImageExplanationGenerator(
        ImageExplanationConfig(enabled=True, base_url="http://vlm.local/v1", model="vlm-model")
    )

    result = await generator.enrich_markdown(
        "![](http://minio/rag/parsed/kb/doc/images/a.png)\n> 图片解释：已有说明\n",
        kb_id="kb",
        document_id="doc",
        image_assets=[
            ImageAsset(
                image_url="http://minio/rag/parsed/kb/doc/images/a.png",
                object_key="parsed/kb/doc/images/a.png",
                content=b"image-bytes",
            )
        ],
        language="ch",
    )

    assert result.markdown_text.count("> 图片解释：") == 1
    assert result.chunks == []


@pytest.mark.asyncio
async def test_image_explanations_run_with_bounded_concurrency() -> None:
    class ConcurrentGenerator(StubImageExplanationGenerator):
        active = 0
        max_active = 0

        async def _request_image_explanation(self, client, *, prompt: str, asset: ImageAsset) -> str:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return asset.object_key

    generator = ConcurrentGenerator(
        ImageExplanationConfig(
            enabled=True,
            base_url="http://vlm.local/v1",
            model="vlm-model",
            concurrency=2,
        )
    )
    assets = [
        ImageAsset(image_url=f"http://minio/{index}.png", object_key=f"{index}.png", content=b"image")
        for index in range(4)
    ]

    result = await generator.enrich_markdown(
        "\n".join(f"![](http://minio/{index}.png)" for index in range(4)),
        kb_id="kb",
        document_id="doc",
        image_assets=assets,
        language="ch",
    )

    assert generator.max_active == 2
    assert [chunk.text for chunk in result.chunks] == ["0.png", "1.png", "2.png", "3.png"]

