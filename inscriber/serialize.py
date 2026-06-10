"""(De)serialization for the OCR boundary, designed ONCE (PLAN M1b key note).

``OcrPageResult`` / ``Region`` JSON shapes live here so the **cache** (M1b) and the
**bundle** (M2) share one format — the bundle is a superset that never has to
migrate the cache's representation.
"""

from __future__ import annotations

from inscriber.models import Figure, OcrPageResult, Region


def region_to_dict(r: Region) -> dict:
    return {"label": r.label, "bbox_norm": list(r.bbox_norm), "text": r.text}


def figure_to_dict(f: Figure) -> dict:
    return {
        "id": f.id,
        "page": f.page,
        "bbox_norm": list(f.bbox_norm),
        "crop_path": f.crop_path,
        "caption": f.caption,
    }


def figure_from_dict(d: dict) -> Figure:
    b = d["bbox_norm"]
    return Figure(
        id=d["id"],
        page=d["page"],
        bbox_norm=(b[0], b[1], b[2], b[3]),
        crop_path=d.get("crop_path"),
        caption=d.get("caption"),
    )


def region_from_dict(d: dict) -> Region:
    b = d["bbox_norm"]
    return Region(label=d["label"], bbox_norm=(b[0], b[1], b[2], b[3]), text=d.get("text"))


def ocr_page_result_to_dict(r: OcrPageResult) -> dict:
    d = {
        "page_number": r.page_number,
        "markdown": r.markdown,
        "regions": [region_to_dict(x) for x in r.regions],
    }
    # Additive: written only when set, so clean pages keep the original shape
    # (old cache entries / bundles read as not-truncated, DESIGN §8.5/§8.6).
    if r.truncated:
        d["truncated"] = True
    return d


def ocr_page_result_from_dict(d: dict) -> OcrPageResult:
    return OcrPageResult(
        page_number=d["page_number"],
        markdown=d["markdown"],
        regions=[region_from_dict(x) for x in d.get("regions", [])],
        truncated=bool(d.get("truncated", False)),
    )
