from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class ConversionPlan:
    pdf_path: Path
    output_dir: Path
    pages: list[int]
    prefix: str
    trim_top_px: float | None
    trim_bottom_px: float | None
    trim_top_ratio: float | None
    trim_bottom_ratio: float | None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _require_tool(name: str, hint: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Missing required tool '{name}'. {hint}")


def _page_count(pdf_path: Path) -> int:
    result = _run(["pdfinfo", str(pdf_path)])
    if result.returncode != 0:
        raise RuntimeError(f"pdfinfo failed for {pdf_path}:\n{result.stderr.strip()}")
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not parse page count from pdfinfo output for {pdf_path}")


def _parse_page_spec(spec: str, max_page: int) -> list[int]:
    pages: set[int] = set()
    for raw_chunk in spec.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if start > end:
                raise ValueError(f"Invalid range '{chunk}': start must be <= end")
            pages.update(range(start, end + 1))
            continue
        pages.add(int(chunk))

    if not pages:
        raise ValueError("No pages selected. Use --pages like '1,3,5-7'.")

    invalid = sorted(page for page in pages if page < 1 or page > max_page)
    if invalid:
        raise ValueError(f"Pages out of bounds (1..{max_page}): {invalid}")

    return sorted(pages)


def _parse_pages(raw: object, max_page: int) -> list[int]:
    if isinstance(raw, str):
        return _parse_page_spec(raw, max_page=max_page)
    if isinstance(raw, list):
        pages: set[int] = set()
        for item in raw:
            if not isinstance(item, int):
                raise ValueError("Config 'pages' list must contain integers only")
            pages.add(item)
        if not pages:
            raise ValueError("Config 'pages' list is empty")
        invalid = sorted(page for page in pages if page < 1 or page > max_page)
        if invalid:
            raise ValueError(f"Pages out of bounds (1..{max_page}): {invalid}")
        return sorted(pages)
    raise ValueError("Pages must be either a string like '1,3-4' or an integer list")


def _parse_ratio(value: object, field: str) -> float | None:
    if value is None:
        return None
    ratio = float(value)
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"{field} must be in [0, 1)")
    return ratio


def _parse_px(value: object, field: str) -> float | None:
    if value is None:
        return None
    px = float(value)
    if px < 0:
        raise ValueError(f"{field} must be >= 0")
    return px


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config must be a JSON object")
    return payload


def _resolve_value(cli_value: object, config_value: object, default_value: object) -> object:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default_value


def _build_plan(args: argparse.Namespace) -> ConversionPlan:
    config: dict = {}
    if args.config is not None:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        config = _load_config(config_path)

    trim_cfg = config.get("trim", {})
    if trim_cfg is not None and not isinstance(trim_cfg, dict):
        raise ValueError("Config field 'trim' must be an object when provided")

    pdf_value = _resolve_value(args.pdf, config.get("pdf"), None)
    if pdf_value is None:
        raise ValueError("Missing PDF input. Provide --pdf or config 'pdf'.")
    pdf_path = Path(str(pdf_value)).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    _require_tool("pdfinfo", "Install poppler (macOS: `brew install poppler`).")
    _require_tool("pdftocairo", "Install poppler (macOS: `brew install poppler`).")

    pages_value = _resolve_value(args.pages, config.get("pages"), None)
    if pages_value is None:
        raise ValueError("Missing page selection. Provide --pages or config 'pages'.")
    max_page = _page_count(pdf_path)
    pages = _parse_pages(pages_value, max_page=max_page)

    out_value = _resolve_value(args.out, config.get("out"), "out/svg_pages")
    output_dir = Path(str(out_value)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_value = _resolve_value(args.prefix, config.get("prefix"), "page")
    trim_top_px = _parse_px(_resolve_value(args.trim_top_px, trim_cfg.get("top_px"), None), "top_px")
    trim_bottom_px = _parse_px(_resolve_value(args.trim_bottom_px, trim_cfg.get("bottom_px"), None), "bottom_px")
    trim_top_ratio = _parse_ratio(_resolve_value(args.trim_top_ratio, trim_cfg.get("top_ratio"), None), "top_ratio")
    trim_bottom_ratio = _parse_ratio(
        _resolve_value(args.trim_bottom_ratio, trim_cfg.get("bottom_ratio"), None),
        "bottom_ratio",
    )

    return ConversionPlan(
        pdf_path=pdf_path,
        output_dir=output_dir,
        pages=pages,
        prefix=str(prefix_value),
        trim_top_px=trim_top_px,
        trim_bottom_px=trim_bottom_px,
        trim_top_ratio=trim_top_ratio,
        trim_bottom_ratio=trim_bottom_ratio,
    )


def _parse_css_length(value: str | None) -> tuple[float, str] | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("px"):
        return float(text[:-2]), "px"
    if text.endswith("pt"):
        return float(text[:-2]), "pt"
    return float(text), ""


def _format_length(value: float, unit: str) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{text}{unit}" if unit else text


def _qname(tag: str, namespace: str | None) -> str:
    if namespace:
        return f"{{{namespace}}}{tag}"
    return tag


def _detect_namespace(tag: str) -> str | None:
    if tag.startswith("{") and "}" in tag:
        return tag[1 : tag.find("}")]
    return None


def _crop_svg(svg_path: Path, page: int, plan: ConversionPlan) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = _detect_namespace(root.tag)

    width_spec = _parse_css_length(root.attrib.get("width"))
    height_spec = _parse_css_length(root.attrib.get("height"))
    view_box_text = root.attrib.get("viewBox")
    if not view_box_text:
        return

    vb_vals = view_box_text.replace(",", " ").split()
    if len(vb_vals) != 4:
        return
    vb_x, vb_y, vb_w, vb_h = (float(v) for v in vb_vals)
    if vb_h <= 0:
        return

    top_trim = plan.trim_top_px if plan.trim_top_px is not None else 0.0
    bottom_trim = plan.trim_bottom_px if plan.trim_bottom_px is not None else 0.0
    if plan.trim_top_ratio is not None:
        top_trim = vb_h * plan.trim_top_ratio
    if plan.trim_bottom_ratio is not None:
        bottom_trim = vb_h * plan.trim_bottom_ratio

    if top_trim <= 0 and bottom_trim <= 0:
        return
    if top_trim + bottom_trim >= vb_h:
        raise ValueError(
            f"Trim too aggressive for page {page}: top({top_trim}) + bottom({bottom_trim}) >= page height({vb_h})"
        )

    cropped_vb_y = vb_y + top_trim
    cropped_vb_h = vb_h - top_trim - bottom_trim
    root.attrib["viewBox"] = f"{vb_x:.3f} {cropped_vb_y:.3f} {vb_w:.3f} {cropped_vb_h:.3f}"

    if height_spec is not None:
        height_value, height_unit = height_spec
        scale = cropped_vb_h / vb_h
        root.attrib["height"] = _format_length(height_value * scale, height_unit)
    if width_spec is not None:
        width_value, width_unit = width_spec
        root.attrib["width"] = _format_length(width_value, width_unit)

    clip_id = f"page_clip_{page:03d}"
    children = list(root)
    for child in children:
        root.remove(child)

    defs = ET.Element(_qname("defs", ns))
    clip_path = ET.SubElement(defs, _qname("clipPath", ns), {"id": clip_id})
    ET.SubElement(
        clip_path,
        _qname("rect", ns),
        {
            "x": f"{vb_x:.3f}",
            "y": f"{cropped_vb_y:.3f}",
            "width": f"{vb_w:.3f}",
            "height": f"{cropped_vb_h:.3f}",
        },
    )
    wrapped = ET.Element(
        _qname("g", ns),
        {
            "clip-path": f"url(#{clip_id})",
        },
    )
    for child in children:
        wrapped.append(child)

    root.append(defs)
    root.append(wrapped)
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def _convert_page_to_svg(plan: ConversionPlan, page: int) -> Path:
    base_name = f"{plan.prefix}-{page:03d}"
    out_prefix = plan.output_dir / base_name
    result = _run(
        [
            "pdftocairo",
            "-svg",
            "-f",
            str(page),
            "-l",
            str(page),
            str(plan.pdf_path),
            str(out_prefix),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pdftocairo failed for page {page}:\n{result.stderr.strip() or result.stdout.strip()}"
        )

    expected_svg = out_prefix.with_suffix(".svg")
    extensionless = out_prefix
    if extensionless.exists() and not expected_svg.exists():
        extensionless.rename(expected_svg)
    if not expected_svg.exists():
        raise RuntimeError(f"Expected SVG output missing for page {page}: {expected_svg}")
    _crop_svg(expected_svg, page=page, plan=plan)
    return expected_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-pages-to-svg",
        description="Extract selected PDF pages and convert them to SVG files.",
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file for extractor inputs.")
    parser.add_argument("--pdf", default=None, help="Path to source PDF file.")
    parser.add_argument(
        "--pages",
        default=None,
        help="Comma-separated pages/ranges, e.g. '1,3,5-7'.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for generated SVG files (default: out/svg_pages).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: page -> page-001.svg).",
    )
    parser.add_argument(
        "--trim-top-px",
        type=float,
        default=None,
        help="Trim this many units from top of each page in SVG viewBox space.",
    )
    parser.add_argument(
        "--trim-bottom-px",
        type=float,
        default=None,
        help="Trim this many units from bottom of each page in SVG viewBox space.",
    )
    parser.add_argument(
        "--trim-top-ratio",
        type=float,
        default=None,
        help="Trim top ratio of page height (0 <= ratio < 1). Overrides --trim-top-px.",
    )
    parser.add_argument(
        "--trim-bottom-ratio",
        type=float,
        default=None,
        help="Trim bottom ratio of page height (0 <= ratio < 1). Overrides --trim-bottom-px.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plan = _build_plan(args)

    print(f"PDF: {plan.pdf_path}")
    print(f"Output: {plan.output_dir}")
    print(f"Pages: {','.join(str(p) for p in plan.pages)}")
    print(
        "Trim: "
        f"top_px={plan.trim_top_px}, bottom_px={plan.trim_bottom_px}, "
        f"top_ratio={plan.trim_top_ratio}, bottom_ratio={plan.trim_bottom_ratio}"
    )

    for page in plan.pages:
        svg_path = _convert_page_to_svg(plan, page)
        print(svg_path)
    return 0
