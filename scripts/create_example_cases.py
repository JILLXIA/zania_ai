"""Generate synthetic PDF and boundary fixtures locally; never overwrite existing files."""

import argparse
import io
import json
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw
from pypdf import PdfWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def save_new(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as file:
            file.write(data)
    except FileExistsError:
        return False
    return True


def text_pdf() -> bytes:
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=(600, 800), invariant=1)
    pdf.setTitle("Cedar Metrics - Synthetic Security Overview")
    sections = [
        (
            "Hosting and data protection",
            [
                (
                    "Production hosting",
                    "Cedar Metrics production is hosted on Microsoft Azure "
                    "in the West Europe region.",
                ),
                (
                    "Encryption",
                    "Customer data uses AES-256 encryption at rest and TLS 1.3 in transit.",
                ),
                ("Backups", "Backups run every 24 hours and are retained for 30 days."),
            ],
        ),
        (
            "Incident response and access removal",
            [
                (
                    "Security incidents",
                    "Affected customers are notified by email without undue delay. "
                    "The Security Officer coordinates notifications.",
                ),
                (
                    "Employee offboarding",
                    "Access for departing employees is removed within 24 hours after termination. "
                    "This deadline applies to offboarding, not incident notifications.",
                ),
                (
                    "Scope",
                    "This overview does not supply a numeric incident-notification SLA "
                    "or an ISO 27001 certificate number.",
                ),
            ],
        ),
    ]
    for number, (title, items) in enumerate(sections, start=1):
        pdf.setFillColorRGB(0.15, 0.32, 0.27)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(50, 751, "CEDAR METRICS / FICTIONAL TEST DOCUMENT")
        pdf.setFont("Helvetica-Bold", 21)
        pdf.drawString(50, 707, title)
        pdf.setStrokeColorRGB(0.75, 0.81, 0.77)
        pdf.line(50, 686, 550, 686)
        y = 637
        for heading, paragraph in items:
            pdf.setFillColorRGB(0.12, 0.20, 0.18)
            pdf.setFont("Helvetica-Bold", 13)
            pdf.drawString(50, y, heading)
            pdf.setFont("Helvetica", 11)
            y -= 27
            for line in wrap(paragraph, width=83):
                pdf.drawString(50, y, line)
                y -= 18
            y -= 37
        pdf.setFont("Helvetica", 9)
        pdf.setFillColorRGB(0.4, 0.45, 0.42)
        pdf.drawString(50, 44, "Synthetic example. No real company assurances or credentials.")
        pdf.drawRightString(550, 44, f"Page {number}")
        pdf.showPage()
    pdf.save()
    return output.getvalue()


def scanned_pdf() -> bytes:
    # All words are pixels. PdfReader.extract_text() should return an empty string.
    picture = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(picture)
    draw.text((90, 100), "HARBOR LABS", fill="#244a3d", font_size=58)
    draw.text((90, 182), "Synthetic scanned security policy", fill="black", font_size=36)
    draw.line((90, 265, 1110, 265), fill="#80978c", width=3)
    lines = [
        "Cloud provider: Amazon Web Services (AWS)",
        "Hosting region: Ireland",
        "Backup retention: 45 days",
        "Administrator MFA: Required",
    ]
    for i, line in enumerate(lines):
        draw.text((90, 370 + i * 150), line, fill="black", font_size=38)
    draw.text(
        (90, 1420), "Fictional data for testing image extraction.", fill="#555555", font_size=28
    )
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=(600, 800), invariant=1)
    pdf.setTitle("Harbor Labs - Image-only synthetic policy")
    pdf.drawImage(ImageReader(picture), 0, 0, width=600, height=800)
    pdf.showPage()
    pdf.save()
    picture.close()
    return output.getvalue()


def generate(output: Path, include_large: bool = False) -> list[Path]:
    pdf_dir, data_dir = output / "pdf", output / "example-fixtures"
    native = text_pdf()
    encrypted = io.BytesIO()
    writer = PdfWriter(clone_from=io.BytesIO(native))
    writer.encrypt("sample-password")
    writer.write(encrypted)
    blank = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    writer.write(blank)
    fixtures = {
        pdf_dir / "text-only.pdf": native,
        pdf_dir / "scanned-policy.pdf": scanned_pdf(),
        pdf_dir / "encrypted.pdf": encrypted.getvalue(),
        pdf_dir / "blank.pdf": blank.getvalue(),
        data_dir / "questions-invalid-utf8.json": b"\xff\xfe\xfd",
        data_dir / "document-empty-bytes.json": b"",
        data_dir / "corrupt.pdf": b"%PDF-1.7\nThis is deliberately corrupt, not a readable PDF.\n",
        data_dir / "not-a-pdf.pdf": b"This is plain text with a misleading extension.\n",
    }
    nested = "valid text at excessive depth"
    for _ in range(34):
        nested = [nested]
    values = {
        "questions-limit-30.json": ["Which cloud provider hosts the service?"] * 30,
        "questions-too-many-31.json": ["Which cloud provider hosts the service?"] * 31,
        "questions-too-long.json": ["x" * 2001],
        "questions-too-large.json": ["x" * (256 * 1024)],
        "document-too-deep.json": nested,
    }
    for name, value in values.items():
        fixtures[data_dir / name] = json.dumps(value, indent=2).encode() + b"\n"
    if include_large:
        fixtures[data_dir / "document-too-large.json"] = (
            b'{"text":"' + b"x" * (20 * 1024 * 1024) + b'"}'
        )
    for path, contents in fixtures.items():
        save_new(path, contents)
    return list(fixtures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--large", action="store_true", help="Also generate a >20 MiB JSON upload")
    args = parser.parse_args()
    files = generate(args.output, args.large)
    print(f"Prepared {len(files)} fixture paths under {args.output}; existing files were kept.")
