"""Create a small synthetic PDF, including a diagram (requires dev dependencies)."""

import argparse
import io
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def create(path: Path):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(600, 800))
    pdf.setTitle("Synthetic security overview — not the supplied report")
    pdf.drawString(50, 740, "Synthetic service security overview")
    pdf.drawString(50, 700, "The service is hosted on Google Cloud Platform (GCP).")
    pdf.drawString(50, 675, "Affected parties are notified of incidents without undue delay.")
    pdf.drawString(50, 630, "The following diagram shows the service components.")
    picture = Image.new("RGB", (1000, 550), "white")
    draw = ImageDraw.Draw(picture)
    draw.rectangle((25, 25, 975, 525), outline="#203c33", width=5)
    draw.text((60, 60), "Google Cloud Platform (GCP)", fill="black", font_size=45)
    for x, label in [(90, "Redis"), (560, "MongoDB")]:
        draw.rectangle((x, 220, x + 330, 380), outline="#203c33", width=4)
        draw.text((x + 20, 270), label, fill="black", font_size=43)
    pdf.drawImage(ImageReader(picture), 50, 260, width=500, height=275)
    pdf.save()
    with path.open("xb") as file:
        file.write(buffer.getvalue())
    print(f"Created {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, nargs="?", default=Path("examples/document.pdf"))
    create(parser.parse_args().target)
