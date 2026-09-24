import hashlib
import io
import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.config import Settings
from app.main import create_app
from app.models import AnswerOutput, AppError, Evidence, VisionOutput
from app.service import QAService


class FakeEmbeddings:
    """Deterministic lexical vectors; real chunking, FAISS and MMR still run."""

    def __init__(self):
        self.document_calls = 0

    def token_count(self, text):
        return max(1, len(text) // 4)

    def vectors(self, texts):
        result = np.zeros((len(texts), 128), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                column = int.from_bytes(hashlib.sha256(word.encode()).digest()[:2]) % 128
                result[row, column] += 1
        return result

    def embed_documents(self, texts):
        self.document_calls += 1
        return self.vectors(texts)

    def embed_queries(self, texts):
        return self.vectors(texts)


class FakeProvider:
    def __init__(self):
        self.answers = []
        self.images = []
        self.visual_output = VisionOutput(
            kind="content", observations=["The diagram shows Redis inside GCP."]
        )

    async def answer(self, question, chunks):
        self.answers.append((question, chunks))
        if "failure" in question:
            raise AppError(503, "provider_unavailable", "The model provider is unavailable.")
        phrase = (
            "without undue delay"
            if "SLA" in question
            else "Redis"
            if "diagram" in question
            else "GCP"
        )
        chunk = next((c for c in chunks if phrase in c.text), None)
        if chunk is None or "unsupported" in question:
            return AnswerOutput(
                status="not_found", answer="ignored", evidence=[], missing_details=[]
            )
        return AnswerOutput(
            status="partial" if "SLA" in question else "answered",
            answer=f"The document states: {phrase}.",
            missing_details=["a numeric notification SLA"] if "SLA" in question else [],
            evidence=[Evidence(chunk_id=chunk.id, excerpt=phrase)],
        )

    async def analyze_image(self, visual, budget):
        self.images.append(visual)
        return self.visual_output


def pdf_bytes(*, text="The service is hosted on GCP.", image=False, vector=False, pages=1):
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=(600, 800))
    for _ in range(pages):
        if text:
            pdf.drawString(50, 750, text)
        if image:
            picture = Image.new("RGB", (600, 400), "white")
            draw = ImageDraw.Draw(picture)
            draw.rectangle((30, 30, 570, 370), outline="black", width=4)
            draw.text((100, 100), "GCP: Redis", fill="black", font_size=42)
            pdf.drawImage(ImageReader(picture), 80, 250, 440, 300)
        if vector:
            pdf.rect(80, 300, 400, 200)
            pdf.drawString(130, 400, "Redis")
        pdf.showPage()
    pdf.save()
    return output.getvalue()


def uploads(questions=None, document=None, *, kind="json"):
    questions = questions if questions is not None else ["Which cloud provider?"]
    document = document if document is not None else [{"answer": "Hosted on GCP."}]
    return {
        "questions": ("questions.json", json.dumps(questions).encode(), "application/json"),
        "document": (
            f"document.{kind}",
            json.dumps(document).encode() if kind == "json" else document,
            f"application/{kind}",
        ),
    }


@pytest.fixture
def settings():
    return Settings(_env_file=None, OPENAI_API_KEY="")


@pytest.fixture
def service(settings):
    return QAService(settings, FakeEmbeddings(), FakeProvider())


@pytest.fixture
def client(settings, service):
    with TestClient(create_app(settings, service)) as client:
        yield client
