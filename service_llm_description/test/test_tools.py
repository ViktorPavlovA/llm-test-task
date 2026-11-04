import pytest

from src.domain.facts import collect_missing
from src.graph.graph import _normalize_found, needs_web_research
from src.graph.research import (
    build_search_queries,
    clean_search_name,
    family_token,
    research_fields,
    score_hit,
)
from src.repository.tools import (
    PublicURLError,
    assert_public_url,
    html_to_text,
    looks_like_pdf,
    pdf_bytes_to_text,
    search_web,
)
from src.schemas import DeviceStatsModel


def test_ssrf_blocks_private_and_internal_hosts():
    """
    Reject loopback, RFC1918, metadata, and Docker-internal URLs.

    """
    blocked = [
        "http://127.0.0.1/",
        "http://localhost/admin",
        "http://[::1]/",
        "http://10.0.0.5/secret",
        "http://192.168.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://llm:8686/v1",
        "http://host.docker.internal:8686/health",
        "file:///etc/passwd",
        "ftp://example.com/file",
    ]
    for url in blocked:
        with pytest.raises(PublicURLError):
            assert_public_url(url)


def test_html_to_text_strips_scripts():
    """
    Drop script bodies and keep visible paragraph text.

    """
    text = html_to_text(
        "<html><script>alert(1)</script><p>Класс точности 1,5</p></html>"
    )
    assert "alert" not in text
    assert "Класс точности 1,5" in text


def test_search_web_rejects_empty_query():
    """
    Return a recovery hint instead of calling the search backend.

    """
    message = search_web.invoke({"query": "   "})
    assert "пустой query" in message


def test_clean_search_name_strips_variant():
    """
    Drop synthetic ``(вариант N)`` suffixes from search names.

    """
    assert (
        clean_search_name("Манометр показывающий МП-160-12 (вариант 13)")
        == "Манометр показывающий МП-160-12"
    )


def test_build_search_queries_prefer_datasheet():
    """
    Ask DuckDuckGo for the cleaned name, maker, and maker site.

    """
    device = DeviceStatsModel(
        name="Преобразователь давления ПД100-ДИ0.25-29 (вариант 380)",
        manufacturer="ОВЕН",
        specs={"выходной сигнал": "4…20 мА"},
    )
    queries = build_search_queries(device, ["тип корпуса"])
    blob = " ".join(queries)
    assert "вариант" not in blob
    assert "ПД100-ДИ0.25-29" in blob
    assert "ОВЕН" in blob
    assert any("site:owen.ru" in item for item in queries)


def test_score_hit_prefers_maker_datasheet():
    """
    Rank manufacturer datasheet pages above aggregators.

    """
    good = score_hit(
        "ПД100 паспорт",
        "https://owen.ru/product/pd100",
        "паспорт преобразователя давления",
    )
    pdf = score_hit(
        "ПД100 паспорт PDF",
        "https://owen.ru/uploads/re_pd100.pdf",
        "паспорт преобразователя давления",
    )
    bad = score_hit("купить", "https://www.startpage.com/sp/search", "оптовая цена")
    assert good > 0
    assert pdf > good
    assert bad < 0
    assert good > bad


def test_family_token_from_sku():
    """
    Collapse a modification SKU to the product line.

    """
    assert family_token("ПД100-ДИ0.25-29") == "ПД100"
    assert family_token("САУ-М611") == "САУ-М6"
    assert family_token("ДТС112-100М") == "ДТС112"
    assert family_token("МП-160") == ""


def test_research_fields_skip_power_on_analog_gauge():
    """
    Do not search for power or interfaces on a mechanical indicating gauge.

    """
    device = DeviceStatsModel(
        name="Манометр показывающий МП-160-12 (вариант 13)",
        manufacturer="WIKA",
    )
    fields = research_fields(
        device,
        ["питание", "интерфейсы", "класс точности"],
    )
    assert "питание" not in fields
    assert "интерфейсы" not in fields
    assert "класс точности" in fields


def test_research_fields_keep_power_on_transmitter():
    """
    Keep power as a research gap for electronic transmitters.

    """
    device = DeviceStatsModel(
        name="Преобразователь давления ПД100-ДИ0.25-29",
        manufacturer="ОВЕН",
    )
    fields = research_fields(device, ["питание", "тип корпуса"])
    assert fields == ["питание", "тип корпуса"]


def _tiny_pdf(text: str) -> bytes:
    """
    Build a one-page PDF with a Helvetica text stream and a valid xref.

    """
    stream = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("ascii")
    parts = [
        b"%PDF-1.4\n",
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        ),
        (
            b"4 0 obj\n<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream\nendobj\n"
        ),
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    offsets = []
    cursor = 0
    body = b""
    for part in parts:
        offsets.append(cursor)
        body += part
        cursor += len(part)
    xref = [b"xref\n0 6\n0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    tail = b"".join(xref) + b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
    start = len(body)
    return body + tail + str(start).encode("ascii") + b"\n%%EOF\n"


def test_pdf_bytes_to_text_reads_content():
    """
    Extract printable text from a tiny one-page PDF.

    """
    pdf = _tiny_pdf("IP65 12-36 V")
    assert looks_like_pdf("https://owen.ru/file.bin", "application/pdf", pdf)
    text = pdf_bytes_to_text(pdf)
    assert "IP65" in text
    assert "12-36" in text


def test_normalize_found_drops_placeholders():
    """
    Drop empty-meaning extractor values such as «не указаны».

    """
    found = _normalize_found(
        {"интерфейсы": "не указаны", "питание": "12 В DC"}
    )
    assert found == {"питание": "12 В DC"}


def test_needs_web_research_routing():
    """
    Run research only when search is on, fields are missing, and a name exists.

    """
    device = DeviceStatsModel(name="Манометр X")
    missing = collect_missing(device)
    assert needs_web_research(missing, device.name, llm_search_internet=True) is True
    assert needs_web_research(missing, device.name, llm_search_internet=False) is False
    assert needs_web_research([], device.name, llm_search_internet=True) is False
    assert needs_web_research(missing, "", llm_search_internet=True) is False
