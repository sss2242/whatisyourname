"""Tests for the filing discoverer framework.

Tests the protocol, BSE and ASX discoverers with mocked API responses.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from operator1.clients.filing_discoverer import (
    ASXFilingDiscoverer,
    BSEFilingDiscoverer,
    FilingDiscovery,
    FilingMetadata,
    DISCOVERER_REGISTRY,
    get_discoverer,
    _parse_bse_report_date,
    _classify_bse_filing_type,
)


# ---------------------------------------------------------------------------
# BSE filing date parsing tests
# ---------------------------------------------------------------------------

class TestBSEReportDateParsing:
    """Test parsing of fiscal period end dates from BSE subject lines."""

    def test_quarter_ended_month_day_year(self):
        subject = "Consolidated And Standalone Unaudited Financial Results For The Quarter And Nine Months Ended December 31, 2025"
        assert _parse_bse_report_date(subject) == "2025-12-31"

    def test_year_ended_march(self):
        subject = "Financial Results For The Year Ended March 31, 2025"
        assert _parse_bse_report_date(subject) == "2025-03-31"

    def test_half_year_ended_september(self):
        subject = "Results For The Quarter And Half Year Ended September 30, 2024"
        assert _parse_bse_report_date(subject) == "2024-09-30"

    def test_quarter_ended_june(self):
        subject = "Consolidated And Standalone Unaudited Financial Results For The Quarter Ended June 30, 2025"
        assert _parse_bse_report_date(subject) == "2025-06-30"

    def test_no_date_found(self):
        subject = "Board Meeting Intimation"
        assert _parse_bse_report_date(subject) == ""


class TestBSEFilingTypeClassification:
    """Test classification of BSE filing types."""

    def test_annual(self):
        assert _classify_bse_filing_type("Results For The Year Ended March 2025") == "annual"

    def test_quarterly(self):
        assert _classify_bse_filing_type("Results For The Quarter Ended June 2025") == "quarterly"

    def test_interim(self):
        assert _classify_bse_filing_type("Results For The Half Year Ended September 2024") == "interim"

    def test_default_quarterly(self):
        assert _classify_bse_filing_type("Unaudited Financial Results") == "quarterly"


# ---------------------------------------------------------------------------
# BSE Filing Discoverer tests
# ---------------------------------------------------------------------------

class TestBSEFilingDiscoverer:
    """Test BSE India filing discovery with mocked API."""

    @patch("operator1.clients.filing_discoverer.requests.get")
    def test_discover_filings_success(self, mock_get):
        """Test successful filing discovery from BSE API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Table": [
                {
                    "NEWSSUB": "Consolidated Financial Results For The Quarter Ended December 31, 2025",
                    "ATTACHMENTNAME": "38a2f910-438f-4fc0-8abc-c6cd5933b5ac.pdf",
                    "NEWS_DT": "2026-01-16T19:07:13.5",
                },
                {
                    "NEWSSUB": "Financial Results For The Year Ended March 31, 2025",
                    "ATTACHMENTNAME": "8abf0b4e-efad-4581-9962-01caf39489b9.pdf",
                    "NEWS_DT": "2025-05-15T18:30:00.0",
                },
            ],
            "Table1": [{"RWCNT": "2"}],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        discoverer = BSEFilingDiscoverer()
        result = discoverer.discover_filings("500325", years=2)

        assert isinstance(result, FilingDiscovery)
        assert result.has_filings
        assert len(result.filings) == 2
        assert result.filings[0].filing_date == "2026-01-16"
        assert result.filings[0].report_date == "2025-12-31"
        assert result.filings[0].filing_type == "quarterly"
        assert result.filings[0].document_format == "pdf"
        assert "38a2f910" in result.filings[0].document_url

        assert result.filings[1].filing_type == "annual"
        assert result.filings[1].report_date == "2025-03-31"
        assert len(result.annual_filings()) == 1
        assert len(result.quarterly_filings()) == 1

    @patch("operator1.clients.filing_discoverer.requests.get")
    def test_discover_filings_empty(self, mock_get):
        """Test handling of empty results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"Table": [], "Table1": [{"RWCNT": "0"}]}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        discoverer = BSEFilingDiscoverer()
        result = discoverer.discover_filings("999999")

        assert not result.has_filings
        assert len(result.errors) == 1

    @patch("operator1.clients.filing_discoverer.requests.get")
    def test_discover_filings_api_error(self, mock_get):
        """Test handling of API errors."""
        mock_get.side_effect = Exception("Connection timeout")

        discoverer = BSEFilingDiscoverer()
        result = discoverer.discover_filings("500325")

        assert not result.has_filings
        assert len(result.errors) == 1
        assert "Connection timeout" in result.errors[0]

    @patch("operator1.clients.filing_discoverer.requests.get")
    def test_download_filing_success(self, mock_get):
        """Test PDF download."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"%PDF-1.4 fake pdf content"
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        discoverer = BSEFilingDiscoverer()
        filing = FilingMetadata(
            title="Test Filing",
            document_url="https://www.bseindia.com/xml-data/corpfiling/AttachLive/test.pdf",
            document_format="pdf",
        )
        content = discoverer.download_filing(filing)

        assert content.startswith(b"%PDF")

    @patch("operator1.clients.filing_discoverer.requests.get")
    def test_download_filing_not_pdf(self, mock_get):
        """Test rejection of non-PDF content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html>Not a PDF</html>"
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        discoverer = BSEFilingDiscoverer()
        filing = FilingMetadata(
            title="Test", document_url="https://example.com/test.pdf",
        )

        with pytest.raises(ValueError, match="Expected PDF"):
            discoverer.download_filing(filing)


# ---------------------------------------------------------------------------
# ASX Filing Discoverer tests
# ---------------------------------------------------------------------------

class TestASXFilingDiscoverer:
    """Test ASX filing discovery with mocked MarkitDigital API."""

    @patch("operator1.clients.filing_discoverer.requests.get")
    def test_discover_filings_success(self, mock_get):
        """Test successful announcement discovery from ASX."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "displayName": "BHP GROUP LIMITED",
                "items": [
                    {
                        "announcementType": "PERIODIC REPORTS",
                        "date": "2026-02-16T21:39:18.000Z",
                        "documentKey": "2924-03057188-3A687224",
                        "fileSize": "500KB",
                        "headline": "HY2026 Results Presentation",
                        "isPriceSensitive": False,
                        "url": "",
                    },
                    {
                        "announcementType": "DISTRIBUTION ANNOUNCEMENT",
                        "date": "2026-02-16T21:38:23.000Z",
                        "documentKey": "2924-03057186-3A687223",
                        "fileSize": "200KB",
                        "headline": "Half Yearly Report and Accounts",
                        "isPriceSensitive": False,
                        "url": "",
                    },
                    {
                        "announcementType": "OTHER",
                        "date": "2026-01-15T10:00:00.000Z",
                        "documentKey": "2924-03050000-3A680000",
                        "fileSize": "100KB",
                        "headline": "Change of Director Interest",
                        "isPriceSensitive": False,
                        "url": "",
                    },
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        discoverer = ASXFilingDiscoverer()
        result = discoverer.discover_filings("BHP", years=2)

        assert isinstance(result, FilingDiscovery)
        assert result.has_filings
        # Should have 2 filings (the "Change of Director" is filtered out)
        assert len(result.filings) == 2
        assert result.filings[0].filing_date == "2026-02-16"
        assert result.filings[0].market_id == "au_asx"

    @patch("operator1.clients.filing_discoverer.requests.get")
    def test_discover_filings_api_error(self, mock_get):
        """Test handling of API errors."""
        mock_get.side_effect = Exception("Service unavailable")

        discoverer = ASXFilingDiscoverer()
        result = discoverer.discover_filings("BHP")

        assert not result.has_filings
        assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Test the discoverer registry."""

    def test_bse_registered(self):
        assert "in_bse" in DISCOVERER_REGISTRY

    def test_asx_registered(self):
        assert "au_asx" in DISCOVERER_REGISTRY

    def test_get_discoverer_bse(self):
        d = get_discoverer("in_bse")
        assert d is not None
        assert isinstance(d, BSEFilingDiscoverer)

    def test_get_discoverer_unknown(self):
        d = get_discoverer("xx_unknown")
        assert d is None
