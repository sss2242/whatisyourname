"""Normalize sector labels from SIC codes and raw strings to standard names.

SEC EDGAR returns SIC-based labels like "Electronic Computers" while the
pipeline's sector-aware logic (financial health floors, MC survival
overrides) checks for GICS-like names ("technology", "financial_services").
This module bridges the gap.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Keyword patterns mapped to standardized sector names.
# Order matters -- first match wins.  Patterns are checked against
# the lowercased raw sector string.
_KEYWORD_TO_SECTOR: list[tuple[str, str]] = [
    # Technology
    ("computer", "technology"),
    ("electronic", "technology"),
    ("semiconductor", "technology"),
    ("software", "technology"),
    ("data processing", "technology"),
    ("programming", "technology"),
    ("prepackaged software", "technology"),
    ("printed circuit", "technology"),
    ("integrated circuit", "technology"),
    ("patent owner", "technology"),
    ("information tech", "technology"),
    # Communication Services
    ("telecom", "communication_services"),
    ("communic", "communication_services"),
    ("broadcast", "communication_services"),
    ("cable", "communication_services"),
    ("radio", "communication_services"),
    ("television", "communication_services"),
    # Financial Services
    ("bank", "financial_services"),
    ("savings institution", "financial_services"),
    ("insurance", "financial_services"),
    ("investment", "financial_services"),
    ("security broker", "financial_services"),
    ("commodity", "financial_services"),
    ("finance", "financial_services"),
    ("credit", "financial_services"),
    ("loan", "financial_services"),
    ("real estate", "financial_services"),
    # Healthcare
    ("pharma", "healthcare"),
    ("biological", "healthcare"),
    ("medical", "healthcare"),
    ("surgical", "healthcare"),
    ("health", "healthcare"),
    ("hospital", "healthcare"),
    ("dental", "healthcare"),
    # Energy
    ("oil", "energy"),
    ("gas", "energy"),
    ("petroleum", "energy"),
    ("crude", "energy"),
    ("coal", "energy"),
    ("natural gas", "energy"),
    ("pipeline", "energy"),
    # Consumer Discretionary
    ("retail", "consumer_discretionary"),
    ("restaurant", "consumer_discretionary"),
    ("hotel", "consumer_discretionary"),
    ("motor vehicle", "consumer_discretionary"),
    ("auto", "consumer_discretionary"),
    ("apparel", "consumer_discretionary"),
    ("footwear", "consumer_discretionary"),
    ("toy", "consumer_discretionary"),
    # Consumer Staples
    ("food", "consumer_staples"),
    ("beverage", "consumer_staples"),
    ("tobacco", "consumer_staples"),
    ("grocery", "consumer_staples"),
    ("soap", "consumer_staples"),
    ("household", "consumer_staples"),
    # Industrials
    ("aircraft", "industrials"),
    ("aerospace", "industrials"),
    ("defense", "industrials"),
    ("construction", "industrials"),
    ("industrial", "industrials"),
    ("machinery", "industrials"),
    ("metal", "industrials"),
    ("steel", "industrials"),
    ("railroad", "industrials"),
    ("trucking", "industrials"),
    # Utilities
    ("electric util", "utilities"),
    ("gas util", "utilities"),
    ("water supply", "utilities"),
    ("utility", "utilities"),
    # Materials
    ("chemical", "materials"),
    ("mining", "materials"),
    ("paper", "materials"),
    ("lumber", "materials"),
    ("plastic", "materials"),
]


def normalize_sector(raw_sector: str) -> str:
    """Map a raw sector label to a standardized sector name.

    Handles SIC-based labels ("Electronic Computers"), GICS labels
    ("Information Technology"), and free-form strings.

    Returns one of: technology, communication_services, financial_services,
    healthcare, energy, consumer_discretionary, consumer_staples,
    industrials, utilities, materials, or the original string lowercased
    if no match is found.
    """
    if not raw_sector:
        return ""

    lower = raw_sector.strip().lower()

    # Direct match for already-normalized names
    _STANDARD_NAMES = {
        "technology", "communication_services", "financial_services",
        "healthcare", "energy", "consumer_discretionary", "consumer_staples",
        "industrials", "utilities", "materials",
    }
    if lower in _STANDARD_NAMES:
        return lower
    # Common GICS aliases
    if lower in ("information technology", "tech"):
        return "technology"
    if lower in ("financials", "financial"):
        return "financial_services"
    if lower in ("communication services", "telecom"):
        return "communication_services"
    if lower in ("health care",):
        return "healthcare"
    if lower in ("consumer discretionary",):
        return "consumer_discretionary"
    if lower in ("consumer staples",):
        return "consumer_staples"

    # Keyword matching
    for keyword, sector in _KEYWORD_TO_SECTOR:
        if keyword in lower:
            return sector

    # No match -- return lowercased original
    return lower
