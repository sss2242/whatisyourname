"""Report generation modules (profile builder, Gemini report).

Public API:
    - ``build_company_profile``: T7.1 -- aggregate all pipeline outputs
      into a single JSON profile.
    - ``generate_report``: T7.2 -- produce a Markdown report (via Gemini
      or a local fallback template), optional charts, and optional PDF.
    - ``generate_charts``: standalone chart generator for cache data.
"""

from operator1.report.profile_builder import build_company_profile
from operator1.report.report_generator import generate_charts, generate_report
from operator1.report.enhanced_outputs import generate_enhanced_outputs

__all__ = [
    "build_company_profile",
    "generate_charts",
    "generate_report",
    "generate_enhanced_outputs",
]
