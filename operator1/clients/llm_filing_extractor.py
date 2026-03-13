"""Universal LLM-aided financial filing extractor.

Extracts structured financial data from PDF, HTML, and iXBRL filings
using the existing Gemini/Claude LLM clients. Works across all markets.

The extractor handles three input formats:
1. **iXBRL**: Machine-parseable tagged data (best quality, no LLM needed)
2. **HTML**: Semi-structured tables (light LLM validation)
3. **PDF**: Unstructured text (full LLM extraction)

The extracted data is validated using accounting identities before
being returned in canonical long format.

Usage:
    extractor = LLMFilingExtractor(llm_client)
    result = extractor.extract_from_pdf(pdf_bytes, market_id="au_asx")
    result = extractor.extract_from_html(html_text, market_id="hk_hkex")
    result = extractor.extract_from_ixbrl(ixbrl_text, market_id="ca_sedar")
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-market taxonomy hints injected into the LLM extraction prompt.
# Each entry gives the LLM regional context: accounting standard, language,
# currency, and the local-language keywords it should look for in filings.
# This dramatically improves extraction accuracy for non-English filings.
# ---------------------------------------------------------------------------

_MARKET_TAXONOMY_HINTS: dict[str, str] = {
    "us_sec_edgar": (
        "MARKET: United States (SEC EDGAR). Standard: US-GAAP. Currency: USD.\n"
        "Language: English.\n"
        "Key taxonomy terms to search for:\n"
        "- Revenue / Net Sales / Revenue from Contract with Customer\n"
        "- Cost of Revenue / Cost of Goods Sold\n"
        "- Stockholders' Equity / Shareholders' Equity\n"
        "- Earnings Per Share Basic / Diluted\n"
        "- Net Cash Provided by Operating Activities\n"
        "- Payments to Acquire Property, Plant and Equipment (capex)\n"
    ),
    "uk_companies_house": (
        "MARKET: United Kingdom (Companies House / LSE). Standard: UK-GAAP / FRS 102 / IFRS. Currency: GBP.\n"
        "Language: English.\n"
        "Key taxonomy terms to search for:\n"
        "- Turnover (= revenue), Cost of Sales\n"
        "- Gross Profit, Operating Profit\n"
        "- Profit Before Tax, Profit for the Period\n"
        "- Shareholder Funds (= total equity)\n"
        "- Creditors Due Within One Year (= current liabilities)\n"
        "- Creditors Due After One Year (= long-term debt)\n"
        "- Cash at Bank and in Hand\n"
    ),
    "eu_esef": (
        "MARKET: European Union (ESEF / Euronext). Standard: IFRS. Currency: EUR.\n"
        "Language: English or local EU language.\n"
        "Key IFRS taxonomy terms:\n"
        "- Revenue, Cost of Sales, Gross Profit\n"
        "- Profit (Loss) from Operating Activities\n"
        "- Finance Costs (= interest expense)\n"
        "- Assets, Liabilities, Equity\n"
        "- Cash Flows from Operating/Investing/Financing Activities\n"
        "- Purchase of Property, Plant and Equipment (capex)\n"
    ),
    "fr_esef": (
        "MARKET: France (Euronext Paris / ESEF). Standard: IFRS. Currency: EUR.\n"
        "Language: French (sometimes English).\n"
        "Key French taxonomy terms:\n"
        "- Chiffre d'affaires (revenue), Cout des ventes (cost of sales)\n"
        "- Resultat operationnel (operating income)\n"
        "- Resultat net (net income), Resultat avant impots (EBT/EBIT)\n"
        "- Total actif (total assets), Total passif (total liabilities)\n"
        "- Capitaux propres (total equity)\n"
        "- Tresorerie et equivalents (cash and equivalents)\n"
        "- Flux de tresorerie operationnels (operating cash flow)\n"
    ),
    "de_esef": (
        "MARKET: Germany (Frankfurt / XETRA / ESEF). Standard: IFRS. Currency: EUR.\n"
        "Language: German (sometimes English).\n"
        "Key German taxonomy terms:\n"
        "- Umsatzerlose (revenue), Umsatzkosten (cost of revenue)\n"
        "- Bruttoergebnis (gross profit), Betriebsergebnis (operating income)\n"
        "- Jahresuberschuss / Konzernergebnis (net income)\n"
        "- Bilanzsumme / Aktiva gesamt (total assets)\n"
        "- Verbindlichkeiten (liabilities), Eigenkapital (equity)\n"
        "- Zahlungsmittel und Zahlungsmittelaquivalente (cash)\n"
        "- Cashflow aus betrieblicher Tatigkeit (operating cash flow)\n"
    ),
    "jp_jquants": (
        "MARKET: Japan (TSE / J-Quants / EDINET). Standard: JPPFS (Japan GAAP) or IFRS. Currency: JPY.\n"
        "Language: Japanese (sometimes English).\n"
        "Key Japanese taxonomy terms:\n"
        "- \u58f2\u4e0a\u9ad8 / \u55b6\u696d\u53ce\u76ca (revenue / net sales)\n"
        "- \u58f2\u4e0a\u539f\u4fa1 (cost of sales), \u58f2\u4e0a\u7dcf\u5229\u76ca (gross profit)\n"
        "- \u55b6\u696d\u5229\u76ca (operating income), \u7d4c\u5e38\u5229\u76ca (ordinary income / EBIT)\n"
        "- \u5f53\u671f\u7d14\u5229\u76ca (net income), \u6cd5\u4eba\u7a0e\u7b49 (taxes)\n"
        "- \u8cc7\u7523\u5408\u8a08 (total assets), \u8ca0\u50b5\u5408\u8a08 (total liabilities)\n"
        "- \u7d14\u8cc7\u7523 (net assets / equity)\n"
        "- \u73fe\u91d1\u53ca\u3073\u9810\u91d1 (cash and deposits)\n"
        "- \u77ed\u671f\u501f\u5165\u91d1 (short-term loans), \u9577\u671f\u501f\u5165\u91d1 (long-term loans)\n"
        "- \u55b6\u696d\u6d3b\u52d5\u306b\u3088\u308b\u30ad\u30e3\u30c3\u30b7\u30e5\u30fb\u30d5\u30ed\u30fc (operating CF)\n"
        "- \u6709\u5f62\u56fa\u5b9a\u8cc7\u7523\u306e\u53d6\u5f97 (capex)\n"
        "Amounts are often in millions of yen (\u767e\u4e07\u5186). Multiply by 1,000,000.\n"
    ),
    "kr_dart": (
        "MARKET: South Korea (KRX / DART). Standard: K-IFRS. Currency: KRW.\n"
        "Language: Korean.\n"
        "Key Korean K-IFRS taxonomy terms:\n"
        "- \ub9e4\ucd9c\uc561 / \uc218\uc775(\ub9e4\ucd9c\uc561) (revenue)\n"
        "- \ub9e4\ucd9c\uc6d0\uac00 (cost of revenue), \ub9e4\ucd9c\ucd1d\uc774\uc775 (gross profit)\n"
        "- \uc601\uc5c5\uc774\uc775 (operating income)\n"
        "- \ub2f9\uae30\uc21c\uc774\uc775 (net income), \ubc95\uc778\uc138\ube44\uc6a9 (taxes)\n"
        "- \uc774\uc790\ube44\uc6a9 / \uae08\uc735\ube44\uc6a9 (interest expense / finance costs)\n"
        "- \uc790\uc0b0\ucd1d\uacc4 (total assets), \ubd80\ucc44\ucd1d\uacc4 (total liabilities)\n"
        "- \uc790\ubcf8\ucd1d\uacc4 (total equity)\n"
        "- \uc720\ub3d9\uc790\uc0b0 (current assets), \uc720\ub3d9\ubd80\ucc44 (current liabilities)\n"
        "- \ud604\uae08\ubc0f\ud604\uae08\uc131\uc790\uc0b0 (cash and equivalents)\n"
        "- \ub2e8\uae30\ucc28\uc785\uae08 (short-term debt), \uc7a5\uae30\ucc28\uc785\uae08 (long-term debt)\n"
        "- \ub9e4\ucd9c\ucc44\uad8c (receivables), \uc7ac\uace0\uc790\uc0b0 (inventory)\n"
        "- \ub9e4\uc785\ucc44\ubb34 (payables), \uc774\uc775\uc789\uc5ec\uae08 (retained earnings)\n"
        "- \uc601\uc5c5\ud65c\ub3d9\ud604\uae08\ud750\ub984 (operating CF)\n"
        "- \ud310\ub9e4\ube44\uc640\uad00\ub9ac\ube44 (SGA expenses)\n"
        "Amounts are often in millions of KRW (\ubc31\ub9cc\uc6d0). Multiply by 1,000,000.\n"
    ),
    "tw_mops": (
        "MARKET: Taiwan (TWSE / MOPS). Standard: TIFRS (Taiwan IFRS). Currency: TWD.\n"
        "Language: Traditional Chinese.\n"
        "Key Traditional Chinese taxonomy terms:\n"
        "- \u71df\u696d\u6536\u5165\u5408\u8a08 (revenue), \u71df\u696d\u6210\u672c\u5408\u8a08 (cost of revenue)\n"
        "- \u71df\u696d\u6bdb\u5229 (gross profit), \u71df\u696d\u5229\u76ca (operating income)\n"
        "- \u672c\u671f\u6de8\u5229 (net income), \u7a05\u524d\u6de8\u5229 (EBT/EBIT)\n"
        "- \u6240\u5f97\u7a05\u8cbb\u7528 (taxes), \u5229\u606f\u8cbb\u7528 (interest expense)\n"
        "- \u8cc7\u7522\u7e3d\u8a08 (total assets), \u8ca0\u50b5\u7e3d\u8a08 (total liabilities)\n"
        "- \u6b0a\u76ca\u7e3d\u8a08 (total equity)\n"
        "- \u6d41\u52d5\u8cc7\u7522\u5408\u8a08 (current assets), \u6d41\u52d5\u8ca0\u50b5\u5408\u8a08 (current liabilities)\n"
        "- \u73fe\u91d1\u53ca\u7d04\u7576\u73fe\u91d1 (cash and equivalents)\n"
        "- \u77ed\u671f\u501f\u6b3e (short-term debt), \u9577\u671f\u501f\u6b3e (long-term debt)\n"
        "- \u4fdd\u7559\u76c8\u9918 (retained earnings)\n"
        "- \u71df\u696d\u6d3b\u52d5\u4e4b\u6de8\u73fe\u91d1\u6d41\u5165 (operating CF)\n"
        "- \u53d6\u5f97\u4e0d\u52d5\u7522\u3001\u5ee0\u623f\u53ca\u8a2d\u5099 (capex)\n"
        "Amounts are often in thousands of TWD (\u4edf\u5143). Multiply by 1,000.\n"
    ),
    "cn_sse": (
        "MARKET: China (SSE / SZSE). Standard: CAS (Chinese Accounting Standards). Currency: CNY/RMB.\n"
        "Language: Simplified Chinese.\n"
        "Key Simplified Chinese taxonomy terms:\n"
        "- \u8425\u4e1a\u6536\u5165 / \u8425\u4e1a\u603b\u6536\u5165 (revenue)\n"
        "- \u8425\u4e1a\u6210\u672c / \u8425\u4e1a\u603b\u6210\u672c (cost of revenue)\n"
        "- \u8425\u4e1a\u5229\u6da6 (operating income), \u5229\u6da6\u603b\u989d (EBIT)\n"
        "- \u51c0\u5229\u6da6 (net income), \u6240\u5f97\u7a0e\u8d39\u7528 (taxes)\n"
        "- \u5229\u606f\u652f\u51fa / \u8d22\u52a1\u8d39\u7528 (interest / finance costs)\n"
        "- \u8d44\u4ea7\u603b\u8ba1 (total assets), \u8d1f\u503a\u5408\u8ba1 (total liabilities)\n"
        "- \u6240\u6709\u8005\u6743\u76ca\u5408\u8ba1 (total equity)\n"
        "- \u6d41\u52a8\u8d44\u4ea7\u5408\u8ba1 (current assets), \u6d41\u52a8\u8d1f\u503a\u5408\u8ba1 (current liabilities)\n"
        "- \u8d27\u5e01\u8d44\u91d1 (cash and equivalents)\n"
        "- \u77ed\u671f\u501f\u6b3e (short-term debt), \u957f\u671f\u501f\u6b3e (long-term debt)\n"
        "- \u672a\u5206\u914d\u5229\u6da6 (retained earnings)\n"
        "- \u5e94\u6536\u8d26\u6b3e (receivables), \u5b58\u8d27 (inventory), \u5e94\u4ed8\u8d26\u6b3e (payables)\n"
        "- \u7ecf\u8425\u6d3b\u52a8\u4ea7\u751f\u7684\u73b0\u91d1\u6d41\u91cf\u51c0\u989d (operating CF)\n"
        "- \u9500\u552e\u8d39\u7528 (selling expenses), \u7ba1\u7406\u8d39\u7528 (admin expenses)\n"
        "- \u7814\u53d1\u8d39\u7528 (R&D expenses)\n"
        "Amounts are often in yuan or wan yuan (\u4e07\u5143, 10,000). Check the unit header.\n"
    ),
    "br_cvm": (
        "MARKET: Brazil (B3 / CVM). Standard: BR-GAAP / IFRS. Currency: BRL.\n"
        "Language: Portuguese.\n"
        "Key Portuguese taxonomy terms:\n"
        "- Receita de Venda (revenue), Custo dos Bens Vendidos (cost of revenue)\n"
        "- Resultado Bruto (gross profit)\n"
        "- Resultado Operacional (operating income)\n"
        "- Lucro/Prejuizo do Periodo (net income)\n"
        "- Resultado Antes dos Tributos (EBT/EBIT)\n"
        "- Imposto de Renda (taxes)\n"
        "- Ativo Total (total assets), Passivo Total (total liabilities)\n"
        "- Patrimonio Liquido (total equity)\n"
        "- Ativo Circulante (current assets), Passivo Circulante (current liabilities)\n"
        "- Caixa e Equivalentes (cash and equivalents)\n"
        "- Emprestimos de Curto Prazo (short-term debt)\n"
        "- Emprestimos de Longo Prazo (long-term debt)\n"
        "- Caixa Liquido das Atividades Operacionais (operating CF)\n"
        "CVM uses hierarchical account codes: 3.01=revenue, 1=total assets, 2=liabilities.\n"
    ),
    "cl_cmf": (
        "MARKET: Chile (Santiago / CMF). Standard: IFRS (Chile). Currency: CLP.\n"
        "Language: Spanish.\n"
        "Key Spanish taxonomy terms (FECU format):\n"
        "- Ingresos de actividades ordinarias (revenue)\n"
        "- Costo de ventas (cost of sales)\n"
        "- Ganancia bruta (gross profit)\n"
        "- Ganancia por actividades de operacion (operating income)\n"
        "- Ganancia / Perdida (net income)\n"
        "- Total de activos (total assets), Total de pasivos (total liabilities)\n"
        "- Total de patrimonio (total equity)\n"
        "- Activos corrientes (current assets), Pasivos corrientes (current liabilities)\n"
        "- Efectivo y equivalentes (cash and equivalents)\n"
        "- Flujos de efectivo de actividades de operacion (operating CF)\n"
    ),
    "in_bse": (
        "MARKET: India (BSE / NSE). Standard: Ind AS (IFRS-converged). Currency: INR.\n"
        "Language: English.\n"
        "Key taxonomy terms:\n"
        "- Revenue from Operations / Total Income\n"
        "- Cost of Materials Consumed / Cost of Revenue\n"
        "- Profit Before Tax, Profit After Tax / Net Profit\n"
        "- Tax Expense / Current Tax / Deferred Tax\n"
        "- Total Assets, Total Liabilities, Total Equity / Net Worth\n"
        "- Trade Receivables, Inventories, Trade Payables\n"
        "- Cash and Cash Equivalents\n"
        "- Borrowings (current / non-current)\n"
        "Amounts are often in lakhs (1 lakh = 100,000) or crores (1 crore = 10,000,000).\n"
    ),
    "ca_sedar": (
        "MARKET: Canada (TSX / SEDAR+). Standard: IFRS or US-GAAP. Currency: CAD.\n"
        "Language: English or French.\n"
        "Key taxonomy terms:\n"
        "- Revenue / Net Sales, Cost of Sales\n"
        "- Operating Income / Loss, Net Income / Loss\n"
        "- Total Assets, Total Liabilities, Shareholders' Equity\n"
        "- Cash and Cash Equivalents\n"
        "- Cash Flows from Operating / Investing / Financing Activities\n"
        "Canadian IFRS filings use the same IFRS taxonomy tags as EU ESEF.\n"
    ),
    "au_asx": (
        "MARKET: Australia (ASX). Standard: AASB (IFRS-based). Currency: AUD.\n"
        "Language: English.\n"
        "Key taxonomy terms:\n"
        "- Revenue / Sales Revenue, Cost of Sales\n"
        "- Profit Before Income Tax, Income Tax Expense\n"
        "- Net Profit After Tax (NPAT)\n"
        "- Total Assets, Total Liabilities, Total Equity / Net Assets\n"
        "- Current Assets, Current Liabilities, Non-current Liabilities\n"
        "- Cash and Cash Equivalents, Trade Receivables, Inventories\n"
        "- Net Cash from Operating / Investing / Financing Activities\n"
        "- Payments for Property, Plant and Equipment (capex)\n"
        "Appendix 4D (half-year) and 4E (annual) results use standardized headers.\n"
    ),
    "hk_hkex": (
        "MARKET: Hong Kong (HKEX). Standard: HKFRS (IFRS-identical). Currency: HKD.\n"
        "Language: English and/or Traditional Chinese.\n"
        "Key taxonomy terms (English / Chinese):\n"
        "- Revenue / \u6536\u5165 (revenue)\n"
        "- Cost of Sales / \u92b7\u552e\u6210\u672c (cost of revenue)\n"
        "- Profit from Operations / \u7d93\u71df\u6ea2\u5229 (operating income)\n"
        "- Profit for the Year / \u5e74\u5ea6\u6ea2\u5229 (net income)\n"
        "- Total Assets / \u7e3d\u8cc7\u7522, Total Liabilities / \u7e3d\u8ca0\u50b5\n"
        "- Total Equity / \u7e3d\u6b0a\u76ca\n"
        "- Cash and Cash Equivalents / \u73fe\u91d1\u53ca\u73fe\u91d1\u7b49\u50f9\u7269\n"
        "- Trade Receivables / \u61c9\u6536\u8cec\u6b3e, Inventories / \u5b58\u8ca8\n"
        "- Net Cash from Operating Activities / \u7d93\u71df\u6d3b\u52d5\u6240\u5f97\u73fe\u91d1\u6de8\u984d\n"
    ),
    "sg_sgx": (
        "MARKET: Singapore (SGX). Standard: SFRS(I) (IFRS-identical). Currency: SGD.\n"
        "Language: English.\n"
        "Key taxonomy terms:\n"
        "- Revenue, Cost of Sales, Gross Profit\n"
        "- Profit Before Tax, Income Tax Expense, Net Profit\n"
        "- Total Assets, Total Liabilities, Total Equity\n"
        "- Cash and Cash Equivalents\n"
        "- Net Cash from Operating / Investing / Financing Activities\n"
        "Singapore uses IFRS-identical standards (SFRS(I)). Same tags as IFRS.\n"
    ),
    "mx_bmv": (
        "MARKET: Mexico (BMV / CNBV). Standard: IFRS (mandatory for listed). Currency: MXN.\n"
        "Language: Spanish.\n"
        "Key Spanish taxonomy terms:\n"
        "- Ingresos (revenue), Costo de Ventas (cost of sales)\n"
        "- Utilidad Bruta (gross profit), Utilidad de Operacion (operating income)\n"
        "- Utilidad Neta (net income)\n"
        "- Activos Totales (total assets), Pasivos Totales (total liabilities)\n"
        "- Capital Contable (total equity)\n"
        "- Efectivo y Equivalentes (cash and equivalents)\n"
        "- Flujos de Efectivo de Actividades de Operacion (operating CF)\n"
    ),
    "za_jse": (
        "MARKET: South Africa (JSE). Standard: IFRS (mandatory). Currency: ZAR.\n"
        "Language: English.\n"
        "Key taxonomy terms:\n"
        "- Revenue, Cost of Sales, Gross Profit\n"
        "- Operating Profit, Profit Before Tax, Net Profit\n"
        "- Total Assets, Total Liabilities, Total Equity\n"
        "- Cash and Cash Equivalents\n"
        "- Headline Earnings Per Share (HEPS) -- JSE-specific metric\n"
        "South African filings follow standard IFRS terminology.\n"
    ),
    "ch_six": (
        "MARKET: Switzerland (SIX). Standard: IFRS or Swiss GAAP FER. Currency: CHF.\n"
        "Language: German, French, Italian, or English.\n"
        "Key taxonomy terms (German / English):\n"
        "- Umsatz / Nettoumsatz (revenue), Bruttogewinn (gross profit)\n"
        "- Betriebsergebnis / EBIT (operating income)\n"
        "- Reingewinn (net income), Ertragssteuern (taxes)\n"
        "- Bilanzsumme (total assets), Eigenkapital (equity)\n"
        "- Flussige Mittel (cash and equivalents)\n"
        "- Geldfluss aus Betriebstatigkeit (operating CF)\n"
        "Swiss GAAP FER is simpler than IFRS; some fields may not be present.\n"
    ),
    "sa_tadawul": (
        "MARKET: Saudi Arabia (Tadawul). Standard: IFRS (mandatory since 2017). Currency: SAR.\n"
        "Language: Arabic (with English translation often available).\n"
        "Key Arabic taxonomy terms:\n"
        "- \u0627\u0644\u0625\u064a\u0631\u0627\u062f\u0627\u062a (revenue), \u062a\u0643\u0644\u0641\u0629 \u0627\u0644\u0625\u064a\u0631\u0627\u062f\u0627\u062a (cost of revenue)\n"
        "- \u0625\u062c\u0645\u0627\u0644\u064a \u0627\u0644\u0631\u0628\u062d (gross profit), \u0627\u0644\u0631\u0628\u062d \u0627\u0644\u062a\u0634\u063a\u064a\u0644\u064a (operating income)\n"
        "- \u0635\u0627\u0641\u064a \u0627\u0644\u0631\u0628\u062d (net income), \u0645\u0635\u0631\u0648\u0641 \u0627\u0644\u0632\u0643\u0627\u0629 \u0648\u0627\u0644\u0636\u0631\u064a\u0628\u0629 (zakat & tax)\n"
        "- \u0625\u062c\u0645\u0627\u0644\u064a \u0627\u0644\u0645\u0648\u062c\u0648\u062f\u0627\u062a (total assets), \u0625\u062c\u0645\u0627\u0644\u064a \u0627\u0644\u0645\u0637\u0644\u0648\u0628\u0627\u062a (total liabilities)\n"
        "- \u0625\u062c\u0645\u0627\u0644\u064a \u062d\u0642\u0648\u0642 \u0627\u0644\u0645\u0644\u0643\u064a\u0629 (total equity)\n"
        "- \u0627\u0644\u0646\u0642\u062f \u0648\u0645\u0627 \u064a\u0639\u0627\u062f\u0644\u0647 (cash and equivalents)\n"
        "- \u0627\u0644\u062a\u062f\u0641\u0642\u0627\u062a \u0627\u0644\u0646\u0642\u062f\u064a\u0629 \u0645\u0646 \u0627\u0644\u0623\u0646\u0634\u0637\u0629 \u0627\u0644\u062a\u0634\u063a\u064a\u0644\u064a\u0629 (operating CF)\n"
        "Note: Saudi filings include Zakat (Islamic tax) in addition to income tax.\n"
    ),
    "ae_dfm": (
        "MARKET: UAE (DFM / ADX). Standard: IFRS (mandatory). Currency: AED.\n"
        "Language: Arabic and English (bilingual filings).\n"
        "Key Arabic taxonomy terms:\n"
        "- \u0627\u0644\u0625\u064a\u0631\u0627\u062f\u0627\u062a (revenue), \u062a\u0643\u0644\u0641\u0629 \u0627\u0644\u0625\u064a\u0631\u0627\u062f\u0627\u062a (cost of revenue)\n"
        "- \u0635\u0627\u0641\u064a \u0627\u0644\u0631\u0628\u062d (net income)\n"
        "- \u0625\u062c\u0645\u0627\u0644\u064a \u0627\u0644\u0645\u0648\u062c\u0648\u062f\u0627\u062a (total assets), \u0625\u062c\u0645\u0627\u0644\u064a \u0627\u0644\u0645\u0637\u0644\u0648\u0628\u0627\u062a (total liabilities)\n"
        "- \u062d\u0642\u0648\u0642 \u0627\u0644\u0645\u0644\u0643\u064a\u0629 (equity), \u0627\u0644\u0646\u0642\u062f \u0648\u0645\u0627 \u064a\u0639\u0627\u062f\u0644\u0647 (cash)\n"
        "UAE uses standard IFRS terminology. English sections follow IFRS tags.\n"
    ),
    # EU sub-markets with local language hints
    "nl_esef": (
        "MARKET: Netherlands (Euronext Amsterdam / ESEF). Standard: IFRS. Currency: EUR.\n"
        "Language: Dutch or English.\n"
        "Key Dutch taxonomy terms:\n"
        "- Omzet / Netto-omzet (revenue), Kostprijs (cost of sales)\n"
        "- Bedrijfsresultaat (operating income), Nettoresultaat (net income)\n"
        "- Totale activa (total assets), Totale passiva (total liabilities)\n"
        "- Eigen vermogen (equity), Geldmiddelen (cash)\n"
        "- Kasstroom uit bedrijfsactiviteiten (operating CF)\n"
    ),
    "es_esef": (
        "MARKET: Spain (BME / ESEF). Standard: IFRS. Currency: EUR.\n"
        "Language: Spanish.\n"
        "Key Spanish taxonomy terms:\n"
        "- Ingresos / Cifra de negocios (revenue), Coste de ventas (cost of sales)\n"
        "- Resultado de explotacion (operating income)\n"
        "- Resultado del ejercicio (net income)\n"
        "- Total activo (total assets), Total pasivo (total liabilities)\n"
        "- Patrimonio neto (equity)\n"
        "- Efectivo y equivalentes (cash)\n"
        "- Flujos de efectivo de actividades de explotacion (operating CF)\n"
    ),
    "it_esef": (
        "MARKET: Italy (Borsa Italiana / ESEF). Standard: IFRS. Currency: EUR.\n"
        "Language: Italian or English.\n"
        "Key Italian taxonomy terms:\n"
        "- Ricavi (revenue), Costo del venduto (cost of sales)\n"
        "- Risultato operativo (operating income)\n"
        "- Utile / Perdita netta (net income)\n"
        "- Totale attivo (total assets), Totale passivo (total liabilities)\n"
        "- Patrimonio netto (equity)\n"
        "- Disponibilita liquide (cash and equivalents)\n"
        "- Flusso di cassa operativo (operating CF)\n"
    ),
    "se_esef": (
        "MARKET: Sweden (Nasdaq Stockholm / ESEF). Standard: IFRS. Currency: SEK.\n"
        "Language: Swedish or English.\n"
        "Key Swedish taxonomy terms:\n"
        "- Nettoomsattning (revenue), Kostnad for salda varor (cost of sales)\n"
        "- Rorelseresultat (operating income), Arets resultat (net income)\n"
        "- Summa tillgangar (total assets), Summa skulder (total liabilities)\n"
        "- Eget kapital (equity)\n"
        "- Likvida medel (cash and equivalents)\n"
        "- Kassaflode fran den lopande verksamheten (operating CF)\n"
    ),
}


def get_taxonomy_hint(market_id: str) -> str:
    """Return the taxonomy hint string for a given market, or empty string."""
    return _MARKET_TAXONOMY_HINTS.get(market_id, "")


# Canonical financial fields the extractor should look for.
# These match STATEMENT_FIELDS in cache_builder.py.
EXTRACTION_FIELDS = [
    "revenue", "cost_of_revenue", "gross_profit",
    "operating_income", "ebit", "ebitda", "net_income",
    "interest_expense", "taxes",
    "total_assets", "total_liabilities", "total_equity",
    "current_assets", "current_liabilities",
    "cash_and_equivalents", "short_term_debt", "long_term_debt",
    "total_debt", "retained_earnings",
    "operating_cash_flow", "capex", "free_cash_flow",
    "investing_cf", "financing_cf", "dividends_paid",
    "stock_buybacks",
    "sga_expenses", "rd_expenses",
    "eps", "eps_diluted",
]

# The LLM prompt -- designed to extract ALL fields in a single call.
_EXTRACTION_PROMPT = """You are a financial data extraction expert. Extract the following financial data from this filing document.
{taxonomy_hint}
Return ONLY a valid JSON object with these exact keys (use null if a value is not found):

INCOME STATEMENT:
- revenue: Total revenue/sales (number)
- cost_of_revenue: Cost of goods sold (number)
- gross_profit: Gross profit (number)
- operating_income: Operating income/profit (number)
- ebit: Earnings before interest and taxes (number)
- ebitda: EBITDA (number)
- net_income: Net income/profit (number)
- interest_expense: Interest expense (number)
- taxes: Income tax expense (number)
- sga_expenses: Selling, general and administrative expenses (number)
- rd_expenses: Research and development expenses (number)
- eps: Earnings per share basic (number)
- eps_diluted: Earnings per share diluted (number)

BALANCE SHEET:
- total_assets: Total assets (number)
- total_liabilities: Total liabilities (number)
- total_equity: Total shareholders equity (number)
- current_assets: Total current assets (number)
- current_liabilities: Total current liabilities (number)
- cash_and_equivalents: Cash and cash equivalents (number)
- short_term_debt: Short-term borrowings/debt (number)
- long_term_debt: Long-term debt (number)
- total_debt: Total debt (number)
- retained_earnings: Retained earnings (number)

CASH FLOW STATEMENT:
- operating_cash_flow: Net cash from operating activities (number)
- capex: Capital expenditures (number, usually negative)
- free_cash_flow: Free cash flow (number)
- investing_cf: Net cash from investing activities (number)
- financing_cf: Net cash from financing activities (number)
- dividends_paid: Dividends paid (number, usually negative)
- stock_buybacks: Share repurchases (number, usually negative)

METADATA:
- report_date: The period end date in YYYY-MM-DD format
- filing_date: The date the filing was published in YYYY-MM-DD format
- currency: 3-letter currency code (e.g. CAD, AUD, HKD)
- period_type: One of "annual", "quarterly", "semiannual"

All monetary values should be in the filing's native currency and unit.
If amounts are in thousands, multiply by 1000. If in millions, multiply by 1,000,000.
Return raw numbers, not formatted strings.

FILING TEXT:
{text}
"""

# Shorter prompt for validation/cross-check of existing data
_VALIDATION_PROMPT = """Verify this financial data extracted from a {market} filing.
Check for obvious errors (e.g., total_assets != total_liabilities + total_equity).
Return the corrected JSON if errors found, or the same JSON if correct.
Add a "validation_notes" field with any issues found.

Data to validate:
{data}

Filing text excerpt:
{text}
"""


@dataclass
class ExtractionResult:
    """Result of financial data extraction."""

    success: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    source_format: str = ""  # "ixbrl", "html", "pdf"
    validation_passed: bool = False
    validation_notes: list[str] = field(default_factory=list)
    error: str = ""
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# PDF page scoring helpers (used by extract_from_pdf)
# ---------------------------------------------------------------------------

# Keywords that indicate financial statement content
_FINANCIAL_KEYWORDS = [
    "income statement", "profit and loss", "profit or loss",
    "statement of comprehensive income", "statement of profit",
    "consolidated income", "results of operations",
    "balance sheet", "statement of financial position",
    "consolidated balance", "assets and liabilities",
    "cash flow", "statement of cash flows", "consolidated cash flow",
    "revenue", "total revenue", "net sales", "cost of sales",
    "gross profit", "operating income", "operating profit",
    "profit before tax", "net income", "net profit",
    "total assets", "total liabilities", "total equity",
    "shareholders equity", "current assets", "current liabilities",
    "cash and cash equivalents", "operating activities",
    "investing activities", "financing activities",
    "earnings per share", "dividends per share",
    "us$m", "us$ m", "a$m", "hk$m", "rm m", "r m",
]

# Keywords that indicate non-financial content (noise)
_NOISE_KEYWORDS = [
    "table of contents", "risk factors", "forward-looking",
    "management discussion", "corporate governance",
    "board of directors", "remuneration", "sustainability",
    "environmental", "safety", "community",
]


def _score_page_for_financials(text: str) -> float:
    """Score a single page for financial statement content."""
    lower = text.lower()
    score = 0.0

    for kw in _FINANCIAL_KEYWORDS:
        if kw in lower:
            score += 2.0

    # Bonus for pages with many numbers (financial tables)
    number_count = len(re.findall(r'\d[\d,]+\.?\d*', text))
    if number_count > 10:
        score += number_count * 0.1

    # Bonus for pipe-separated content (table extraction format)
    pipe_count = text.count("|")
    if pipe_count > 5:
        score += pipe_count * 0.05

    for kw in _NOISE_KEYWORDS:
        if kw in lower:
            score -= 1.0

    return score


def _select_top_pages(
    page_texts: list[tuple[int, str]],
    max_chars: int = 12000,
) -> str:
    """Score pages and select the highest-scoring ones up to max_chars.

    Always includes page 0 (cover/summary) for metadata context,
    then fills with the highest-scoring financial pages.
    """
    if not page_texts:
        return ""

    # Score each page
    scored = [(pn, text, _score_page_for_financials(text)) for pn, text in page_texts]
    scored.sort(key=lambda x: x[2], reverse=True)

    selected: list[tuple[int, str]] = []
    total_chars = 0

    # Always include page 0 for metadata (report date, currency, period)
    page_zero = next((s for s in scored if s[0] == 0), None)
    if page_zero:
        selected.append((page_zero[0], page_zero[1]))
        total_chars += len(page_zero[1])

    # Add highest-scoring pages
    for page_num, text, score in scored:
        if page_num == 0:
            continue
        if score <= 0:
            continue
        if total_chars + len(text) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 200:
                selected.append((page_num, text[:remaining]))
            break
        selected.append((page_num, text))
        total_chars += len(text)

    # Sort by page number for coherent reading order
    selected.sort(key=lambda x: x[0])
    return "\n\n".join(f"--- Page {pn + 1} ---\n{t}" for pn, t in selected)


class LLMFilingExtractor:
    """Universal financial filing extractor using LLM."""

    def __init__(self, llm_client=None):
        """Initialize with an LLM client (Gemini or Claude).

        Parameters
        ----------
        llm_client:
            An LLM client from llm_factory (GeminiClient or ClaudeClient).
            If None, extraction from unstructured formats will fail gracefully.
        """
        self._llm = llm_client

    # ------------------------------------------------------------------
    # iXBRL extraction (structured, no LLM needed)
    # ------------------------------------------------------------------

    def extract_from_ixbrl(
        self,
        ixbrl_text: str,
        market_id: str = "",
    ) -> ExtractionResult:
        """Extract financials from iXBRL-tagged HTML.

        Uses the ixbrl-parse library (already installed) for structured
        extraction. No LLM call needed.
        """
        result = ExtractionResult(source_format="ixbrl")

        try:
            from ixbrl_parse import IXBRL
        except ImportError:
            result.error = "ixbrl-parse not installed"
            return result

        try:
            doc = IXBRL(io.StringIO(ixbrl_text))
            facts = {}
            for fact in doc.numeric:
                name = fact.get("name", "")
                value = fact.get("value")
                context = fact.get("context", {})
                if value is not None:
                    # Store by concept name
                    facts[name] = {
                        "value": float(value),
                        "context": context,
                    }

            if not facts:
                result.error = "No numeric facts found in iXBRL"
                return result

            # Map iXBRL concept names to canonical fields
            data = self._map_ixbrl_concepts(facts, market_id)
            result.data = data
            result.success = bool(data)
            result.validation_passed = self._validate_accounting_identities(data)
            return result

        except Exception as exc:
            result.error = f"iXBRL parsing failed: {exc}"
            return result

    # ------------------------------------------------------------------
    # HTML extraction (semi-structured)
    # ------------------------------------------------------------------

    def extract_from_html(
        self,
        html_text: str,
        market_id: str = "",
    ) -> ExtractionResult:
        """Extract financials from HTML filing (with tables)."""
        result = ExtractionResult(source_format="html")

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            result.error = "beautifulsoup4 not installed"
            return result

        try:
            soup = BeautifulSoup(html_text, "html.parser")

            # Extract all tables
            tables = soup.find_all("table")
            table_texts = []
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    row_text = " | ".join(c.get_text(strip=True) for c in cells)
                    if row_text.strip():
                        table_texts.append(row_text)

            if not table_texts:
                # Fall back to full text
                text = soup.get_text(separator="\n", strip=True)
            else:
                text = "\n".join(table_texts)

            # Truncate to ~12K chars for LLM context
            text = text[:12000]

            return self._extract_via_llm(text, market_id, "html")

        except Exception as exc:
            result.error = f"HTML extraction failed: {exc}"
            return result

    # ------------------------------------------------------------------
    # PDF extraction (unstructured)
    # ------------------------------------------------------------------

    def extract_from_pdf(
        self,
        pdf_bytes: bytes,
        market_id: str = "",
    ) -> ExtractionResult:
        """Extract financials from a PDF filing."""
        result = ExtractionResult(source_format="pdf")

        try:
            import pdfplumber
        except ImportError:
            result.error = "pdfplumber not installed (pip install pdfplumber)"
            return result

        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                # Extract text per page with table preference
                page_texts: list[tuple[int, str]] = []
                for i, page in enumerate(pdf.pages):
                    page_parts = []
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                if row:
                                    row_text = " | ".join(
                                        str(cell).strip() for cell in row if cell
                                    )
                                    if row_text:
                                        page_parts.append(row_text)
                    else:
                        text = page.extract_text()
                        if text:
                            page_parts.append(text)
                    if page_parts:
                        page_texts.append((i, "\n".join(page_parts)))

            if not page_texts:
                result.error = "No text extracted from PDF"
                return result

            # Smart page selection: score pages for financial content
            # and select the top-scoring ones (up to 12K chars).
            # This ensures financial statements (often buried in pages
            # 20-60 of a 100+ page document) reach the LLM.
            text = _select_top_pages(page_texts, max_chars=12000)

            if not text.strip():
                # Fallback: use first 12K chars
                text = "\n".join(t for _, t in page_texts)[:12000]

            logger.info(
                "PDF smart selection: %d pages total, %d chars selected",
                len(page_texts), len(text),
            )

            return self._extract_via_llm(text, market_id, "pdf")

        except Exception as exc:
            result.error = f"PDF extraction failed: {exc}"
            return result

    # ------------------------------------------------------------------
    # LLM extraction core
    # ------------------------------------------------------------------

    def _extract_via_llm(
        self,
        text: str,
        market_id: str,
        source_format: str,
    ) -> ExtractionResult:
        """Send text to LLM for structured extraction."""
        result = ExtractionResult(source_format=source_format)

        if self._llm is None:
            result.error = "No LLM client available for extraction"
            return result

        # Build taxonomy hint for this market (empty string if unknown market)
        hint = get_taxonomy_hint(market_id)
        taxonomy_section = f"\n{hint}\n" if hint else ""
        prompt = _EXTRACTION_PROMPT.format(
            taxonomy_hint=taxonomy_section,
            text=text,
        )

        try:
            # Use the LLM's generate method
            response = self._llm.generate(prompt)
            if not response:
                result.error = "LLM returned empty response"
                return result

            # Parse JSON from response
            data = self._parse_llm_json(response)
            if not data:
                result.error = "Could not parse JSON from LLM response"
                return result

            # Clean and validate
            cleaned = self._clean_extracted_data(data)
            result.data = cleaned
            result.success = bool(cleaned)
            result.validation_passed = self._validate_accounting_identities(cleaned)

            if not result.validation_passed:
                result.validation_notes.append(
                    "Accounting identity check failed (assets != liabilities + equity)"
                )

            return result

        except Exception as exc:
            result.error = f"LLM extraction failed: {exc}"
            return result

    def validate_existing_data(
        self,
        data: dict[str, Any],
        filing_text: str = "",
        market_id: str = "",
    ) -> ExtractionResult:
        """Validate/cross-check existing extracted data using LLM.

        Can be used on data from any source (yfinance, EDGAR, etc.)
        to verify against the original filing text.
        """
        result = ExtractionResult(source_format="validation")

        if self._llm is None:
            # No LLM -- just do accounting identity check
            result.validation_passed = self._validate_accounting_identities(data)
            result.data = data
            result.success = True
            return result

        if filing_text:
            prompt = _VALIDATION_PROMPT.format(
                market=market_id,
                data=json.dumps(data, indent=2),
                text=filing_text[:8000],
            )
            try:
                response = self._llm.generate(prompt)
                validated = self._parse_llm_json(response)
                if validated:
                    notes = validated.pop("validation_notes", [])
                    if isinstance(notes, str):
                        notes = [notes]
                    result.validation_notes = notes
                    result.data = self._clean_extracted_data(validated)
                    result.success = True
                    result.validation_passed = not bool(notes) or all(
                        "correct" in n.lower() or "no issues" in n.lower()
                        for n in notes
                    )
                    return result
            except Exception as exc:
                logger.debug("LLM validation failed: %s", exc)

        # Fallback: accounting identity check only
        result.validation_passed = self._validate_accounting_identities(data)
        result.data = data
        result.success = True
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_llm_json(self, response: str) -> dict | None:
        """Extract JSON from LLM response (handles markdown fences)."""
        if not response:
            return None

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fence
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _clean_extracted_data(self, data: dict) -> dict:
        """Clean and normalize extracted financial data."""
        cleaned = {}
        for key in EXTRACTION_FIELDS:
            val = data.get(key)
            if val is not None:
                try:
                    cleaned[key] = float(val)
                except (TypeError, ValueError):
                    pass

        # Metadata
        for meta_key in ("report_date", "filing_date", "currency", "period_type"):
            if meta_key in data and data[meta_key]:
                cleaned[meta_key] = str(data[meta_key])

        return cleaned

    def _validate_accounting_identities(self, data: dict) -> bool:
        """Check basic accounting identities."""
        issues = []

        # Balance sheet: assets = liabilities + equity
        assets = data.get("total_assets")
        liabilities = data.get("total_liabilities")
        equity = data.get("total_equity")

        if all(v is not None for v in [assets, liabilities, equity]):
            expected = liabilities + equity
            if assets > 0 and abs(assets - expected) / assets > 0.05:
                issues.append(
                    f"total_assets ({assets}) != total_liabilities ({liabilities}) + "
                    f"total_equity ({equity}) = {expected}"
                )

        # Cash flow: FCF ~= operating_cf - capex
        ocf = data.get("operating_cash_flow")
        capex = data.get("capex")
        fcf = data.get("free_cash_flow")

        if all(v is not None for v in [ocf, capex, fcf]):
            expected_fcf = ocf + capex  # capex is typically negative
            if abs(fcf) > 0 and abs(fcf - expected_fcf) / abs(fcf) > 0.1:
                issues.append(
                    f"free_cash_flow ({fcf}) != operating_cf ({ocf}) + capex ({capex}) = {expected_fcf}"
                )

        if issues:
            logger.debug("Accounting identity issues: %s", issues)

        return len(issues) == 0

    def _map_ixbrl_concepts(
        self,
        facts: dict[str, dict],
        market_id: str,
    ) -> dict:
        """Map iXBRL concept names to canonical financial fields."""
        # IFRS concept name patterns -> canonical names
        ifrs_map = {
            "Revenue": "revenue",
            "CostOfSales": "cost_of_revenue",
            "GrossProfit": "gross_profit",
            "ProfitLossFromOperatingActivities": "operating_income",
            "ProfitLossBeforeTax": "ebit",
            "ProfitLoss": "net_income",
            "IncomeTaxExpenseContinuingOperations": "taxes",
            "FinanceCosts": "interest_expense",
            "Assets": "total_assets",
            "Liabilities": "total_liabilities",
            "Equity": "total_equity",
            "CurrentAssets": "current_assets",
            "CurrentLiabilities": "current_liabilities",
            "CashAndCashEquivalents": "cash_and_equivalents",
            "NoncurrentBorrowings": "long_term_debt",
            "CurrentBorrowings": "short_term_debt",
            "RetainedEarnings": "retained_earnings",
            "CashFlowsFromUsedInOperatingActivities": "operating_cash_flow",
            "PurchaseOfPropertyPlantAndEquipment": "capex",
            "CashFlowsFromUsedInInvestingActivities": "investing_cf",
            "CashFlowsFromUsedInFinancingActivities": "financing_cf",
            "DividendsPaid": "dividends_paid",
            "EarningsPerShare": "eps",
            "DilutedEarningsPerShare": "eps_diluted",
        }

        result = {}
        for fact_name, fact_data in facts.items():
            # Try exact match first, then partial match
            canonical = None
            for pattern, canonical_name in ifrs_map.items():
                if pattern.lower() in fact_name.lower():
                    canonical = canonical_name
                    break

            if canonical and canonical not in result:
                result[canonical] = fact_data["value"]

        return result

    # ------------------------------------------------------------------
    # Convenience: extract to canonical long-format DataFrame
    # ------------------------------------------------------------------

    def to_canonical_dataframe(
        self,
        extraction: ExtractionResult,
        market_id: str = "",
    ) -> pd.DataFrame:
        """Convert extraction result to canonical long-format DataFrame.

        Returns DataFrame with columns: canonical_name, value, report_date, filing_date.
        """
        if not extraction.success or not extraction.data:
            return pd.DataFrame()

        data = extraction.data
        report_date = data.get("report_date", "")
        filing_date = data.get("filing_date", report_date)
        currency = data.get("currency", "")

        records = []
        for field_name in EXTRACTION_FIELDS:
            value = data.get(field_name)
            if value is not None:
                records.append({
                    "canonical_name": field_name,
                    "value": float(value),
                    "report_date": pd.Timestamp(report_date) if report_date else pd.NaT,
                    "filing_date": pd.Timestamp(filing_date) if filing_date else pd.NaT,
                    "currency": currency,
                    "source": f"llm_extract_{extraction.source_format}",
                    "market_id": market_id,
                })

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)
