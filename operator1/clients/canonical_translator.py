"""Canonical data translator -- normalizes all API responses for processing.

Each regional PIT client returns data in its own format (XBRL concepts,
local field names, local date formats, local currencies).  This module
translates raw API data into the canonical field format defined in
``config/canonical_fields.yml`` so downstream processing modules receive
uniform input regardless of the data source.

Translation responsibilities:
  1. Map region-specific concept/field names to canonical names
  2. Ensure filing_date and report_date columns are present and parsed
  3. Normalize numeric values (strip commas, handle local formats)
  4. Convert local date formats (ROC calendar, Japanese era, etc.)
  5. Tag each row with source metadata for audit trail

Usage:
    from operator1.clients.canonical_translator import translate_financials
    canonical_df = translate_financials(raw_df, market_id="jp_jquants")
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical field names (matching config/canonical_fields.yml)
# ---------------------------------------------------------------------------

# Income statement canonical names
CANONICAL_INCOME = {
    "revenue", "gross_profit", "operating_income", "ebit", "ebitda",
    "net_income", "taxes", "interest_expense", "cost_of_revenue",
    "rd_expenses", "sga_expenses", "eps_basic", "eps_diluted",
}

# Balance sheet canonical names
CANONICAL_BALANCE = {
    "total_assets", "total_liabilities", "total_equity",
    "current_assets", "current_liabilities", "cash_and_equivalents",
    "short_term_debt", "long_term_debt", "total_debt",
    "retained_earnings", "goodwill", "intangible_assets",
    "receivables", "inventory", "payables",
}

# Cash flow canonical names
CANONICAL_CASHFLOW = {
    "operating_cash_flow", "investing_cf", "financing_cf",
    "capex", "free_cash_flow", "dividends_paid",
}

# Profile canonical field names
CANONICAL_PROFILE = {
    "name", "ticker", "isin", "country", "sector", "industry",
    "sub_industry", "exchange", "currency", "market_cap",
    "shares_outstanding", "lei", "cik",
}


# ---------------------------------------------------------------------------
# Region-specific concept -> canonical name mappings
# ---------------------------------------------------------------------------

# ESEF / IFRS concepts (EU, and any IFRS reporter)
_IFRS_MAP: dict[str, str] = {
    # Income statement
    "ifrs-full:Revenue": "revenue",
    "ifrs-full:CostOfSales": "cost_of_revenue",
    "ifrs-full:GrossProfit": "gross_profit",
    "ifrs-full:ProfitLossFromOperatingActivities": "operating_income",
    "ifrs-full:ProfitLoss": "net_income",
    "ifrs-full:ProfitLossBeforeTax": "ebit",
    "ifrs-full:IncomeTaxExpenseContinuingOperations": "taxes",
    "ifrs-full:FinanceCosts": "interest_expense",
    "ifrs-full:SellingGeneralAndAdministrativeExpense": "sga_expenses",
    "ifrs-full:ResearchAndDevelopmentExpense": "rd_expenses",
    # Balance sheet
    "ifrs-full:Assets": "total_assets",
    "ifrs-full:Liabilities": "total_liabilities",
    "ifrs-full:Equity": "total_equity",
    "ifrs-full:CurrentAssets": "current_assets",
    "ifrs-full:CurrentLiabilities": "current_liabilities",
    "ifrs-full:CashAndCashEquivalents": "cash_and_equivalents",
    "ifrs-full:NoncurrentLiabilities": "long_term_debt",
    "ifrs-full:RetainedEarnings": "retained_earnings",
    "ifrs-full:Goodwill": "goodwill",
    "ifrs-full:IntangibleAssetsOtherThanGoodwill": "intangible_assets",
    "ifrs-full:TradeAndOtherCurrentReceivables": "receivables",
    "ifrs-full:Inventories": "inventory",
    "ifrs-full:TradeAndOtherCurrentPayables": "payables",
    # Cash flow
    "ifrs-full:CashFlowsFromUsedInOperatingActivities": "operating_cash_flow",
    "ifrs-full:CashFlowsFromUsedInInvestingActivities": "investing_cf",
    "ifrs-full:CashFlowsFromUsedInFinancingActivities": "financing_cf",
    "ifrs-full:PurchaseOfPropertyPlantAndEquipment": "capex",
    "ifrs-full:DividendsPaid": "dividends_paid",
    # Additional IFRS concepts for full coverage
    "ifrs-full:BasicEarningsLossPerShare": "eps_basic",
    "ifrs-full:DilutedEarningsLossPerShare": "eps_diluted",
    "ifrs-full:CurrentBorrowings": "short_term_debt",
    "ifrs-full:ShorttermBorrowings": "short_term_debt",
    "ifrs-full:Borrowings": "total_debt",
    "ifrs-full:NoncurrentBorrowings": "long_term_debt",
    "ifrs-full:DepreciationAndAmortisationExpense": "ebitda",  # proxy: D&A component
    # free_cash_flow: computed downstream (operating_cash_flow - capex)
}

# Japan GAAP (JPPFS) concepts (EDINET)
_JPPFS_MAP: dict[str, str] = {
    # Income statement
    "jppfs_cor:NetSales": "revenue",
    "jppfs_cor:Revenue": "revenue",
    "jppfs_cor:CostOfSales": "cost_of_revenue",
    "jppfs_cor:GrossProfit": "gross_profit",
    "jppfs_cor:OperatingIncome": "operating_income",
    "jppfs_cor:OrdinaryIncome": "ebit",
    "jppfs_cor:ProfitLoss": "net_income",
    "jppfs_cor:ProfitLossAttributableToOwnersOfParent": "net_income",
    "jppfs_cor:IncomeTaxes": "taxes",
    "jppfs_cor:InterestExpense": "interest_expense",
    "jppfs_cor:SellingGeneralAndAdministrativeExpenses": "sga_expenses",
    # Balance sheet
    "jppfs_cor:TotalAssets": "total_assets",
    "jppfs_cor:TotalLiabilities": "total_liabilities",
    "jppfs_cor:NetAssets": "total_equity",
    "jppfs_cor:CurrentAssets": "current_assets",
    "jppfs_cor:CurrentLiabilities": "current_liabilities",
    "jppfs_cor:CashAndDeposits": "cash_and_equivalents",
    "jppfs_cor:ShortTermLoansPayable": "short_term_debt",
    "jppfs_cor:LongTermLoansPayable": "long_term_debt",
    "jppfs_cor:RetainedEarnings": "retained_earnings",
    "jppfs_cor:Goodwill": "goodwill",
    "jppfs_cor:NotesAndAccountsReceivableTrade": "receivables",
    "jppfs_cor:MerchandiseAndFinishedGoods": "inventory",
    "jppfs_cor:NotesAndAccountsPayableTrade": "payables",
    # Cash flow
    "jppfs_cor:NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "jppfs_cor:NetCashProvidedByUsedInInvestingActivities": "investing_cf",
    "jppfs_cor:NetCashProvidedByUsedInFinancingActivities": "financing_cf",
    "jppfs_cor:PurchaseOfPropertyPlantAndEquipment": "capex",
}

# Taiwan GAAP / TIFRS concepts (MOPS)
_TIFRS_MAP: dict[str, str] = {
    # Chinese field names from MOPS responses
    "營業收入合計": "revenue",
    "營業成本合計": "cost_of_revenue",
    "營業毛利（毛損）": "gross_profit",
    "營業利益（損失）": "operating_income",
    "本期淨利（淨損）": "net_income",
    "稅前淨利（淨損）": "ebit",
    "所得稅費用（利益）": "taxes",
    "利息費用": "interest_expense",
    "推銷費用": "sga_expenses",
    "研究發展費用": "rd_expenses",
    # Balance sheet
    "資產總計": "total_assets",
    "負債總計": "total_liabilities",
    "權益總計": "total_equity",
    "流動資產合計": "current_assets",
    "流動負債合計": "current_liabilities",
    "現金及約當現金": "cash_and_equivalents",
    "短期借款": "short_term_debt",
    "長期借款": "long_term_debt",
    "保留盈餘": "retained_earnings",
    "商譽": "goodwill",
    "應收帳款淨額": "receivables",
    "存貨": "inventory",
    "應付帳款": "payables",
    # Cash flow
    "營業活動之淨現金流入（流出）": "operating_cash_flow",
    "投資活動之淨現金流入（流出）": "investing_cf",
    "籌資活動之淨現金流入（流出）": "financing_cf",
    "取得不動產、廠房及設備": "capex",
}

# Brazil CVM account codes -> canonical names
# CVM uses hierarchical account codes (e.g. 3.01 = Revenue)
_CVM_ACCOUNT_MAP: dict[str, str] = {
    # DRE (Income statement)
    "3.01": "revenue",
    "3.02": "cost_of_revenue",
    "3.03": "gross_profit",
    "3.05": "operating_income",
    "3.11": "net_income",
    "3.08": "ebit",
    "3.08.01": "taxes",
    "3.06.02": "interest_expense",
    # BPA (Balance sheet - assets)
    "1": "total_assets",
    "1.01": "current_assets",
    "1.01.01": "cash_and_equivalents",
    "1.01.03": "receivables",
    "1.01.04": "inventory",
    "1.02.04": "goodwill",
    "1.02.04.01": "intangible_assets",
    # BPP (Balance sheet - liabilities & equity)
    "2": "total_liabilities",
    "2.01": "current_liabilities",
    "2.01.04": "short_term_debt",
    "2.02.01": "long_term_debt",
    "2.03": "total_equity",
    "2.03.04": "retained_earnings",
    "2.01.02": "payables",
    # DFC (Cash flow)
    "6.01": "operating_cash_flow",
    "6.02": "investing_cf",
    "6.03": "financing_cf",
    "6.02.01": "capex",
    # Missing canonical fields -- added for full coverage
    "6.03.04": "dividends_paid",        # Dividends paid to shareholders
    "3.09": "ebitda",                    # EBITDA (some CVM filers report this)
    "3.11.01": "eps_basic",             # Earnings per share (basic)
    "3.11.02": "eps_diluted",           # Earnings per share (diluted)
    "3.04": "sga_expenses",             # Selling, general & administrative
    "3.04.01": "rd_expenses",           # R&D expenses (subset of SGA in CVM)
    "2.01.04+2.02.01": "total_debt",    # Not a real code; handled via computation
    # free_cash_flow: computed downstream (operating_cash_flow - capex)
}

# Chile CMF / FECU concepts -> canonical names
# FECU uses IFRS-based reporting with Spanish field names
_CMF_MAP: dict[str, str] = {
    # Income statement (Estado de Resultados)
    "Ingresos de actividades ordinarias": "revenue",
    "Costo de ventas": "cost_of_revenue",
    "Ganancia bruta": "gross_profit",
    "Ganancia (pérdida) por actividades de operación": "operating_income",
    "Ganancia (pérdida)": "net_income",
    "Ganancia (pérdida), antes de impuestos": "ebit",
    "Gasto por impuestos a las ganancias": "taxes",
    "Costos financieros": "interest_expense",
    # Balance sheet
    "Total de activos": "total_assets",
    "Total de pasivos": "total_liabilities",
    "Total de patrimonio": "total_equity",
    "Activos corrientes totales": "current_assets",
    "Pasivos corrientes totales": "current_liabilities",
    "Efectivo y equivalentes al efectivo": "cash_and_equivalents",
    "Otros pasivos financieros corrientes": "short_term_debt",
    "Otros pasivos financieros no corrientes": "long_term_debt",
    "Ganancias acumuladas": "retained_earnings",
    "Plusvalía": "goodwill",
    "Deudores comerciales y otras cuentas por cobrar corrientes": "receivables",
    "Inventarios corrientes": "inventory",
    "Cuentas por pagar comerciales y otras cuentas por pagar": "payables",
    # Cash flow
    "Flujos de efectivo procedentes de actividades de operación": "operating_cash_flow",
    "Flujos de efectivo procedentes de actividades de inversión": "investing_cf",
    "Flujos de efectivo procedentes de actividades de financiación": "financing_cf",
    "Compras de propiedades, planta y equipo": "capex",
}

# UK GAAP concepts (Companies House iXBRL)
_UKGAAP_MAP: dict[str, str] = {
    # Income statement -- UK GAAP
    "uk-gaap:Turnover": "revenue",
    "uk-gaap:TurnoverRevenue": "revenue",
    "uk-gaap:GrossProfitLoss": "gross_profit",
    "uk-gaap:OperatingProfitLoss": "operating_income",
    "uk-gaap:ProfitLossOnOrdinaryActivitiesBeforeTax": "ebit",
    "uk-gaap:ProfitLossForPeriod": "net_income",
    "uk-gaap:TaxOnProfitOnOrdinaryActivities": "taxes",
    "uk-gaap:InterestPayableAndSimilarCharges": "interest_expense",
    # Income statement -- FRS 102 (iXBRL tags from Companies House filings)
    "frs102:TurnoverRevenue": "revenue",
    "frs102:Turnover": "revenue",
    "frs102:GrossProfit": "gross_profit",
    "frs102:GrossProfitLoss": "gross_profit",
    "frs102:OperatingProfit": "operating_income",
    "frs102:OperatingProfitLoss": "operating_income",
    "frs102:ProfitLossBeforeTax": "ebit",
    "frs102:ProfitLossForPeriod": "net_income",
    "frs102:TaxOnProfit": "taxes",
    "frs102:InterestPayable": "interest_expense",
    # Income statement -- bare names (no namespace prefix)
    "Turnover": "revenue",
    "TurnoverRevenue": "revenue",
    "OperatingProfit": "operating_income",
    "ProfitLossBeforeTax": "ebit",
    "ProfitLossForPeriod": "net_income",
    # Balance sheet -- UK GAAP
    "uk-gaap:FixedAssets": "total_assets",
    "uk-gaap:TotalAssetsLessCurrentLiabilities": "total_assets",
    "uk-gaap:Creditors": "total_liabilities",
    "uk-gaap:ShareholderFunds": "total_equity",
    "uk-gaap:CurrentAssets": "current_assets",
    "uk-gaap:CreditorsDueWithinOneYear": "current_liabilities",
    "uk-gaap:CashBankInHand": "cash_and_equivalents",
    "uk-gaap:CreditorsDueAfterOneYear": "long_term_debt",
    # Balance sheet -- FRS 102
    "frs102:FixedAssets": "total_assets",
    "frs102:TotalAssetsLessCurrentLiabilities": "total_assets",
    "frs102:NetCurrentAssetsLiabilities": "current_assets",
    "frs102:CalledUpShareCapital": "total_equity",
    "frs102:ShareholderFunds": "total_equity",
    "frs102:Creditors": "total_liabilities",
    "frs102:CreditorsDueWithinOneYear": "current_liabilities",
    "frs102:CreditorsDueAfterOneYear": "long_term_debt",
    "frs102:CashBankInHand": "cash_and_equivalents",
    "frs102:CashAtBankAndInHand": "cash_and_equivalents",
    # Balance sheet -- bare names
    "FixedAssets": "total_assets",
    "ShareholderFunds": "total_equity",
    "CalledUpShareCapital": "total_equity",
    # Cash flow -- UK GAAP
    "uk-gaap:NetCashInflowOutflowFromOperatingActivities": "operating_cash_flow",
    "uk-gaap:NetCashInflowOutflowFromInvestingActivities": "investing_cf",
    "uk-gaap:NetCashInflowOutflowFromFinancingActivities": "financing_cf",
    # Cash flow -- FRS 102
    "frs102:NetCashFromOperatingActivities": "operating_cash_flow",
    "frs102:NetCashUsedInInvestingActivities": "investing_cf",
    "frs102:NetCashUsedInFinancingActivities": "financing_cf",
}

# South Korea DART / K-IFRS concepts (Korean account names from DART API)
_DART_MAP: dict[str, str] = {
    # Income statement (Korean account names returned by DART API)
    "매출액": "revenue",
    "수익(매출액)": "revenue",
    "매출원가": "cost_of_revenue",
    "매출총이익": "gross_profit",
    "영업이익": "operating_income",
    "영업이익(손실)": "operating_income",
    "법인세비용차감전순이익(손실)": "ebit",
    "당기순이익": "net_income",
    "당기순이익(손실)": "net_income",
    "법인세비용": "taxes",
    "이자비용": "interest_expense",
    "금융비용": "interest_expense",
    "판매비와관리비": "sga_expenses",
    # Balance sheet
    "자산총계": "total_assets",
    "부채총계": "total_liabilities",
    "자본총계": "total_equity",
    "유동자산": "current_assets",
    "유동부채": "current_liabilities",
    "현금및현금성자산": "cash_and_equivalents",
    "단기차입금": "short_term_debt",
    "장기차입금": "long_term_debt",
    "이익잉여금": "retained_earnings",
    "영업권": "goodwill",
    "매출채권 및 기타유동채권": "receivables",
    "재고자산": "inventory",
    "매입채무 및 기타유동채무": "payables",
    # Cash flow
    "영업활동현금흐름": "operating_cash_flow",
    "영업활동으로인한현금흐름": "operating_cash_flow",
    "투자활동현금흐름": "investing_cf",
    "투자활동으로인한현금흐름": "investing_cf",
    "재무활동현금흐름": "financing_cf",
    "재무활동으로인한현금흐름": "financing_cf",
    "유형자산의 취득": "capex",
}

# Chinese Accounting Standards (CAS) -- Simplified Chinese field names
# Used by akshare/Sina Finance for SSE/SZSE data
# Research: .roo/research/cn-sse-2026-02-24.md
_CAS_MAP: dict[str, str] = {
    # Income statement (利润表)
    "营业收入": "revenue",
    "营业总收入": "revenue",
    "营业成本": "cost_of_revenue",
    "营业总成本": "cost_of_revenue",
    "营业利润": "operating_income",
    "利润总额": "ebit",
    "净利润": "net_income",
    "归属于母公司所有者的净利润": "net_income",
    "所得税费用": "taxes",
    "利息支出": "interest_expense",
    "财务费用": "interest_expense",
    "销售费用": "sga_expenses",
    "管理费用": "sga_expenses",
    "研发费用": "rd_expenses",
    "毛利润": "gross_profit",
    # Balance sheet (资产负债表)
    "资产总计": "total_assets",
    "负债合计": "total_liabilities",
    "所有者权益合计": "total_equity",
    "归属于母公司所有者权益合计": "total_equity",
    "流动资产合计": "current_assets",
    "流动负债合计": "current_liabilities",
    "货币资金": "cash_and_equivalents",
    "短期借款": "short_term_debt",
    "长期借款": "long_term_debt",
    "未分配利润": "retained_earnings",
    "商誉": "goodwill",
    "无形资产": "intangible_assets",
    "应收账款": "receivables",
    "存货": "inventory",
    "应付账款": "payables",
    # Cash flow (现金流量表)
    "经营活动产生的现金流量净额": "operating_cash_flow",
    "投资活动产生的现金流量净额": "investing_cf",
    "筹资活动产生的现金流量净额": "financing_cf",
    "购建固定资产、无形资产和其他长期资产支付的现金": "capex",
    "分配股利、利润或偿付利息支付的现金": "dividends_paid",
}

# US GAAP concepts (SEC EDGAR) -- already canonical in the client,
# but included here for completeness and for cross-referencing
_USGAAP_MAP: dict[str, str] = {
    # Namespaced form (for cross-region IFRS/XBRL usage)
    "us-gaap:Revenues": "revenue",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "us-gaap:CostOfRevenue": "cost_of_revenue",
    "us-gaap:CostOfGoodsAndServicesSold": "cost_of_revenue",
    "us-gaap:GrossProfit": "gross_profit",
    "us-gaap:OperatingIncomeLoss": "operating_income",
    "us-gaap:NetIncomeLoss": "net_income",
    "us-gaap:IncomeTaxExpenseBenefit": "taxes",
    "us-gaap:InterestExpense": "interest_expense",
    "us-gaap:Assets": "total_assets",
    "us-gaap:Liabilities": "total_liabilities",
    "us-gaap:StockholdersEquity": "total_equity",
    "us-gaap:AssetsCurrent": "current_assets",
    "us-gaap:LiabilitiesCurrent": "current_liabilities",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue": "cash_and_equivalents",
    "us-gaap:ShortTermBorrowings": "short_term_debt",
    "us-gaap:LongTermDebt": "long_term_debt",
    "us-gaap:NetCashProvidedByOperatingActivities": "operating_cash_flow",
    "us-gaap:NetCashProvidedByInvestingActivities": "investing_cf",
    "us-gaap:NetCashProvidedByFinancingActivities": "financing_cf",
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    # Bare form (SEC EDGAR companyfacts uses bare concept names)
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "CostOfRevenue": "cost_of_revenue",
    "CostOfGoodsAndServicesSold": "cost_of_revenue",
    "GrossProfit": "gross_profit",
    "OperatingIncomeLoss": "operating_income",
    "NetIncomeLoss": "net_income",
    "EarningsPerShareBasic": "eps_basic",
    "EarningsPerShareDiluted": "eps_diluted",
    "IncomeTaxExpenseBenefit": "taxes",
    "InterestExpense": "interest_expense",
    "SellingGeneralAndAdministrativeExpense": "sga_expenses",
    "ResearchAndDevelopmentExpense": "rd_expenses",
    "Assets": "total_assets",
    "Liabilities": "total_liabilities",
    "StockholdersEquity": "total_equity",
    "AssetsCurrent": "current_assets",
    "LiabilitiesCurrent": "current_liabilities",
    "CashAndCashEquivalentsAtCarryingValue": "cash_and_equivalents",
    "ShortTermBorrowings": "short_term_debt",
    "LongTermDebt": "long_term_debt",
    "AccountsReceivableNetCurrent": "receivables",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "NetCashProvidedByUsedInInvestingActivities": "investing_cf",
    "NetCashProvidedByUsedInFinancingActivities": "financing_cf",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "PaymentsOfDividends": "dividends_paid",
    "PaymentsForRepurchaseOfCommonStock": "stock_buybacks",
    # Missing canonical fields -- added for full coverage
    # ebit (namespaced + bare)
    "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes": "ebit",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxes": "ebit",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "ebit",
    # ebitda -- SEC filers rarely report EBITDA directly; use operating income as proxy
    "us-gaap:OperatingExpenses": "ebitda",  # some filers report this
    # goodwill
    "us-gaap:Goodwill": "goodwill",
    "Goodwill": "goodwill",
    # intangible_assets
    "us-gaap:IntangibleAssetsNetExcludingGoodwill": "intangible_assets",
    "IntangibleAssetsNetExcludingGoodwill": "intangible_assets",
    "FiniteLivedIntangibleAssetsNet": "intangible_assets",
    # inventory
    "us-gaap:InventoryNet": "inventory",
    "InventoryNet": "inventory",
    # payables
    "us-gaap:AccountsPayableCurrent": "payables",
    "AccountsPayableCurrent": "payables",
    "AccountsPayableAndAccruedLiabilitiesCurrent": "payables",
    # retained_earnings
    "us-gaap:RetainedEarningsAccumulatedDeficit": "retained_earnings",
    "RetainedEarningsAccumulatedDeficit": "retained_earnings",
    # total_debt
    "us-gaap:DebtAndCapitalLeaseObligations": "total_debt",
    "DebtAndCapitalLeaseObligations": "total_debt",
    "LongTermDebtAndCapitalLeaseObligations": "total_debt",
    # free_cash_flow -- not a GAAP concept, but some filers report it
    # We compute it downstream: operating_cash_flow - capex
}

# Market -> accounting standard label (for cross-standard comparison caveats)
MARKET_ACCOUNTING_STANDARD: dict[str, str] = {
    "us_sec_edgar": "US-GAAP",
    "uk_companies_house": "UK-GAAP / IFRS",
    "eu_esef": "IFRS",
    "fr_esef": "IFRS",
    "de_esef": "IFRS",
    "jp_jquants": "JPPFS (Japan GAAP) / IFRS",
    "kr_dart": "K-IFRS",
    "tw_mops": "TIFRS (Taiwan IFRS)",
    "br_cvm": "BR-GAAP / IFRS",
    "cl_cmf": "IFRS (Chile)",
    "cn_sse": "CAS (Chinese Accounting Standards)",
    "in_bse": "Ind AS (IFRS-converged)",
    "ca_sedar": "IFRS / US-GAAP",
    "au_asx": "AASB (IFRS-based)",
    "hk_hkex": "HKFRS (IFRS-identical)",
    "sg_sgx": "SFRS(I) (IFRS-identical)",
    "mx_bmv": "IFRS",
    "za_jse": "IFRS",
    "ch_six": "IFRS / Swiss GAAP",
    "sa_tadawul": "IFRS",
    "ae_dfm": "IFRS",
    "nl_esef": "IFRS",
    "es_esef": "IFRS",
    "it_esef": "IFRS",
    "se_esef": "IFRS",
}


# Master mapping: market_id -> concept map
_MARKET_CONCEPT_MAPS: dict[str, dict[str, str]] = {
    "us_sec_edgar": _USGAAP_MAP,
    "uk_companies_house": {**_UKGAAP_MAP, **_IFRS_MAP},  # UK: UK GAAP + IFRS
    "eu_esef": _IFRS_MAP,
    "fr_esef": _IFRS_MAP,
    "de_esef": _IFRS_MAP,
    "jp_jquants": {**_JPPFS_MAP, **_IFRS_MAP},    # Japan: JPPFS + IFRS adopters
    "tw_mops": {**_TIFRS_MAP, **_IFRS_MAP},      # Taiwan: TIFRS + IFRS
    "br_cvm": _CVM_ACCOUNT_MAP,
    "cl_cmf": {**_CMF_MAP, **_IFRS_MAP},          # Chile: FECU (IFRS) + IFRS
    "kr_dart": {**_DART_MAP, **_IFRS_MAP},         # Korea: K-IFRS + IFRS
    # Phase 2 -- ESEF-based markets (all use IFRS)
    "nl_esef": _IFRS_MAP,
    "es_esef": _IFRS_MAP,
    "it_esef": _IFRS_MAP,
    "se_esef": _IFRS_MAP,
    # Phase 2 -- non-ESEF markets (IFRS adopters or close equivalents)
    "ca_sedar": {**_IFRS_MAP, **_USGAAP_MAP},     # Canada: IFRS + some US GAAP filers
    "au_asx": _IFRS_MAP,                           # Australia: AASB (IFRS-based)
    "in_bse": _IFRS_MAP,                           # India: Ind AS (IFRS-converged)
    "cn_sse": {**_CAS_MAP, **_IFRS_MAP},             # China: CAS (Simplified Chinese) + IFRS fallback
    "hk_hkex": _IFRS_MAP,                          # Hong Kong: HKFRS (IFRS-identical)
    "sg_sgx": _IFRS_MAP,                           # Singapore: SFRS(I) (IFRS-identical)
    "mx_bmv": _IFRS_MAP,                           # Mexico: IFRS mandatory
    "za_jse": _IFRS_MAP,                           # South Africa: IFRS mandatory
    "ch_six": _IFRS_MAP,                           # Switzerland: IFRS or Swiss GAAP
    "sa_tadawul": _IFRS_MAP,                       # Saudi Arabia: IFRS mandatory
    "ae_dfm": _IFRS_MAP,                           # UAE: IFRS mandatory
}


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

def _normalize_roc_date(date_str: str) -> str:
    """Convert ROC (Taiwan) date format to ISO format.

    ROC uses year = CE_year - 1911.  Formats: 113/01/15 or 1130115.
    """
    if not date_str or not isinstance(date_str, str):
        return date_str

    # Format: 113/01/15
    match = re.match(r"(\d{2,3})/(\d{2})/(\d{2})", date_str)
    if match:
        roc_year = int(match.group(1))
        return f"{roc_year + 1911}-{match.group(2)}-{match.group(3)}"

    # Format: 1130115 (7 digits)
    match = re.match(r"(\d{3})(\d{2})(\d{2})$", date_str)
    if match:
        roc_year = int(match.group(1))
        return f"{roc_year + 1911}-{match.group(2)}-{match.group(3)}"

    return date_str


def _normalize_japanese_era_date(date_str: str) -> str:
    """Convert Japanese era date to ISO format if needed.

    EDINET typically uses ISO dates, but some older filings may use era dates.
    Reiwa era: R1 = 2019, R2 = 2020, etc.
    """
    if not date_str or not isinstance(date_str, str):
        return date_str

    # Format: R5.12.31 (Reiwa 5)
    match = re.match(r"R(\d+)\.(\d{1,2})\.(\d{1,2})", date_str)
    if match:
        year = int(match.group(1)) + 2018
        return f"{year}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

    return date_str


# Market-specific date normalizers
_DATE_NORMALIZERS: dict[str, Any] = {
    "tw_mops": _normalize_roc_date,
    "jp_jquants": _normalize_japanese_era_date,
}


# ---------------------------------------------------------------------------
# Numeric normalization
# ---------------------------------------------------------------------------

def _normalize_numeric(value: Any) -> float | None:
    """Convert various numeric representations to float.

    Handles: commas (1,234,567), parentheses for negatives ((500)),
    local decimal separators (1.234,56 in EU), percentage signs, etc.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    s = value.strip()
    if not s or s in ("-", "--", "N/A", "n/a", ""):
        return None

    # Handle parentheses as negative: (500) -> -500
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        is_negative = True
    elif s.startswith("-"):
        s = s[1:]
        is_negative = True

    # Remove percentage signs
    s = s.rstrip("%")

    # Remove currency symbols
    s = re.sub(r"[R$€£¥₩NT$CL$]", "", s).strip()

    # Handle EU-style decimals: 1.234.567,89 -> 1234567.89
    if "," in s and "." in s:
        if s.rindex(",") > s.rindex("."):
            # EU format: 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:
            # US format: 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        # Could be EU decimal (1234,56) or US thousands (1,234)
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Likely EU decimal
            s = s.replace(",", ".")
        else:
            # Likely US thousands
            s = s.replace(",", "")

    try:
        result = float(s)
        return -result if is_negative else result
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Currency metadata
# ---------------------------------------------------------------------------

_MARKET_CURRENCIES: dict[str, str] = {
    # Phase 1 -- core 10 markets
    "us_sec_edgar": "USD",
    "uk_companies_house": "GBP",
    "eu_esef": "EUR",
    "fr_esef": "EUR",
    "de_esef": "EUR",
    "jp_jquants": "JPY",
    "kr_dart": "KRW",
    "tw_mops": "TWD",
    "br_cvm": "BRL",
    "cl_cmf": "CLP",
    # Phase 2 -- 15 new markets
    "ca_sedar": "CAD",
    "au_asx": "AUD",
    "in_bse": "INR",
    "cn_sse": "CNY",
    "hk_hkex": "HKD",
    "sg_sgx": "SGD",
    "mx_bmv": "MXN",
    "za_jse": "ZAR",
    "ch_six": "CHF",
    "nl_esef": "EUR",
    "es_esef": "EUR",
    "it_esef": "EUR",
    "se_esef": "SEK",
    "sa_tadawul": "SAR",
    "ae_dfm": "AED",
}


# ---------------------------------------------------------------------------
# Public API: translate financial statements
# ---------------------------------------------------------------------------

def translate_financials(
    df: pd.DataFrame,
    market_id: str,
    statement_type: str = "",
) -> pd.DataFrame:
    """Translate a raw financial DataFrame to canonical format.

    Performs:
    1. Concept name mapping (region-specific -> canonical)
    2. Date normalization (ROC, Japanese era, etc.)
    3. Numeric value normalization
    4. Adds source metadata columns

    Parameters
    ----------
    df:
        Raw DataFrame from a PIT client. Expected columns vary by region
        but typically include: concept, value, filing_date, report_date.
    market_id:
        PIT market identifier for selecting the right mappings.
    statement_type:
        Optional hint: "income", "balance", "cashflow".

    Returns
    -------
    pd.DataFrame with canonical column names and normalized values.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    # 1. Map concept names to canonical
    concept_map = _MARKET_CONCEPT_MAPS.get(market_id, {})
    if concept_map and "concept" in result.columns:
        result["canonical_name"] = result["concept"].map(
            lambda c: _map_concept(c, concept_map)
        )
        # For CVM: also try account_code mapping
        if market_id == "br_cvm" and "account_code" in result.columns:
            result["canonical_name"] = result.apply(
                lambda row: (
                    row["canonical_name"]
                    if pd.notna(row.get("canonical_name")) and row["canonical_name"]
                    else _CVM_ACCOUNT_MAP.get(str(row.get("account_code", "")), "")
                ),
                axis=1,
            )
    elif "concept" in result.columns:
        # No mapping available; keep original concept as canonical_name
        result["canonical_name"] = result["concept"]

    # 2. Normalize dates
    date_normalizer = _DATE_NORMALIZERS.get(market_id)
    for col in ("filing_date", "report_date", "period_start"):
        if col in result.columns:
            if date_normalizer:
                result[col] = result[col].apply(
                    lambda x: date_normalizer(str(x)) if pd.notna(x) else x
                )
            result[col] = pd.to_datetime(result[col], errors="coerce")

    # 3. Normalize numeric values
    if "value" in result.columns:
        result["value"] = result["value"].apply(_normalize_numeric)

    # 4. Add source metadata
    result["market_id"] = market_id
    result["currency"] = _MARKET_CURRENCIES.get(market_id, "")
    if statement_type:
        result["statement_type"] = statement_type

    # 5. Drop rows where canonical_name is empty (unmapped concepts)
    if "canonical_name" in result.columns:
        result = result[result["canonical_name"].astype(str).str.len() > 0]

    return result


def translate_profile(
    profile: dict[str, Any],
    market_id: str,
) -> dict[str, Any]:
    """Translate a raw profile dict to canonical field names.

    Ensures all canonical profile fields are present (with empty defaults)
    and normalizes field names from region-specific variants.
    """
    if not profile:
        return {}

    result: dict[str, Any] = {}

    # Direct mappings for common aliases
    _PROFILE_ALIASES: dict[str, str] = {
        "company_name": "name",
        "denom_social": "name",
        "razonSocial": "name",
        "filerName": "name",
        "entity_name": "name",
        "symbol": "ticker",
        "stock_code": "ticker",
        "secCode": "ticker",
        "公司代號": "ticker",
        "公司名稱": "name",
        "country_code": "country",
        "market_sector": "sector",
        "icb_sector": "sector",
        "icb_industry": "industry",
        "SETOR_ATIVIDADE": "sector",
        "clasificacion": "sector",
        "actividad": "industry",
        "category33": "sector",
    }

    # Map all fields through aliases
    for key, value in profile.items():
        canonical_key = _PROFILE_ALIASES.get(key, key)
        if canonical_key in CANONICAL_PROFILE:
            if value and not result.get(canonical_key):
                result[canonical_key] = value
        else:
            # Preserve non-canonical fields as-is
            result[key] = value

    # Ensure all canonical profile fields exist
    for field in CANONICAL_PROFILE:
        result.setdefault(field, "")

    # Set currency from market if not present
    if not result.get("currency"):
        result["currency"] = _MARKET_CURRENCIES.get(market_id, "")

    result["market_id"] = market_id

    return result


# ---------------------------------------------------------------------------
# Pivot to wide format (one row per period, columns = canonical fields)
# ---------------------------------------------------------------------------

def pivot_to_canonical_wide(
    df: pd.DataFrame,
    date_col: str = "report_date",
) -> pd.DataFrame:
    """Pivot a long-format canonical DataFrame to wide format.

    Input:  long format with columns [canonical_name, value, report_date, ...]
    Output: wide format with one row per report_date and canonical names as columns.

    This is the format expected by downstream processing modules
    (financial health, ratio computation, forecasting, etc.).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if "canonical_name" not in df.columns or "value" not in df.columns:
        return df

    if date_col not in df.columns:
        return df

    # Keep only rows with actual values
    valid = df[df["value"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    # For duplicates within same period+concept, keep the latest filing
    if "filing_date" in valid.columns:
        valid = valid.sort_values("filing_date", ascending=False)
        valid = valid.drop_duplicates(
            subset=[date_col, "canonical_name"],
            keep="first",
        )

    # Pivot
    try:
        wide = valid.pivot_table(
            index=date_col,
            columns="canonical_name",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide = wide.sort_values(date_col, ascending=False)
        return wide
    except Exception as exc:
        logger.warning("Pivot to wide format failed: %s", exc)
        return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Alias map: old / variant field names produced by clients -> canonical names
# that cache_builder.STATEMENT_FIELDS expects.  Clients are not changed;
# this bridge ensures their output flows into the app correctly.
_FIELD_ALIASES: dict[str, str] = {
    "sga_expense": "sga_expenses",
    "research_and_development": "rd_expenses",
    "share_buyback": "stock_buybacks",
    "buybacks": "stock_buybacks",
    "operating_cashflow": "operating_cash_flow",
    "investing_cashflow": "investing_cf",
    "financing_cashflow": "financing_cf",
    "pretax_income": "ebit",
}


def _map_concept(concept: str, concept_map: dict[str, str]) -> str:
    """Map a raw concept name to its canonical equivalent.

    Tries exact match first, then prefix-stripped match, then alias
    resolution, then checks if the concept is already a canonical name.
    """
    if not concept:
        return ""

    # Exact match in the market-specific concept map
    if concept in concept_map:
        result = concept_map[concept]
        # Apply alias resolution on the mapped result too
        return _FIELD_ALIASES.get(result, result)

    # Strip namespace prefix and retry (e.g. "ifrs-full:Revenue" -> "Revenue")
    if ":" in concept:
        short = concept.split(":")[-1]
        for key, canonical in concept_map.items():
            if key.endswith(short):
                return _FIELD_ALIASES.get(canonical, canonical)

    # Check alias map (handles client output that uses old/variant names)
    concept_lower = concept.lower().replace(" ", "_")
    if concept_lower in _FIELD_ALIASES:
        return _FIELD_ALIASES[concept_lower]

    # Try case-insensitive match on the concept value itself
    # (handles cases where the concept IS already the canonical name)
    all_canonical = (
        CANONICAL_INCOME | CANONICAL_BALANCE | CANONICAL_CASHFLOW
    )
    if concept_lower in all_canonical:
        return concept_lower

    return ""


def get_concept_map(market_id: str) -> dict[str, str]:
    """Return the concept mapping dict for a given market.

    Useful for clients that want to do their own mapping.
    """
    return dict(_MARKET_CONCEPT_MAPS.get(market_id, {}))
