2026-03-07 14:32:13 | INFO    | operator1.main | ============================================================
2026-03-07 14:32:13 | INFO    | operator1.main | OPERATOR 1 -- Point-in-Time Financial Analysis
2026-03-07 14:32:13 | INFO    | operator1.main | ============================================================
2026-03-07 14:32:13 | INFO    | operator1.secrets_loader | Loaded environment from /home/manem/githubu-isu-meanu/.env
2026-03-07 14:32:13 | INFO    | operator1.secrets_loader | All 9 required API keys loaded successfully.
2026-03-07 14:32:13 | INFO    | operator1.main | 
2026-03-07 14:32:13 | INFO    | operator1.main | Step 1: Selecting data source and company...
2026-03-07 14:32:13 | INFO    | operator1.main | Market: Japan (Tokyo Stock Exchange (JPX)) -- J-Quants
2026-03-07 14:32:14 | INFO    | operator1.clients.jp_jquants_wrapper | J-Quants ClientV2 initialized successfully
2026-03-07 14:32:18 | INFO    | operator1.main | Company found: TOYOTA MOTOR CORPORATION (7203)
2026-03-07 14:32:18 | INFO    | operator1.main | Target: TOYOTA MOTOR CORPORATION () via J-Quants
2026-03-07 14:32:18 | INFO    | operator1.main | Macro source: e-Stat (Statistics Bureau of Japan) (Japan)
2026-03-07 14:32:18 | INFO    | operator1.main | 
2026-03-07 14:32:18 | INFO    | operator1.main | Step 2: Fetching company profile from J-Quants...
2026-03-07 14:32:21 | INFO    | operator1.main | Profile loaded: KYOKUYO CO.,LTD. (1301), sector=水産・農林業
2026-03-07 14:32:21 | INFO    | operator1.main | 
2026-03-07 14:32:21 | INFO    | operator1.main | Step 3: Fetching point-in-time financial data...
2026-03-07 14:32:21 | INFO    | operator1.main | Quotes: 0 rows
2026-03-07 14:39:06 | ERROR   | operator1.clients.jp_jquants_wrapper | J-Quants get_financials failed for : 400 Client Error: Bad Request for url: https://api.jquants.com/v2/fins/summary?date=20210308
2026-03-07 14:39:06 | INFO    | operator1.main | Cashflow: 0 rows
2026-03-07 14:39:06 | ERROR   | operator1.clients.jp_jquants_wrapper | J-Quants get_financials failed for : HTTPSConnectionPool(host='api.jquants.com', port=443): Max retries exceeded with url: /v2/fins/summary?date=20210308 (Caused by ResponseError('too many 429 error responses'))
2026-03-07 14:39:06 | INFO    | operator1.main | Income: 0 rows
2026-03-07 14:39:07 | ERROR   | operator1.clients.jp_jquants_wrapper | J-Quants get_financials failed for : HTTPSConnectionPool(host='api.jquants.com', port=443): Max retries exceeded with url: /v2/fins/summary?date=20210311 (Caused by ResponseError('too many 429 error responses'))
2026-03-07 14:39:07 | INFO    | operator1.main | Balance: 0 rows
2026-03-07 14:39:07 | INFO    | operator1.quality.data_reconciliation | Reconciliation complete: 0 statements, no issues
2026-03-07 14:39:07 | INFO    | operator1.main | Data reconciliation: clean
2026-03-07 14:39:07 | ERROR   | operator1.main | No data retrieved for  from J-Quants. Check the identifier and try again.
```

**Log saved to:** `cache/pipeline_run.md`
