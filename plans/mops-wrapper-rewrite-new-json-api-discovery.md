# MOPS Wrapper Rewrite Plan

## Discovery Summary

MOPS (Taiwan's Market Observation Post System) was completely redesigned as a Vue.js SPA in 2025/2026. The old form POST endpoints (`/mops/web/ajax_t163sb04`) are now WAF-blocked from outside Taiwan. 

### New API discovered:
- **Base URL**: `https://mops.twse.com.tw/mops/api/`
- **Method**: JSON POST with `Content-Type: application/json`
- **Requires**: `curl_cffi` with Chrome TLS impersonation (plain `requests` gets blocked)
- **NOT geo-blocked**: Works globally

### Working endpoints:
| Endpoint | Statement | Rows (TSMC) |
|----------|-----------|-------------|
| `t164sb04` | Consolidated income statement | 46 rows |
| `t164sb03` | Consolidated balance sheet | 68 rows |
| `t164sb05` | Consolidated cash flow | 81 rows |

### API contract:
```json
POST /mops/api/t164sb04
{
  "companyId": "2330",
  "dataType": "1",
  "season": "4",
  "year": "113",
  "subsidiaryCompanyId": ""
}
```

### Response format:
```json
{
  "code": 200,
  "message": "查詢成功",
  "result": {
    "reportType": "合併",
    "year": "114",
    "season": "4",
    "companyAbbreviation": "台積電",
    "titles": [...],
    "reportList": [
      ["營業收入合計", "3,809,054,272", "100.00", "2,894,307,699", "100.00"],
      ...
    ],
    "urlList": [...]
  }
}
```

### Bridge endpoint:
- `redirectToOld` returns a URL to `mopsov.twse.com.tw` which serves the old HTML tables for quarterly data

### Key differences from old wrapper:
| Old | New |
|-----|-----|
| `cached_post()` (plain requests) | `curl_cffi` Session with Chrome impersonation |
| Form POST to `/mops/web/ajax_t163sb04` | JSON POST to `/mops/api/t164sb04` |
| `co_id`, `year`, `season` params | `companyId`, `dataType`, `season`, `year` params |
| HTML table response (pd.read_html) | Structured JSON with `reportList` arrays |
| WAF-blocked from outside Taiwan | Works globally |

## Implementation

1. Rewrite `_fetch_mops_financials()` to use new JSON API
2. Add Chinese -> canonical field name mapping
3. Use `curl_cffi` session instead of `cached_post()`
4. Parse `reportList` arrays into DataFrames with proper canonical names
5. Maintain backward compatibility with PITClient protocol
