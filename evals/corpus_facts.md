# Corpus Facts Reference

This file lists every planted fact in the eval corpus with its source file.
Use this to derive and verify golden set reference answers.
DO NOT commit this file without verifying the facts against the actual documents.

---

## annual_report_2023.md

- Total revenue 2023: 312 million dollars
- Total revenue 2022: 274 million dollars
- YoY growth: ~13.9 percent (38M / 274M)
- Operating expenses 2023: 198 million dollars
- Operating profit 2023: 114 million dollars
- Net income 2023: 87 million dollars
- Tax rate 2023: 23.7 percent
- Total employees end of 2023: 4,200
- Total employees end of 2022: 3,800
- Product revenue 2023: 224.6 million (72% of 312M)
- CloudSync Pro product revenue 2023: 145 million dollars
- Services revenue 2023: 87.4 million (28%)
- North America revenue 2023: 187 million (60%)
- Europe revenue 2023: 93.6 million (30%)
- Asia-Pacific revenue 2023: 31.4 million (10%), grew 42% YoY
- R&D spend 2023: 47 million dollars (15% of revenue)
- R&D headcount 2023: 620 engineers
- GDPR/CCPA compliance cost increase 2023: 3.2 million dollars
- Revenue growth guidance 2024: 10 to 14 percent
- CloudSync Enterprise launch: Q2 2024

## annual_report_2022.md

- Total revenue 2022: 274 million dollars
- Operating expenses 2022: 181 million dollars
- Operating profit 2022: 93 million dollars
- Net income 2022: 71 million dollars
- Total employees end of 2022: 3,800
- Product revenue 2022: 197 million dollars (71.9%)
- CloudSync Pro product revenue 2022: 121 million dollars
- Services revenue 2022: 77 million dollars (28.1%)
- North America revenue 2022: 164.4 million (60%)
- Europe revenue 2022: 82 million (29.9%)
- Asia-Pacific revenue 2022: 22.1 million, grew 28% YoY (from 17.3M in 2021)
- R&D spend 2022: 38 million dollars (13.9% of revenue)
- R&D headcount 2022: 510 engineers
- Revenue growth guidance 2023: 12 to 16 percent
- Partner certification programme launched: 2022

## policy_v1.md

- Effective date: January 1, 2022
- Tier 2 label in v1: "Internal"
- Tier 2 retention in v1: 3 years
- Tier 3 retention in v1: 7 years
- Tier 4 retention in v1: 10 years
- Breach notification window (customers): 72 hours
- MFA required: Tier 3 and above

## policy_v2.md

- Effective date: January 1, 2024
- Tier 2 label in v2: "Business Use" (renamed from "Internal")
- Tier 2 retention in v2: 2 years (reduced from 3)
- Tier 3 retention in v2: 7 years (unchanged)
- Tier 4 retention in v2: 10 years (unchanged)
- Breach notification window (customers): 72 hours (unchanged)
- New in v2: internal incident report within 24 hours of detection
- New in v2: annual data-minimisation reviews; delete/downgrade within 90 days
- New in v2: MFA also required for Tier 2 accessed remotely

## product_spec.md

- Sync latency (small files, v4.0): 200 ms
- Sync latency (small files, v3.x): 500 ms
- Max file size Enterprise v4.0: 100 GB
- Max file size Enterprise v3.x: 25 GB
- API rate limit v4.0: 1,000 requests/minute
- API rate limit v3.x: 500 requests/minute
- Max RBAC roles v4.0: 500; v3.x: 100
- Professional plan price: 25 USD/user/month
- Business plan price: 45 USD/user/month
- AI Automation Engine requires: 90 days of usage data
- SSO via: SAML 2.0 and OAuth 2.0
- TLS version: 1.3
- Encryption at rest: AES-256

## meeting_notes_q3_2023.md

- Meeting date: September 14, 2023
- v4.0 customer adoption: 78% of Business and Enterprise customers
- Support ticket reduction vs v3.x launch: 22%
- Sync latency p95 actual: 180 ms (better than 200 ms target)
- AI Automation Engine adoption: 31% of eligible customers
- Evaluated reduction: 90 days -> 45 days for AI suggestions
- Engineering headcount grew 510 -> 620 (2022 to Q3 2023)
- New hires in infrastructure: 68 out of 110
- Open Q4 2023 headcount: 15 positions (AI platform)
- Cloud infra cost H1 2023: 14.2 million dollars
- Cloud infra projected full-year 2023: 29 million dollars
- Cloud infra 2022 full-year: 25.9 million dollars
- Tech debt: OldRail replacement estimated 6,000 person-hours by Q2 2024

## partner_programme.md

- Programme launched: 2022
- Number of certification tracks: 5
- Exam cost: 200 USD per attempt
- Certification validity: 2 years
- Partner review time: 10 business days
- Registered partners end 2023: 312
- Silver partners end 2023: 87
- Gold partners end 2023: 24
- Partner-sourced revenue 2023: ~48 million dollars (~15.4% of total)

---

## Multi-hop facts (require combining two documents)

1. R&D headcount increase 2022->2023: 510 (annual_report_2022) -> 620 (annual_report_2023) = +110 engineers
2. CloudSync Pro revenue growth 2022->2023: 121M -> 145M = +24M (+19.8%)
3. Asia-Pacific revenue growth 2022->2023: 22.1M -> 31.4M = +9.3M (+42%)
4. Policy change: Tier 2 retention reduced from 3 years (v1) to 2 years (v2)
5. Cloud infra cost increase rate: 29M (2023 projected) vs 25.9M (2022) = ~12% increase (mentioned in meeting_notes)
