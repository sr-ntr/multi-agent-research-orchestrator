# Engineering All-Hands Meeting Notes - Q3 2023

**Date**: September 14, 2023
**Facilitator**: Priya Nair, VP Engineering
**Attendees**: All engineering staff (approximately 620 engineers)

---

## 1. CloudSync Pro v4.0 Post-Launch Review

The v4.0 launch in Q1 2023 was declared a success by the product team. Key metrics after six months in production:

- **Customer adoption**: 78 percent of Business and Enterprise customers have upgraded from v3.x.
- **Support ticket volume**: Reduced by 22 percent compared to the v3.x launch period, attributed to improved in-product error messages.
- **Sync latency (p95)**: 180 milliseconds for files under 50 MB, ahead of the 200 ms target.

The AI Automation Engine feature has seen slower adoption than expected. Only 31 percent of eligible customers have enabled it. The product team believes the 90-day data requirement before suggestions appear is a friction point and is evaluating a reduction to 45 days.

## 2. Hiring Update

The engineering team grew from 510 to 620 engineers between end of 2022 and the end of Q3 2023. The majority of new hires (68 out of 110) are in the infrastructure and reliability engineering sub-teams. Open headcount for Q4 2023 is **15 positions**, primarily in the AI platform team.

## 3. Infrastructure Cost Review

Cloud infrastructure costs for H1 2023 totalled **14.2 million dollars**. The infrastructure team projects full-year 2023 costs of **29 million dollars**, which is 12 percent higher than the 2022 full-year cost of **25.9 million dollars**.

The increase is driven primarily by the additional compute required for the AI Automation Engine. The team is evaluating spot instances and reserved-capacity contracts to reduce costs by an estimated 15 to 20 percent in 2024.

## 4. Security Posture

The security team completed its annual penetration test in August 2023. No critical vulnerabilities were identified. Two medium-severity findings were remediated within 14 days of discovery. The next penetration test is scheduled for February 2024.

## 5. Technical Debt Initiative

The architecture team presented a proposal to retire the legacy synchronisation daemon (codenamed "OldRail"), which dates to 2017 and is responsible for approximately 18 percent of production incidents. The proposed timeline is a full replacement by Q2 2024, at an estimated engineering cost of **6,000 person-hours**.

## 6. Next All-Hands

The Q4 2023 engineering all-hands is scheduled for December 7, 2023.
