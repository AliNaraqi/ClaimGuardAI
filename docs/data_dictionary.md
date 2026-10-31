# Data Dictionary — `fraud_oracle.csv`

**Source:** Kaggle — [Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
**Rows × columns:** 15,420 × 33 (verified after download, 2026-10-24)
**Target:** `FraudFound_P` — binary flag as provided by the dataset author; **not** a verified legal fraud determination
**Scope:** claims from 1994–1996 only (see `Year`)

Produced from `notebooks/01_eda.ipynb`. All "available at scoring time" assessments are judgment calls documented for the leakage audit (Week 1 exit gate) — revisit before deployment framing.

| Field | Type | Definition | Permitted / observed range | Missingness | Scoring-time available | Notes |
|---|---|---|---|---|---|---|
| `Month` | categorical | Month of accident | 12 month abbreviations | none | Yes | |
| `WeekOfMonth` | int | Week-of-month of accident | 1–5 | none | Yes | |
| `DayOfWeek` | categorical | Day of week of accident | Mon–Sun | none | Yes | |
| `Make` | categorical | Vehicle manufacturer | 19 makes | none | Yes | |
| `AccidentArea` | categorical | Urban / Rural | Urban, Rural | none | Yes | |
| `DayOfWeekClaimed` | categorical | Day of week claim was filed | Mon–Sun, plus 1 sentinel `'0'` row | 1 sentinel row | Yes | treat `'0'` as missing, not a day |
| `MonthClaimed` | categorical | Month claim was filed | 12 months, plus 1 sentinel `'0'` row | 1 sentinel row | Yes | same row as above |
| `WeekOfMonthClaimed` | int | Week-of-month claim was filed | 1–5 | none | Yes | |
| `Sex` | categorical | Policyholder sex | Male, Female | none | Yes | protected attribute — see Responsible AI notes |
| `MaritalStatus` | categorical | Policyholder marital status | Single, Married, Widow, Divorced | none | Yes | |
| `Age` | int | Policyholder age | 0–80 | **320 rows = 0** (sentinel, see below) | Yes | `Age==0` always co-occurs with `AgeOfPolicyHolder=='16 to 17'` — encoding quirk, not a literal age; flag or impute for modeling |
| `Fault` | categorical | Who was at fault | Policy Holder, Third Party | none | Caution | fault determination may not be fully finalized at first scoring — document as an early-but-not-day-zero signal |
| `PolicyType` | categorical | Combined vehicle category + coverage | 9 combinations | none | Yes | redundant with `VehicleCategory` + `BasePolicy` |
| `VehicleCategory` | categorical | Sport / Utility / Sedan | 3 values | none | Yes | |
| `VehiclePrice` | categorical | Price bucket | 6 buckets | none | Yes | |
| `FraudFound_P` | int (target) | Dataset-provided fraud flag | 0 / 1 | none | — | **target; never a feature.** Prevalence: 5.99% positive |
| `PolicyNumber` | int | Row identifier | unique per row (15,420 unique) | none | N/A | **exclude from features** — pure identifier |
| `RepNumber` | int | Claims rep identifier | small set of IDs | none | Exclude | not a legitimate predictor of claim risk |
| `Deductible` | int | Policy deductible amount | numeric | none | Yes | |
| `DriverRating` | int | Driver rating score | 1–4 | none | Yes | |
| `Days_Policy_Accident` | categorical | Days between policy start and accident | bucketed ranges incl. `none` | none | Yes | |
| `Days_Policy_Claim` | categorical | Days between policy start and claim | bucketed ranges incl. `none` | none | Yes | |
| `PastNumberOfClaims` | categorical | Prior claims count bucket | none, 1, 2 to 4, more than 4 | none | Yes | |
| `AgeOfVehicle` | categorical | Vehicle age bucket | new, 2–7 years, more than 7 | none | Yes | |
| `AgeOfPolicyHolder` | categorical | Policyholder age bucket | 16–17 … over 65 | none | Yes | preferred over raw `Age` given the sentinel issue |
| `PoliceReportFiled` | categorical | Whether police report filed | Yes / No | none | Yes, if captured at intake | |
| `WitnessPresent` | categorical | Whether a witness was present | Yes / No | none | Yes, if captured at intake | |
| `AgentType` | categorical | Internal / External agent | 2 values | none | Yes | |
| `NumberOfSuppliments` | categorical | Supplementary claim count bucket | none, 1 to 2, 3 to 5, more than 5 | none | Yes | |
| `AddressChange_Claim` | categorical | Time since last address change | no change, under 6 months … 4 to 8 years | none | Yes | recent address change is a common fraud-analytics heuristic feature |
| `NumberOfCars` | categorical | Cars on the policy | 1 vehicle … more than 8 | none | Yes | |
| `Year` | int | Accident year | 1994, 1995, 1996 | none | Yes | **dataset is historical; do not frame outputs as reflecting current claims** |
| `BasePolicy` | categorical | Liability / Collision / All Perils | 3 values | none | Yes | |

## Known data-quality issues (full detail in `notebooks/01_eda.ipynb`)

1. **Missingness is partly sentinel-encoded, not NaN.** `.isna().sum()` reports zero, which is misleading — `DayOfWeekClaimed`/`MonthClaimed` use `'0'` as a placeholder (1 row), and `Age` uses `0` as a placeholder (320 rows).
2. **`Age == 0` quirk:** all 320 zero-age rows map exactly to `AgeOfPolicyHolder == '16 to 17'`. Treat as unknown/minor-encoding, not a true age of zero.
3. **No full-row duplicates.**
4. **`PolicyNumber` is a 1:1 row identifier** (15,420 unique values for 15,420 rows) — must be excluded from model features to avoid identifier memorization.
5. **No post-outcome leakage fields identified** (e.g., no "investigation result" or "recovery amount" column exists). `Fault` is flagged for documentation caution only.

## Protected / sensitive attributes

`Sex`, `MaritalStatus`, and age fields are present. Per the project's Responsible AI guidance (see project plan §10), these require explicit fairness consideration before/if used in training — assess exclusion vs. documented inclusion with group-level fairness checks.
