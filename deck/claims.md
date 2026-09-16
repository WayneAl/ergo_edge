# Claims ledger — every number that appears on a slide

Rule: no number on a slide without a row here. 🟢 official/public price or statistic · 🟡 press, paper or
market research (cite year) · 🔴 team assumption — the slide shows a visible "team estimate" mark.
`ref` is the index in the References slide (18). Notes may be in Chinese; slides are English only.

| # | Claim on slide | Value | Grade | Source | ref | Slide |
|---|---|---|---|---|---|---|
| 1 | UGen300 compute | 40 TOPS INT4 / 20 TOPS INT8, 8 GB LPDDR4, 2.5 W typical, USB 3.1 Gen2 10 Gbps, 150 g fanless | 🟢 | ASUS UGen300 techspec page | R1 | 10 |
| 2 | UGen300 price | NT$6,890 (ASUS TW estore); US$299.99 | 🟢 | ASUS product page; cnx-software 2026-08-03 | R1, R2 | 10, 14 |
| 3 | Ready-made pose model | `yolov8m_pose` (COCO-17) compiled `.hef` for Hailo-10H | 🟢 | Hailo Model Zoo v5.4.0 hailo10h | R3 | 6, 10 |
| 4 | Hand keypoints | `hand_landmark_lite` (21 pts) `.hef` for Hailo-10H | 🟢 | Hailo Model Zoo | R3 | 6, 10 |
| 5 | Edge LLM for reports | Qwen2.5-1.5B-Instruct on Hailo-10H: 7.35 tok/s, TTFT 0.37 s, 2048 ctx | 🟢 | Hailo Model Zoo GenAI v5.3.0 MODELS.rst | R4 | 7, 9 |
| 6 | Independent power figure | Pi 5 + Hailo-10H LLM inference measured at 1.87 W | 🟡 | arXiv 2603.23640 (2026) | R5 | 10 |
| 7 | Major occupational accident deaths, Taiwan 2025 | 251 (287 in 2024); construction 105 = 42 % | 🟢 | MOL / OSHA Taiwan, ROC year 114 statistics | R6 | 3 |
| 8 | Employer duty | Occupational Safety and Health Act (TW) requires hazard identification and control, incl. ergonomic hazards (Art. 6 ¶2) | 🟢 | OSH Act, Taiwan | R7 | 3 |
| 9 | Insurance lever | Occupational accident insurance experience rating: lower accident rate → lower premium | 🟡 | Occupational Accident Insurance and Protection Act (TW), experience-rating rules | R8 | 3 |
| 10 | Minimum wage 2026 | NT$29,500 / month, NT$196 / hour, effective 2026-01-01 | 🟢 | MOL announcement 2025-10-21 | R6 | 14 (appendix) |
| 11 | OSH manager loaded hourly cost | NT$258.6 / h (NT$45,000 / month, 174 h) | 🔴 | team assumption | — | 14 |
| 12 | Manual assessment time | 1 h / day observation + 8 h / month reporting = 38 h / month | 🔴 | team assumption from typical EHS practice | — | 2, 14 |
| 13 | Labour saving per station | NT$117,922 / year (= 38 h × 12 × 258.6) | 🔴 | computed from 11, 12 | — | 14 |
| 14 | Indirect cost of one WMSD incident | NT$100k–800k (lost time, replacement, claims, management) | 🔴 | team assumption | — | 14 |
| 15 | CAPEX per station (existing PC) | NT$58,090 = 6,890 + 1,200 webcam + 50,000 deployment service | 🔴 | 2 + team pricing | — | 14 |
| 16 | OPEX per station | NT$1,300 / year (power + maintenance) | 🔴 | 14.5 W × 8,760 h × NT$3–4/kWh ≈ NT$381–508 + 10 % HW | — | 14 |
| 17 | ROI scenarios | Conservative 75,858 / Base 177,922 / Optimistic 384,589 NT$ per year → payback 9.3 / 3.9 / 1.8 months; 3-yr ROI 285 / 812 / 1,879 % | 🔴 | computed: payback = CAPEX ÷ (benefit − OPEX) × 12 | — | 14 |
| 18 | Sensitivity | benefit at 50 % → payback 8.0 months | 🔴 | computed | — | 14 |
| 19 | AI in manufacturing: efficiency ≠ savings | 100 manufacturing executives: 0 report significant revenue growth or cost savings, 64 % report efficiency gains | 🟡 | Grant Thornton 2026 AI Impact Survey (via LINE Today) | R9 | 14 |
| 20 | REBA method | Hignett & McAtamney 2000; score 1–15; bands 1 / 2–3 / 4–7 / 8–10 / 11–15 | 🟢 | Applied Ergonomics 31(2):201–205 | R10 | 6, appendix |
| 21 | RULA method | McAtamney & Corlett 1993; score 1–7 | 🟢 | Applied Ergonomics 24(2):91–99 | R11 | 6, appendix |
| 22 | Pose-based ergonomics is an active research line | 3D-HPE industrial risk framework (2022); HPE for WMSD posture detection (2025); real-time 2D framework (IJWHM); EMG-validated REBA/RULA adjustment factors (2026) | 🟡 | R12–R15 | R12–R15 | 6, 15 |
| 23 | Dev-laptop pose throughput | yolov8m-pose ≈ 48 FPS at 1080p on Apple M4 Pro (MPS), measured in golf_coach | 🟢 | own measurement 2026-09-16 | — | 12 |
| 24 | Hailo-10H pose throughput | to be measured on hardware in Stage II — no number on the slide until then | — | — | — | 12 (TODO) |
| 25 | Validation targets | ≥ 25 FPS @ 1080p per stream; Cohen's κ ≥ 0.7 vs two certified assessors; 7 × 24 h soak with drop and false-alert counts | 🔴 | team targets | — | 13 |
| 26 | Competitors | CerebrumEdge ErgoEdge, TuMeke, VelocityEHS Industrial Ergonomics: smartphone/cloud video assessment; Soter Analytics: wearables | 🟡 | vendor sites (verify URLs before final) | R16–R19 | 15 |
| 27 | ASUS footprint | ASUS keeps some own sites; most assembly at ODMs such as Pegatron (spun off 2008) | 🟡 | Wikipedia (Pegatron); aggregator pages | R20 | 16 (phrased as an ask) |
| 28 | Stage II hardware | finalist teams receive one UGen300, not returned | 🟢 | contest rules §3 | R21 | 16 |
| 29 | Unobserved share of the shift | "the 95 % of the shift nobody watches" | 🔴 | hackathon-topics.md §1.2 wording（沒人看到的那 95% 時間）; illustrative, not measured. Only consistent with claim 12 if the 1 h/day of observation is spread over 2–3+ stations | — | 2 |
| 30 | WMSDs lead occupational disease claims | "WMSDs are the largest category of occupational disease claims" (no number on the slide) | verify | no source in the repo. Find MOL labour-insurance occupational-disease statistics by category (or an EU-OSHA/BLS equivalent and say where), or drop the sentence before final | — | 2 |
| 31 | Illustrative event | Station 3 / "S3", 2026-10-02 14:32:07, REBA 9 High, trunk flexion 64°, upper arm 95°, 38 s | example | sample data from the deck brief, not a measurement; labelled "Example event, illustrative values" on slides 9 and 11 | — | 5, 9, 11 |
| 32 | Target segment | manufacturers with ≥ 200 staff | 🔴 | team segment choice, hackathon-topics.md §1.1（200 人以上工廠） | — | 4 |
| 33 | Weekly report generation time | 30–60 s per report | 🔴 | hackathon-topics.md §1.4: a few hundred tokens at 7.35 tok/s (claim 5) | — | 7 |
| 34 | Deployment durations | site survey 1 day; install 1–2 days; calibration 3–5 days; parallel run with manual scoring 2 weeks; training 1 day | 🔴 | hackathon-topics.md §1.7 | — | 8, 13 |
| 35 | Dev-laptop measurement condition | yolov8m-pose ≈ 48 FPS at 640 px model input (imgsz), Apple M4 Pro, MPS | 🟢 | golf_coach README.md backend table: "≈48 FPS on an Apple M4 Pro (MPS, 640 px imgsz)". Claim 23 says "at 1080p" but the source records a 640 px model input and no 1080p condition; slide 12 uses the source wording | — | 12 |
| 36 | Hailo runtime version targeted | pyHailoRT / HailoRT 5.3 | 🟢 | golf_coach README.md (HailoBackend written against the pyHailoRT 5.3 API shape, UNVERIFIED on hardware); ASUS e-Manual firmware 5.3.0 (hackathon-topics.md §0.2) | — | 12 |
| 37 | κ study sample size | ≥ 200 sampled postures | 🔴 | team target (deck brief) | — | 13 |
| 38 | ROI scenario inputs | Conservative: OSH hours NT$55,858 (= 18 h/month × 12 × NT$258.6, "about half") + 1 incident per 5 years × NT$100,000; Base: NT$117,922 (38 h/month) + 1 per 5 years × NT$300,000; Optimistic: NT$117,922 + 1 per 3 years × NT$800,000. Claim 18's 8.0 months is the base case at 50 % (conservative at 50 % would be 19.0 months) | 🔴 | hackathon-topics.md §1.6; payback, 3-yr ROI and sensitivity arithmetic re-checked 2026-09-16（保守情境 55,858 = 18 h，不是 38 h 的一半 19 h） | — | 14, A2 |
| 39 | REBA segment thresholds and activity score | trunk upright / 0–20° / 20–60° / > 60° → 1–4, +1 twist or side-bend; neck 0–20° → 1, > 20° or extension → 2, +1 twist or side-bend; legs bilateral 1 / unilateral or unstable 2, +1 knee 30–60°, +2 > 60°; upper arm 20° ext–20° flex 1, 20–45° 2, 45–90° 3, > 90° 4, +1 raised, +1 abducted, −1 supported; lower arm 60–100° → 1 else 2; wrist 0–15° → 1, > 15° → 2, +1 deviation or twist; activity +1 static > 1 min, +1 repetition > 4/min, +1 rapid large change | 🟢 | Hignett & McAtamney 2000 | R10 | 6, A1 |
| 40 | Stage II plan | 12 weeks from finalist notice: weeks 1–2 bring-up; 3–6 REBA/RULA engine, overlay, events; 7–9 dashboard, weekly LLM report, installer; 10–12 κ study and 7 × 24 h soak | 🔴 | team plan (deck brief; hackathon-topics.md §1.10 M1–M3) | — | 16 |
| 41 | Contest calendar | finalists announced 2026-10-26; finals 2026-12-19, which is week 8 of the 12-week plan | 🟢 | contest site, as recorded in the hackathon-topics.md header | R21 | 16 |
| 42 | R16 URL | cerebrumedge.com (CerebrumEdge ErgoEdge) | verify | root domain given in the deck brief; confirm it resolves to the named product before final | R16 | 15, 18 |
| 43 | R17 URL | tumeke.io (TuMeke) | verify | root domain given in the deck brief; confirm before final | R17 | 15, 18 |
| 44 | R18 URL | velocityehs.com (VelocityEHS Industrial Ergonomics) | verify | root domain given in the deck brief; confirm before final | R18 | 15, 18 |
| 45 | R19 URL | soteranalytics.com (Soter Analytics) | verify | root domain given in the deck brief; confirm before final | R19 | 15, 18 |
| 46 | Competitor table cells | Cloud video: coverage "Sampled clips", video leaves site "Yes", worn device "No", cost grows with "Clips filmed", explains by angle "Yes, per clip", offline "No". Wearables: "Continuous, while worn", "No video", "Yes", "Workers", "Partial", "Partial" | verify | derived from the category labels in claim 26, NOT read from the vendor sites. The slide footnote says "vendor claims, from public sites, 2026" — check every cell against the four sites before final | R16–R19 | 15 |
| 47 | Reference URLs missing | R6 (MOL accident statistics, minimum-wage announcement), R7 (OSH Act), R8 (Occupational Accident Insurance and Protection Act), R20 (Pegatron) have no URL in hackathon-topics.md | verify | add URLs (e.g. law.moj.gov.tw pages) before final | R6–R8, R20 | 18 |
