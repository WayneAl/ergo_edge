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
