# NEW research — ผล walk-forward (ขั้นที่ 3–9) — 2026-10-18

Branch `feat/new-research` · spec `prompts/new_research.md` · ใช้ `config.yaml` และ `common_eval/` ตาม commit d7e94e7 (เพิ่มเฉพาะ key ใต้ `new:`)
ช่วงข้อมูล: development + walk-forward ถึง 2025-06-30 เท่านั้น **ไม่ได้โหลดหรือดู holdout (≥ 2025-07-01)**
ตารางทั้งหมดอยู่ใน `new_walkforward_20261018.xlsx` กราฟใน `reports/charts/new_*`

## สรุปผู้บริหาร

1. **ยังไม่มี edge ทั้งฝั่ง UP:** ทุกรุ่นไม่ผ่านเกณฑ์หัวข้อ 11 ครบ (0 จาก 16 รุ่น × 2 exit)
2. **NEW_XGB (UP) แย่กว่าสุ่ม:** expectancy −8.3%/เทรด (186 เทรด), percentile 10.5 เทียบ RANDOM ที่จับเวลาเดียวกัน — ซ้ำรอย LEGACY: เลือกเหรียญได้แย่กว่าสุ่ม
3. **ตัวที่ดีที่สุดคือ baseline ไม่ใช่ XGBoost:** NEW_LOGISTIC (native exit) +4.9%/เทรด, CAGR +6.5%, ผ่าน RANDOM P95 ทั้งสองแบบ, ยังกำไรเมื่อต้นทุน 2 เท่า (+4.7%) และตัด 3 เทรดดีที่สุด (+2.9%)
4. แต่ NEW_LOGISTIC **ไม่ผ่าน:** CI 95% ของผลต่างกับ LEGACY_FIXED คร่อม 0 [−5.2%, +25.5%] และ Sharpe 0.37 / Calmar 0.11 ต่ำกว่า BTC_HOLD (0.71 / 0.40). กำไรมาจากปี 2023 ปีเดียว (+127%) ส่วน 12 เดือนหลัง development (2024-07 → 2025-06) ขาดทุน −7.0%/เทรด
5. **โมเดล UP ไม่มีทักษะ:** XGB PR-AUC 0.148 ต่ำกว่า base rate 0.155, ROC 0.47. ตอนที่ใส่ macro เป็นฟีเจอร์ (ผิดกฎหัวข้อ 1, แก้แล้ว) PR-AUC ขึ้นเป็น 0.19 ซึ่งยืนยันว่า "ทักษะ" มาจากการจับเวลา ไม่ใช่การเลือกเหรียญ
6. **ฝั่ง DOWN มีสัญญาณจริงแต่อ่อน:** logistic PR-AUC 0.285 vs base 0.191, ROC 0.61 แต่ส่วนใหญ่เป็น timing (คะแนนเฉลี่ยรายวันอย่างเดียว ROC 0.63); ส่วนการเลือกเหรียญภายในวันเหลือ ROC 0.53–0.55. **backtest ฝั่ง DOWN ยังไม่รัน** เพราะต้องคิด funding ซึ่ง common_eval ยังไม่รองรับ — เสนอ diff รออนุมัติ
7. **pattern แบบ "เหรียญเก่า ห่าง ATH ฐานนาน" (คล้าย ZEC) ไม่มีพลังทำนาย:** UP rate 23.1% vs 19.9% ทั่วไป (lift 1.16), cluster 3 แบบ silhouette 0.17, similarity search top-20 lift 0.3–1.0
8. **event 2X ไม่ใช่เหตุการณ์หายาก:** 19% ของ coin-day ขึ้น 2 เท่าภายใน 90 วันโดยไม่ลงเกิน 25% และขึ้นกับสภาวะตลาดมาก (2020 39%, 2022 11%, 2025H1 2.7%)
9. ข้อแนะนำ: **ไม่ควรใช้ XGBoost**; ถ้าจะนำรุ่นใดไปทดสอบ holdout ให้ใช้ NEW_LOGISTIC (native) เป็นสมมติฐานเดียวที่บันทึกไว้ล่วงหน้า และคาดว่ามีโอกาสสูงที่จะไม่ผ่าน
10. ความมั่นใจในข้อสรุป "UP ไม่มี edge": สูง (หลายวิธีให้ผลตรงกัน). ความมั่นใจใน NEW_LOGISTIC: ต่ำ (ทดสอบหลายรุ่น, ขึ้นกับปีเดียว)

## หัวข้อ 2: ZEC — การยืนยันจากวันที่ และการเปิดเผยข้อมูลที่ผู้วิจัยเคยเห็น

- ข้อมูล ZEC ใน `data_new/` สิ้นสุด 2025-06-30 20:00. ช่วงที่ spec ระบุ (ก.ย.–พ.ย. 2025) อยู่หลัง 2025-07-01 ทั้งหมด จึงอยู่ใน holdout. **ยืนยันวันที่แน่นอนไม่ได้โดยไม่ดูราคา** จึงรายงานตามนี้และไม่ได้ดูราคา
- **ผมไม่ได้ "บอด" ต่อ ZEC รอบนี้ 100%:** README ของ LEGACY ระบุว่า preset `greedy` ซื้อ ZEC ที่ $52.83 (2025-09-14) ขาย $452.04 และ b7 เข้า $229.8; ในการ audit ขั้นที่ 1–2 ผมรัน backtest ช่วง 2021–2026 ก่อนล็อก holdout และเห็นว่า rotation ถือ ZEC +729%. การป้องกันที่ใช้: ไม่มีฟีเจอร์/กฎ/template ที่เลือกด้วยมือจาก ZEC; กฎ baseline สร้างอัตโนมัติจากตาราง separation; template = centroid ของ cluster จาก event ทั้งหมด; กลุ่ม "คล้าย ZEC" นิยามตาม spec หัวข้อ 5 เท่านั้น
- ข้อมูล ZEC ก่อน 2025-07-01 ใช้ตามปกติ: ZEC มี UP event 11 ครั้งในข้อมูล (ล่าสุด 2024-07-05 และ 2024-10-03)

## หัวข้อ 3: ข้อมูล (B1)

| ข้อ | สิ่งที่ทำ | หมายเหตุ |
|---|---|---|
| 1 | ดาวน์โหลด 4H ใหม่ทุกเหรียญ (703) พร้อม Quote_Volume, Trades, Taker_Buy_Volume, Taker_Buy_Quote_Volume — futures เป็นหลัก ต่อด้วย spot ก่อนมี futures | **ทำเป็นโมดูลใหม่ `new_research/download.py`** แทนการแก้ `core/data_downloader.py` (ตัวนั้นเป็น REST รายเหรียญของ basket) เพื่อไม่ให้ LEGACY เปลี่ยน. 506 เหรียญมีข้อมูลก่อน cut; แถว ≥ 2025-07-01 เขียนแยกที่ `data_new_holdout/` (ไม่ถูกอ่าน) |
| 2 | Funding_Rate = NaN ก่อนมี futures + `has_futures` | settlement ดิบแยกที่ `data_funding/` (703 เหรียญ) สำหรับคิด funding |
| 3 | ตัด bar volume = 0 | + ตัด wick ผิดปกติ (> 3 เท่า / < 1/3 ของตัวแท่ง) 49 แท่งใน 49 เหรียญ ส่วนใหญ่เป็นแท่งแรกของการลิสต์ spot |
| 4 | `^IRX` → US3M, DGS2 (lag 1 วัน) → US2Y | อยู่ใน snapshot; ไม่ได้ใช้ในโมเดล |
| 5 | BTC trend จาก BTCUSDT 4H | `btc_trend` = close/EMA1200 − 1 |
| 6 | FearGreed D+1 | snapshot; ไม่ได้ใช้ในโมเดล |
| 7 | snapshot macro/FRED/FG ที่ `data/snapshots/*_20261018.csv` | ดึงถึง 2025-06-30 เท่านั้น |
| 8 | `data/coin_metadata.csv` | sector จาก CoinGecko categories (ปัจจุบัน ไม่ใช่ point-in-time); 108/506 = other; delist ก่อน holdout 27 เหรียญ |
| 9 | universe เดียวกับ LEGACY_FIXED | backtest ใช้ `data_all` + MIN_QUOTE_VOL_30D ผ่าน common_eval |

**ที่เบี่ยงจาก spec:** HY_SPREAD ไม่ได้ใช้ เพราะ FRED ให้ข้อมูลตั้งแต่ 2023-10-17 เท่านั้น — ถ้าใส่ NaN + flag ก่อนหน้านั้น flag จะกลายเป็นตัวบอกช่วงเวลา (ซ้ำ H2/M4). regime จึงเหลือ 2 ตัว (BTC trend, net liquidity) และ**ใช้เป็นตัวกรองเท่านั้น ไม่เป็นฟีเจอร์** (หัวข้อ 1)

## หัวข้อ 4: event

- UP = high ≥ 2 × close[t] ภายใน 540 bar (90 วัน) ก่อน low < 0.75 × close[t]; DOWN = low ≤ 0.5 × close[t] ก่อน high > 1.25 × close[t]; แท่งเดียวกัน = นับว่าโดนทางเสียก่อน
- label ที่ window ยื่นเข้า holdout = NaN (ตัดทิ้ง); เหรียญที่ delist ก่อน cut ได้ผลตามจริง
- ตรวจ label ด้วยการคำนวณซ้ำแบบ brute force 40 แถวสุ่ม: ตรง 40/40
- bar-level base rate: UP 19.1%, DOWN 19.2% (1 sample/เหรียญ/วัน, 478,888 coin-day, 429 เหรียญ)
- event-level (1 run = 1 event, cooldown 30 วัน, จุดเริ่ม = swing low/high): **UP 1,909 (development 1,465), DOWN 2,292**

| ปี | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025H1 |
|---|---|---|---|---|---|---|---|
| UP event | 75 | 252 | 292 | 391 | 297 | 501 | 82 |
| DOWN event | 74 | 195 | 489 | 460 | 251 | 711 | 84 |

sector ที่มี UP event มากสุด: other 274, AI 187, layer-1 179, exchange 170, DeFi 166, gaming 148, smart-contract 134, meme 125

## หัวข้อ 5: profile ก่อน event (development ≤ 2024-06-30)

- **ข้อแตกต่างที่มีนัยสำคัญที่สุด (UP, Bonferroni p < 0.001):** ณ จุดเริ่ม event ราคาต่ำกว่า EMA300 มาก (มัธยฐาน −25% vs −5% ทั่วไป, AUC 0.15), EMA300 กำลังลง, relative strength 7/30 วันต่ำ, ห่างจาก high 180/540/1080 bar มากกว่า, sector ก็ลง
- **ข้อควรระวัง:** จุดเริ่ม event ถูกนิยามเป็น swing low (ราคาปิดต่ำสุดของ run) ความต่างเหล่านี้จึงเกิดจากนิยามเองส่วนหนึ่ง — มันบอกว่า event เริ่มที่ "จุดต่ำ" ซึ่งรู้ได้หลังเกิดเหตุ ไม่ได้บอกว่าจุดต่ำไหนจะเป็นจุดเริ่ม. DOWN ก็เช่นกัน (AUC 0.90 สำหรับ close/EMA300 เพราะจุดเริ่ม = swing high)
- ณ 30–180 วันก่อน event ฟีเจอร์ส่วนใหญ่อยู่ที่ percentile ~0.45–0.53 (ไม่ต่างจากเหรียญทั่วไป) — ไม่พบ "สัญญาณล่วงหน้า" ที่ชัด
- **กลุ่มคล้าย ZEC** (อายุ ≥ 2 ปี, ต่ำกว่า ATH ≥ 80%, ฐาน ≥ 90 วัน): 5.2% ของ coin-day, UP rate 23.1% vs 19.9% (lift 1.16), event ที่เริ่มในกลุ่มนี้ 80 จาก 1,465
- `base_duration_days` (นับวันที่กรอบ 90 วันกว้างไม่ถึง 100%) มัธยฐาน 0 เกือบทุก lag — นิยามนี้ใช้ไม่ได้กับ alt ที่ผันผวนสูง ควรนิยามใหม่ในรอบถัดไป
- **Cluster (k-means บน 15 ฟีเจอร์ winsorize 1/99%):** k = 3 (silhouette 0.17 — แยกกลุ่มได้อ่อน): 559/603 event อยู่ในสองกลุ่มหลัก (ผันผวนสูงใกล้จุดต่ำ vs เงียบฐานยาว) และกลุ่มเล็ก 59 event (ผันผวนสุดโต่ง). **coin-day ที่ใกล้ centroid ที่สุด 10% มี UP rate 0.84–1.06 เท่าของ base** — profile ไม่ช่วยทำนาย
- กฎ baseline ที่สร้างอัตโนมัติ: UP = close/EMA300 ≤ q75, EMA300 slope ≤ q75, breakout_180 ≤ q75, rs_30d ≤ q75 (ของ positive ในช่วง train ของแต่ละ fold); DOWN = close/EMA300, RSI14, rs_7d, rs_30d ≥ q25

## หัวข้อ 7: similarity search (walk-forward 2022-01 → 2025-06)

| วิธี | top 20 | top 100 | top 500 | top 2,000 | base |
|---|---|---|---|---|---|
| ระยะฟีเจอร์ถึง centroid | 5.0% | 12.0% | 10.2% | 12.4% | 15.5% |
| DTW ราคา + quote volume 90 วัน | 15.0% | 23.0% | 15.4% | 15.2% | 15.5% |
| รวม | 5.0% | 17.0% | 16.4% | 16.7% | 15.5% |

ไม่มีวิธีไหนดีกว่า base อย่างสม่ำเสมอ (DTW top-100 23% เป็นจุดเดียวและไม่คงที่). รายชื่อ top 20 อยู่ใน sheet `similar_matches`

## หัวข้อ 8: โมเดล (walk-forward 7 fold, purge + embargo 540 bar, early stopping บนช่วง validation ท้าย train)

| target | model | PR-AUC | base | ROC | ROC เฉพาะ timing* | ROC ภายในวัน** | prec@10/wk |
|---|---|---|---|---|---|---|---|
| UP | rule | 0.147 | 0.155 | 0.485 | 0.433 | 0.521 | 0.137 |
| UP | logistic | 0.147 | 0.155 | 0.491 | 0.478 | 0.488 | 0.148 |
| UP | **xgb** | **0.148** | 0.155 | **0.471** | 0.413 | 0.526 | 0.175 |
| DOWN | rule | 0.200 | 0.191 | 0.467 | 0.440 | 0.479 | 0.217 |
| DOWN | logistic | **0.285** | 0.191 | 0.613 | 0.631 | 0.535 | 0.250 |
| DOWN | xgb | 0.258 | 0.191 | 0.605 | 0.558 | 0.554 | 0.254 |

\* ใช้แค่ค่าเฉลี่ยคะแนนของวันนั้น (= การจับเวลา) \*\* คะแนนหักค่าเฉลี่ยของวัน (= การเลือกเหรียญ)

- ทุก fold มี event ใน test ≥ 86 เหรียญ (เกิน MIN_EVENTS_PER_FOLD = 10)
- **SHAP (UP xgb):** days_since_futures, log_qv_30d, rs_90d, pct_from_low_1y, range_width_90d — ตัวบนสุด 2 ตัวเป็นอายุตลาด futures และสภาพคล่อง ซึ่งสัมพันธ์กับรุ่นการลิสต์ (ช่วงเวลา) มากกว่าลักษณะเฉพาะของเหรียญ
- **SHAP (DOWN xgb):** days_since_ath, days_since_futures, log_qv_30d, ema1200_slope, coin_age, funding_30d — มีเหตุผลทางตลาด (เหรียญใหม่ที่เพิ่งทำ ATH และ funding สูงมักร่วง) แต่ปนกับรุ่นการลิสต์
- **ตอนแรกผมใส่ btc_trend และ net liquidity เป็นฟีเจอร์ ซึ่งผิดกฎหัวข้อ 1** — SHAP ตอนนั้นมี netliq/btc_trend อยู่อันดับ 1–2 และ PR-AUC ของ UP xgb สูงกว่า (0.19). แก้แล้วและผลทุกตัวในรายงานนี้เป็นหลังแก้
- ข้อสังเกตความเสถียร: การเลื่อน btc_trend ไป −1 (ไม่เปลี่ยนลำดับ) ทำให้ PR-AUC ของ UP xgb เปลี่ยนจาก 0.165 เป็น 0.193 — โมเดลไม่เสถียรระดับนี้ = ไม่มีสัญญาณที่มั่นคง

## หัวข้อ 9: backtest ฝั่ง UP ผ่าน common_eval (spot, ไม่มี leverage, 2022-01 → 2025-06)

กฎสัญญาณที่ตั้งก่อนดูผล: score ≥ quantile 99% ของคะแนนช่วง validation ของ fold นั้น; ตัวกรอง regime = BTC close > EMA1200. exit native ของ NEW = TP +100% / SL −25% / 90 วัน (ตรงกับ label); unified = trailing 35% + 540 bar

| รุ่น | exit | เทรด | win | avg W / L | expectancy | ตัด top-3 | CAGR | MaxDD | Sharpe | Calmar |
|---|---|---|---|---|---|---|---|---|---|---|
| NEW_BASELINE (กฎ) | native | 298 | — | — | −5.38% | −6.44% | −51.2% | −93.4% | −0.63 | −0.55 |
| NEW_BASELINE + regime | native | 199 | 30.7% | +46% / −24% | −2.40% | −3.95% | −31.0% | −85.8% | −0.30 | −0.36 |
| **NEW_XGB** | native | 186 | 17.7% | +62% / −23% | **−8.28%** | −10.0% | −45.7% | −91.9% | −0.79 | −0.50 |
| NEW_XGB | unified | 183 | 27.9% | +38% / −23% | −5.88% | −10.2% | −33.0% | −85.0% | −0.45 | −0.39 |
| NEW_XGB + regime | native | 163 | — | — | −10.68% | −12.7% | −47.5% | −92.8% | −0.97 | −0.51 |
| **NEW_LOGISTIC** | native | 146 | 37.0% | +50% / −21% | **+4.87%** | +2.89% | **+6.5%** | −61.9% | 0.37 | 0.11 |
| NEW_LOGISTIC | unified | 150 | 40.7% | +32% / −20% | +1.13% | −2.28% | −3.4% | −65.1% | 0.16 | −0.05 |
| NEW_LOGISTIC + regime | native | 130 | — | — | +3.72% | +1.47% | −2.5% | −71.1% | 0.15 | −0.04 |
| POST-HOC XGB top 2/วัน*** | native | 216 | — | — | −1.47% | −2.88% | −28.8% | −86.1% | −0.22 | −0.34 |
| LEGACY_FIXED long | native | 266 | 29.7% | — | −4.88% | −6.1% | −50.3% | −92.1% | −0.43 | −0.55 |
| BTC_HOLD | | | | | | | **+26.6%** | −66.9% | **0.71** | **0.40** |
| EQUAL_WEIGHT | | | | | | | −36.7% | −85.2% | −0.15 | −0.43 |

\*\*\* เพิ่มหลังเห็นว่ากฎหลักให้สัญญาณ XGB น้อย จึงถือเป็น post-hoc ไม่ใช้ตัดสิน

**ผลรายปี (native):** NEW_LOGISTIC 2022 −10.6%, **2023 +127.5%**, 2024 +23.9%, 2025H1 −50.5%; NEW_XGB −14.7%, −18.4%, −33.5%, −74.4%. ช่วงหลัง development (2024-07 → 2025-06) ทุกรุ่นติดลบ (NEW_LOGISTIC −7.0%/เทรด, CAGR −43%)

**ตัวกรอง regime (บทเรียนปี 2022):** ช่วยกฎ baseline (CAGR −51% → −31%) แต่ทำให้ XGB และ logistic แย่ลง; ปี 2022 BTC อยู่เหนือ EMA1200 เพียง 3% ของวัน ตัวกรองจึงปิดฝั่ง UP เกือบทั้งปีอย่างที่ตั้งใจ

### RANDOM 1,000 ครั้ง

| รุ่น | exit | RANDOM แบบจับเวลาเดียวกัน (เทรดใกล้เคียง) | RANDOM แบบความถี่เท่ากัน |
|---|---|---|---|
| NEW_XGB | native | percentile **10.5** (P50 −5.7%, P95 −2.2%) | 0.0 |
| NEW_XGB | unified | 50.1 | 4.6 |
| NEW_XGB + regime | native / unified | 10.6 / 46.0 | 0.0 / 0.1 |
| **NEW_LOGISTIC** | native | **99.6** (P50 +0.0%, P95 +3.1%) | **100.0** |
| NEW_LOGISTIC | unified | 85.9 | 96.8 |
| NEW_LOGISTIC + regime | native / unified | 89.2 / 74.4 | 99.9 / 92.1 |

RANDOM แบบความถี่เท่ากันให้จำนวนเทรดมากกว่ามาก (~300 vs 146–186) เพราะสัญญาณของโมเดลกระจุกในบางช่วง จึงใช้แบบจับเวลาเดียวกันเป็นตัวหลัก

### Bootstrap (month-block) ผลต่าง expectancy NEW − LEGACY_FIXED (long)

| รุ่น | exit | ผลต่าง | CI 95% |
|---|---|---|---|
| NEW_XGB | native | −3.4% | [−16.2%, +9.7%] |
| NEW_LOGISTIC | native | +9.8% | [−5.2%, +25.5%] |
| NEW_LOGISTIC | unified | +7.6% | [−3.6%, +19.4%] |

### ความทนทาน (NEW_XGB และ NEW_LOGISTIC, native exit)

| การทดสอบ | NEW_XGB expectancy / CAGR | NEW_LOGISTIC expectancy / CAGR |
|---|---|---|
| ฐาน | −8.3% / −45.7% | +4.9% / +6.5% |
| ค่าธรรมเนียม + slippage 2 เท่า | −8.5% / −46.2% | +4.7% / +5.7% |
| TRAILING 0.28 (unified) | −5.6% / −39.0% | +1.7% / +0.7% |
| TRAILING 0.42 (unified) | −6.2% / −34.0% | +2.8% / −5.3% |
| H_NEW 432 (label + time stop) | −6.8% / −38.3% | ไม่ได้ทดสอบ |
| H_NEW 648 | −5.4% / −46.5% | ไม่ได้ทดสอบ |
| MAE 0.20 (label + SL) | −8.6% / −45.9% | ไม่ได้ทดสอบ |
| MAE 0.30 | −5.5% / −35.7% | ไม่ได้ทดสอบ |

NEW_XGB ขาดทุนทุกการเปลี่ยนพารามิเตอร์ (−5% ถึง −9%/เทรด) — ผลลบไม่ได้มาจากพารามิเตอร์ที่โชคร้ายตัวเดียว; NEW_LOGISTIC ยังบวกเมื่อต้นทุน 2 เท่าและ trailing ±20% แต่การทดสอบ label ±20% ของ NEW_LOGISTIC ยังไม่ได้ทำ (ต้องฝึกใหม่)

### เกณฑ์หัวข้อ 11 (เฉพาะส่วน walk-forward)

| เกณฑ์ | NEW_XGB native | NEW_LOGISTIC native | NEW_LOGISTIC unified |
|---|---|---|---|
| 1. expectancy > LEGACY_FIXED และ CI ไม่คร่อม 0 | ✗ | ✗ (CI คร่อม 0) | ✗ |
| 2. ดีกว่า RANDOM P95 | ✗ (10.5) | ✓ (99.6) | ✗ (85.9) |
| 3. Sharpe หรือ Calmar > BTC_HOLD | ✗ | ✗ (0.37 / 0.11 vs 0.71 / 0.40) | ✗ |
| 4. กำไรเมื่อตัด top-3 และต้นทุน 2 เท่า | ✗ | ✓ | ✗ |
| **ผ่านทั้งหมด** | **✗** | **✗** | **✗** |

**ข้อสรุป: ยังไม่มี edge.** XGBoost ไม่ชนะ baseline (ทั้ง logistic และกฎ) จึงไม่แนะนำให้ใช้ XGBoost

## ฝั่ง DOWN (เพิ่มหลังอนุมัติ funding — commit cebee68)

funding ถูกเพิ่มใน common_eval ตามที่อนุมัติ โดยมีเงื่อนไขเพิ่ม: นับเฉพาะ settlement ที่ entry_time < t < exit_time (คิดจาก event ดิบ ไม่ใช่ผลรวมต่อแท่ง); exit กลางแท่ง (stop/TP/trail) ใช้เวลาเปิดของแท่งที่ออก; มีเทสต์กรณี settlement ก่อน/ตรงเวลาเข้าและตรงเวลาออก (รวม 7 เทสต์ funding, ทั้งหมด 21 เทสต์ผ่าน)

**LEGACY_FIXED short เมื่อคิด funding (S0 คงเดิมไม่มี funding):** S1 +0.77% → +0.08%, S2 +1.49% → +0.77%, **S4 (= LEGACY_FIXED short) +2.56% → +2.70%/เทรด, CAGR −2.3%, MaxDD −89.9%** (native); unified +0.02%

**NEW DOWN (futures 1x, funding, 2022-01 → 2025-06):**

| รุ่น | exit | เทรด | expectancy | CAGR | MaxDD | RANDOM pct (จับเวลา / ความถี่) | ต้นทุน 2× | ตัด top-3 | funding/เทรด |
|---|---|---|---|---|---|---|---|---|---|
| NEW_DOWN_BASELINE | native | 392 | +2.08% | +4.8% | −75.5% | 88.6 / 90.2 | +1.88% | +1.67% | −0.65% |
| NEW_DOWN_BASELINE | unified | 688 | +0.04% | −13.4% | −82.2% | 41.7 / 46.3 | −0.16% | −0.29% | −0.38% |
| NEW_DOWN_LOGISTIC | native | 206 | +0.04% | −15.7% | −80.3% | 89.4 / 29.6 | −0.16% | −0.75% | −0.73% |
| NEW_DOWN_XGB | native | 209 | +1.46% | −6.1% | −76.4% | 96.5 / 75.1 | +1.26% | +0.67% | +0.93% |
| NEW_DOWN_XGB | unified | 271 | +0.64% | −10.4% | −72.2% | 96.9 / 72.8 | +0.44% | −0.11% | +0.74% |
| + ตัวกรอง BTC < EMA1200 | | 48–185 | แย่ลงทุกโมเดล | | | | | | |
| LEGACY_FIXED short | native | 217 | +2.70% | −2.3% | −89.9% | | | | |

- **ไม่มีรุ่น DOWN ใดผ่านเกณฑ์หัวข้อ 11:** ไม่มีรุ่นที่ expectancy สูงกว่า LEGACY_FIXED short โดย CI ไม่คร่อม 0 และไม่มีรุ่นที่ชนะ RANDOM P95 ทั้งสองแบบ
- ความทนทานต่อ label ±20% (native): baseline บวกทั้ง 4 แบบ (+1.4% ถึง +2.9%); XGB บวกทั้ง 4 (+0.1% ถึง +3.5%); logistic ลบ 2 จาก 4
- ผลรายปี baseline: 2022 +233%, 2023 −58%, 2024 −24%, 2025H1 +11%; 12 เดือนหลัง development −1.1%/เทรด
- funding ที่ short ได้รับเฉลี่ยติดลบสำหรับ baseline/logistic (−0.7%/เทรด) — กฎเลือกเหรียญที่เพิ่งขึ้นแรงซึ่งมักมี funding ติดลบหลังราคากลับตัว (short ต้องจ่าย)
- สมมติฐาน DOWN สำหรับ holdout: **NEW_DOWN_BASELINE (native)** — เหตุผลใน `reports/holdout_hypotheses.md`

### (บันทึกเดิม) ข้อเสนอแก้ common_eval

spec หัวข้อ 9 กำหนดให้ DOWN คิด funding และผู้ใช้สั่งให้คิด funding ฝั่ง short ทั้ง LEGACY และ NEW แต่ common_eval (d7e94e7) ยังไม่รองรับ ผมจึง**ไม่ได้แก้** แต่เตรียม diff ไว้ที่ `proposals/common_eval_funding.diff`:

- `config.yaml`: `costs.FUNDING_SIDES: [short]`, `costs.FUNDING_DIR: data_funding`
- `common_eval/data.py`: `load_funding()` รวม settlement ดิบในช่วง (Open[t], Open[t]+4h] ต่อแท่ง (รองรับรอบ 8h/4h/1h) — ตรวจกับข้อมูลจริงแล้ว ผลรวมตรงกับไฟล์ดิบทุกตัว
- `common_eval/backtest.py`: `Engine(..., funding=True)`; short **ได้รับ** เมื่อ funding บวก, long จ่าย (ปิดอยู่ เพราะ UP เป็น spot); ปรับตามราคา (notional เปลี่ยนตามราคา); แท่งที่ออกด้วย stop ไม่นับ settlement ของแท่งนั้น; MTM รวม funding สะสม; trade log เพิ่ม `funding_ret`
- `common_eval/run_short.py`: S0 (AS_IS) ปิด funding เพื่อให้เป็นของเดิมจริง; S1–S4 เปิด
- `tests/test_funding.py`: 6 เทสต์ ผ่านทั้งหมดบนสำเนาที่ patch แล้ว (short ได้รับเมื่อบวก/จ่ายเมื่อลบ, long ไม่โดนคิดเมื่อไม่ได้ตั้งค่า, ปรับตามราคา, แท่งที่ stop, สวิตช์ปิดสำหรับ AS_IS)
- ขนาดผลกระทบที่คาด: ช่วง 2023 → 2025-06 short ได้รับ funding ราว 5–11% ต่อปี (BTC 8.8%, ZEC 4.9%, 1000PEPE 11.5%)

เมื่ออนุมัติ ผมจะ: apply diff → รัน LEGACY_FIXED short (S1–S4) ใหม่ → backtest DOWN (NEW_BASELINE / NEW_LOGISTIC / NEW_XGB, futures 1x) พร้อม RANDOM และ robustness → แล้วจึง freeze

## ทำไม LEGACY_FIXED ฝั่ง long แย่กว่าสุ่ม (ตามที่ขอ)

เทียบ 266 เทรดของ LEGACY_FIXED (native) กับเทรดจาก 200 RANDOM run ที่สุ่มจาก pool เดียวกัน (ฟีเจอร์ของ**วันก่อนหน้า**การเข้า):

- **สภาพคล่อง:** ไม่ต่าง (percentile สภาพคล่องมัธยฐาน 0.74 vs 0.72)
- **เลือก "สุดขั้ว" ไม่เลือก "กลางๆ":** 42% ของไม้มาจาก quintile บนสุดของผลตอบแทน 30 วัน และ 24% จาก quintile ล่างสุด (สุ่ม = 20% / 20%); มีเพียง 19% อยู่ใน 3 quintile กลาง (สุ่ม 60%) — ซึ่งเป็นช่วงที่สุ่มทำได้ดีที่สุด (+3.5% ถึง +4.6%/เทรด). แบบเดียวกันกับ close/EMA300 (45% บนสุด + 31% ล่างสุด). ไม้ของ LEGACY ผันผวนกว่า (ATR 4.7% vs 3.2%)
- **sector:** น้ำหนัก meme มากเกิน (10.9% vs 6.3%) ซึ่งเฉลี่ย −20.9%/เทรด; น้ำหนัก layer-1 น้อยเกิน (8.6% vs 14.9%) ซึ่งเป็นกลุ่มที่ดีที่สุดของ LEGACY (+10.8%)
- **กลไก:** label +100%/−40% ให้รางวัลกับความผันผวน — เหรียญที่แกว่งแรงแตะ +100% บ่อยกว่า โมเดลจึงเรียนรู้ "เลือกเหรียญผันผวนที่สุดขั้ว" ซึ่งก็แตะ −40% บ่อยกว่าเช่นกัน label วัด "ความแกว่ง" ไม่ใช่ "ทิศทาง"

## ข้อจำกัด
- sector เป็นการจัดกลุ่มของ CoinGecko ปัจจุบัน ไม่ใช่ point-in-time
- `funding_30d_isna` เป็นตัวบอกช่วงเวลาอ่อนๆ ในข้อมูลยุคแรก (spot-only) แม้แทบคงที่ในช่วง backtest
- ทดสอบหลายรุ่น (8 รุ่น × 2 exit ฝั่ง UP) — การผ่าน RANDOM P95 ของ NEW_LOGISTIC native อาจเป็นความบังเอิญจาก multiple comparisons
- ไม่ได้สร้าง skills ใน `.claude/skills/` (หัวข้อ 12 เป็นทางเลือก)
- ยังไม่ได้ freeze/tag `new-frozen` เพราะขั้นที่ 9 ฝั่ง DOWN ยังไม่เสร็จ

## ขั้นต่อไป
1. **อนุมัติ** `proposals/common_eval_funding.diff` (หรือแจ้งสิ่งที่ต้องแก้)
2. ผมรัน LEGACY_FIXED short ใหม่ + backtest DOWN ครบ แล้วรายงานเพิ่มเติม
3. freeze: commit + tag `new-frozen` (โมเดลสุดท้ายอยู่ที่ `new_research/models_frozen/`)
4. **รออนุญาตเปิด holdout (ขั้นที่ 10)** — สมมติฐานที่บันทึกไว้ล่วงหน้าสำหรับ holdout: NEW_LOGISTIC (UP, native exit) คือรุ่นเดียวที่มีโอกาส; คาดว่าไม่ผ่านเกณฑ์ข้อ 3
5. ถ้าจะทำรอบใหม่ (ไม่ใช่บน holdout): นิยาม base_duration ใหม่, ใช้ label ที่วัดผลตอบแทนสุทธิแทน barrier, และวัดทุกรุ่นด้วยการเลือกเหรียญภายในวันก่อนดู PR-AUC รวม
