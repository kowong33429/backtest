# รายงานตรวจสอบ LEGACY (ขั้นที่ 1–2) — 2026-10-18

ขอบเขต: entry model (long m1 / short_v2) และ `rotation_backtest.py` ใน `core/` (สำเนาเดิมอยู่ที่ `legacy/` บน branch `compare/legacy-vs-new`)
ยังไม่ได้แก้ bug ใน LEGACY ไม่ได้สร้าง NEW หรือ `common_eval/` และยังไม่ได้ทำขั้นที่ 3–8
ไม่พบ `prompts/new_research.md` หรือโค้ด NEW ในทุก branch รวมถึง `origin/feat/add-agent`
ตัวเลขทั้งหมดรันใหม่เมื่อ 2026-10-18 โดยไม่แก้โค้ดหลัก สคริปต์อยู่ที่ `reports/audit_scripts/`
ตารางละเอียดอยู่ใน `legacy_audit_20261018.xlsx`

---

## สรุปผู้บริหาร

1. **ผลที่ README รายงานส่วนใหญ่ไม่ใช่ edge จริง.** บน universe ที่ไม่มี survivorship (PIT top-120, `data_all`) entry model m1 ขาดทุน (expectancy −2.80%/เทรด, PF 0.85) จากเดิม +8.35% (PF 1.53) บน basket
2. **Rotation b7:** basket +120.1% CAGR → universe **−8.7%** CAGR (DD −76%) ช่วงเดียวกัน (2021-10-28→) ขณะที่ BTC buy&hold +7.0% และ `btc-trend` (ไม่ใช้โมเดล) **+30.3%**
3. **สาเหตุหลักข้อแรก (สูง):** basket ถูกเลือกจากเหรียญที่ rally >200% มาแล้วทั้ง 150 ตัว แค่ "ซื้อทุก bar" ด้วยกฎ TP/SL เดียวกันก็ได้ +5.09% บน basket แต่ −1.93% บน universe
4. **สาเหตุข้อสอง (สูง):** ทักษะของโมเดลเป็น regime/time proxy. TA ล้วน (90 ฟีเจอร์) ได้ PR-AUC เท่า base rate (0.141 vs 0.141; universe 0.074 vs 0.081). ผลมาจาก macro ช้าๆ (CPI/M2/yield curve) 64% ของ gain และคะแนนเฉลี่ยรายวันของทุกเหรียญอย่างเดียวก็ได้ PR-AUC เท่าโมเดลเต็ม
5. **สาเหตุข้อสาม (สูง):** ฝั่ง short คิดกำไรแบบ reciprocal. คำนวณแบบ short จริง expectancy +22.5% → **+0.62%**, PF 2.91 → **1.03**, +$27,947 → **+$763**
6. **ไม่พบ look-ahead ในฟีเจอร์/ไทม์มิ่ง:** ทดสอบ perturbation, lag ของ macro, ไทม์มิ่ง rotation และ purge/embargo ผ่านทั้งหมด. ปัญหาของ LEGACY อยู่ที่การเลือกข้อมูล ตัวแปรที่เป็นตัวแทนเวลา และสูตรคิดกำไร ไม่ได้อยู่ที่การรั่วของข้อมูลอนาคต
7. ความมั่นใจ: **สูง** สำหรับข้อ 1–3 (ทำซ้ำได้ ตัวเลขต่างกันหลายเท่า) และ **กลาง** สำหรับสัดส่วนของแต่ละสาเหตุ (ablation รันครั้งเดียวต่อรุ่น 1 seed)
8. ผลบน universe ไม่ได้พิสูจน์ว่าไม่มี edge เลย. cross-sectional ROC ของ m1_univ ≈ 0.53 (มีสัญญาณอ่อนมาก) แต่ไม่เหลือหลังออกด้วย TIME stop และค่าธรรมเนียม
9. benchmark ที่ NEW ต้องชนะบน universe เดียวกัน: **`btc-trend` +30.3% CAGR / DD −24.6%** (+24.8% หลัง funding ถ้าเทรดผ่าน perp) และ BTC hold +7.0%
10. สถานะ: ยังไม่มี NEW จึงยังตอบไม่ได้ว่า NEW ดีกว่าหรือไม่ ต้องได้ `new_research.md` ก่อนจึงทำขั้นที่ 3–8 ได้

---

## ขั้นที่ 1: LEGACY ทำอะไร

| หัวข้อ | รายละเอียด |
|---|---|
| เป้าหมาย | ต่อ 4H bar ทายว่า "rally +100% ภายใน 60 วันก่อนโดน −40%" จะเริ่มที่นี่หรือไม่ (long, m1). ฝั่ง short เป็นภาพสะท้อน (30 วัน). rotation ถือ top-K ตาม OOF prob + leader/breakout sleeve + BTC fallback |
| Label | triple barrier TP +100% / SL −40% / horizon 60d (m1, 360 bar) หรือ 30d (original, short_v2). เข้า Open[i] สแกน High/Low ตั้งแต่ bar i+1. same-bar → SL. weight = AFML uniqueness |
| ฟีเจอร์ 108 | TA scale-free 90 (RSI×10, ADX×10, BB_Width×9, Dist_SMA×11, Return×11, Dist_High/Low×8, Volume_Surge×10, Pos_In×8, Dist_VWAP×8, SMA_Cross, ATR_Ratio) + non-TA 18 (Funding_Rate, FearGreed, SP500/DXY/GOLD/OIL ret20d/50d, YieldCurve_10Y_2Y, M2 YoY/MoM, CPI YoY/MoM, FEDFUNDS, VIX, HY_SPREAD) |
| โมเดล | XGBoost 300 ต้น lr 0.03 depth 4 subsample/colsample 0.8 scale_pos_weight = neg/pos. short_v2: 600 ต้น depth 5 min_child_weight 50 colsample 0.5 |
| Train/Test | purged expanding walk-forward (6 fold long, 5 fold short), purge ตาม `Label_EndDate`, embargo 3 วัน. backtest ใช้เฉพาะ OOF |
| กฎเข้า/ออก | prob ≥ 0.60 → เข้า Open ของ bar ถัดไป → TP +100% / SL −40% / time stop 60d (Close). 1 ไม้ต่อเหรียญ $100 ไม่จำกัดทุน |
| ต้นทุน | fee 0.10% + slip 0.05% ต่อข้าง. ไม่มี funding. ไม่มี leverage/liquidation |
| Rotation | รายวัน top-10, rebal 7 วัน, hyst 2, breadth>0.5, leader 3 + young 2, BTC fallback, vol target 70%, ต้นทุน 0.15%/ข้าง |
| Universe | basket 150 เหรียญ (144 ใน panel) 2018–2026-10 / `data_all` 703 perp 2020-01-01→2026-10-13 (OOF m1_univ 347 เหรียญ) |
| ผลเดิม | m1: 2,154 เทรด win 43.2% exp +8.35% PF 1.53 +$17,976. short_v2: 1,240 เทรด exp +22.5% PF 2.91. b7 basket +198.8% CAGR / DD −39.5%. **ทำซ้ำได้ตรงทั้งหมด** (b7 universe ได้ −8.7% ต่างจาก README −7.9% เพราะข้อมูลใหม่กว่า) |

---

## ขั้นที่ 2: ปัญหาเรียงตามความรุนแรง

### สูง

**H1 Survivorship / hindsight ของ basket** — `core/trend_scanner.py:443`, `core/dataset.py:82-104`
ทั้ง 150 เหรียญใน basket มี rally >200% อย่างน้อยหนึ่งครั้งในผล scan ซึ่งเป็นการเลือกด้วยผลลัพธ์ในอนาคต

| ตัวชี้วัด | basket | universe PIT top-120 |
|---|---|---|
| m1 thr0.6 expectancy / PF | +8.35% / 1.53 (2,154 เทรด) | **−2.80% / 0.85** (2,502 เทรด) |
| ซื้อทุก bar ไม่ใช้โมเดล | +5.09% / 1.31 | −1.93% / 0.89 |
| สุ่มความถี่เท่ากัน (3 seed) | +4.44…+4.91% | −1.90…−2.28% |
| rotation b7 (2021-10-28→) CAGR / DD | +120.1% / −39.5% | **−8.7% / −76.1%** |

ราว 60% ของ expectancy บน basket (+5.09 จาก +8.35) มาจากการเลือกเหรียญ ไม่ใช่จากโมเดล
วิธีแก้: ใช้ universe point-in-time (`data_all` + `volume_rank`) เป็นค่าเริ่มต้นเสมอ

**H2 โมเดลจับ "ช่วงเวลา" ไม่ใช่ "เหรียญ"** — `core/dataset.py:157-166`, `core/train_entry_model.py:134`
Ablation (XGBoost พารามิเตอร์เดิม, fold เดิม, 1 seed):

| data | ฟีเจอร์ | PR-AUC | base rate | ROC | PR-AUC ของ "ค่าเฉลี่ยรายวันอย่างเดียว" | cross-sectional PR-AUC |
|---|---|---|---|---|---|
| basket | เต็ม 108 | 0.219 | 0.141 | 0.576 | 0.219 | 0.156 |
| basket | **TA ล้วน 90** | **0.141** | 0.141 | 0.479 | 0.137 | 0.148 |
| universe | เต็ม 108 | 0.077 | 0.081 | 0.503 | 0.075 | 0.092 |
| universe | **TA ล้วน 90** | **0.074** | 0.081 | **0.462** | 0.065 | 0.100 |

- gain 64–66% มาจาก 18 ฟีเจอร์ non-TA โดย `CPI_YoY` ตัวเดียว 15–18%, ตามด้วย M2, yield curve, FEDFUNDS, HY_SPREAD ซึ่งมีค่าไม่ซ้ำเพียง ~45–115 ค่าตลอด 7 ปี จึงเท่ากับดัชนีเวลา
- ที่ความถี่สัญญาณเท่ากัน (top 24%): โมเดลเต็ม +7.64%, TA ล้วน **+2.55%** (ต่ำกว่าสุ่ม +4.4…+4.9%) บน basket
- universe: precision ที่ prob ≥0.5/0.6/0.7 = 7.9/7.7/7.4% ต่ำกว่า base 8.1% และลดลงเมื่อ prob สูงขึ้น

วิธีแก้: ตัดหรือแปลงฟีเจอร์ macro ที่เปลี่ยนช้า. วัดผลด้วยอันดับข้ามเหรียญภายในวัน (demean ต่อวัน) ไม่ใช่ PR-AUC รวม

**H3 กำไรฝั่ง short คิดแบบ reciprocal** — `core/entry_backtest_short.py:21-23,100`, `core/labeler_short.py:13-25`
สูตร `entry/exit − 1` ทำให้ราคาลงครึ่งหนึ่งได้ +100% และราคาขึ้น ×1.667 ขาดทุนแค่ −40% แต่ short จริงได้ +50% และเสีย −66.7%

| short_v2 thr0.5 (1,240 เทรด) | log เดิม | short จริง (1 − exit/entry) |
|---|---|---|
| expectancy | +22.54% | **+0.62%** |
| PF | 2.91 | **1.03** |
| กำไรรวม ($100/ไม้) | +$27,947 | **+$763** |
| ขาดทุนต่อไม้ที่โดน SL | −40% | −66.7% |

รายปีของ short จริง: 2020 −19%, 2021 −6%, 2022 −12%, 2023 −8%, 2024 −3%, **2025 +15%**, 2026 −2% (กำไรอยู่ในปีเดียว)
วิธีแก้: คิด P&L ของ short เป็น `1 − exit/entry` และปรับ barrier ของ label ให้สอดคล้อง. เพิ่ม funding และ borrow

### กลาง

| ID | ปัญหา | ไฟล์:บรรทัด | ผลกระทบเป็นตัวเลข | วิธีแก้ |
|---|---|---|---|---|
| M1 | ไม่มีเพดานทุน เทรดไม่อิสระต่อกัน | `entry_backtest.py:58-98` | ถือพร้อมกันสูงสุด 134 ไม้ (เฉลี่ย 51). วันเดียวเข้า 73 ไม้ (2023-07-26). month-block bootstrap 95% CI ของ expectancy: basket [+1.1, +15.9]%, universe [−10.1, +5.0]%. positive 146,413 bar = 3,418 episodes | รายงานระดับพอร์ตที่จำกัดทุน + block bootstrap |
| M2 | เลือก threshold/พารามิเตอร์/preset บน OOF เดียวกับที่รายงาน ไม่มี holdout | `train_entry_model.py:225-228`, README | ≥11 รุ่นโมเดล, sweep thr, rotation 15 รอบ ~20 พารามิเตอร์. ความไว (basket CAGR 120.1%): rebal 6/8 → 82.0/82.7; near-high 0.20/0.30 → 98.2/105.0; lookback 48 → 99.1. บน universe ทุกค่าติดลบ | ล็อก holdout, นับจำนวน trial |
| M3 | ต้นทุนคงที่ ไม่ผูกสภาพคล่อง ไม่คิด funding | `entry_backtest.py:144-145`, `rotation_backtest.py:314` | turnover 38–47×/ปี. ต้นทุน 2×: basket +120.1→+107.1%, 4×: +83.3%; universe −8.7→−14.8%. funding ทำให้ btc-trend 30.3→24.8% | slippage ตามสภาพคล่อง + funding จริง |
| M4 | NaN → 0 กลายเป็นธงช่วงเวลา | `train_entry_model.py:134`, `predict_entry.py:61`, `rotation_paper.py:122`, `data_loader.py:105,205` | HY_SPREAD NaN 35.2% (FRED ให้ข้อมูลเริ่ม 2023-10-17). Funding_Rate=0 ใน 13.6% ของแถว (ก่อนมี perp). ตัดสองตัวนี้แล้ว PR-AUC 0.221 vs 0.219 จึงไม่ใช่ตัวขับหลัก | ปล่อย NaN ให้ XGBoost หรือ drop |
| M5 | Label ไม่ตรงกับการเทรดจริง | `labeler.py:91-94`, `entry_backtest.py:61` | label เข้า Open[i] ไม่นับ bar i เอง, เทรดจริงเข้า Open[i+1] (ไม่ใช่ leak). 53% ของเทรดจบที่ TIME stop (เฉลี่ย +1.6%) ซึ่ง label นับเท่ากับ SL. win rate 43% มาจาก TIME exit เป็นหลัก | นิยาม label ด้วยผลตอบแทนสุทธิตามกฎออกเดียวกัน |
| M6 | prob ไม่ calibrate | `train_entry_model.py:152-157` | basket: prob ≥0.6 → precision จริง 20.1%, ≥0.9 → 38.2% | isotonic หรือใช้ top-k ต่อวัน |
| M7 | คุณภาพ `data_all` | `universe_download.py`, `rotation_backtest.py:254-255,403` | 162/703 เหรียญมี bar แบน volume=0 รวม 341,385 bar (8.8%; BTCST, SC, FTT, RAY แบน 2–5 ปี). มีเพียง 31/703 เหรียญที่ข้อมูลจบก่อน 2026-10-01 อาจตกหล่น perp ที่ delist (**ยังไม่ได้เทียบกับรายการ delist ของ Binance**) | ตัด bar volume=0, ตรวจรายการ delist |
| M8 | ผลขึ้นกับ regime | `rotation_backtest.py:331-354` | universe b7: BTC<SMA200 (902 วัน) รวม **−66.8%**, BTC>SMA200 (909 วัน) +87.8% | benchmark = btc-trend |

### ต่ำ

| ID | ปัญหา | ไฟล์:บรรทัด | หมายเหตุ |
|---|---|---|---|
| L1 | `^IRX` ถูกใช้เป็น US2Y จริงคือ T-bill 13 สัปดาห์ ทำให้ `YieldCurve_10Y_2Y` เป็น 10Y−3M | `entry_finder.py:254,335`, `data_loader.py:50` | ยืนยันแล้ว ไม่ใช่ leak. อยู่อันดับ 4–6 ของ gain (4.8–9.6%). ใช้ FRED DGS2 |
| L2 | Fear&Greed ไม่เลื่อน D+1, หลุดเงียบถ้า API timeout, FRED ใช้ค่าที่ revise แล้ว | `dataset.py:157-159`, `entry_finder.py:387-395` | gain 0.4–1.7% |
| L3 | ไม่มี leverage/liquidation. `backtester.py` เก่ามี `max_position=3.0` ไม่มีโมเดล liquidation | `backtester.py:54` | ตัวเลขที่รายงานเป็น unleveraged จึงไม่เสี่ยงถูก liquidate. `backtester.py` ไม่ใช้กับผลหลักและผมไม่ได้ตรวจลึก |
| L4 | SL/TP เติมที่ระดับเป๊ะ | `exits.py:62-66` | ตรวจ SL 611+315 ไม้ ไม่มีไม้ที่ gap ทะลุ stop → ผลกระทบ 0 |
| L5 | fold แรกฝึกข้อมูลน้อย, 108 ฟีเจอร์ซ้ำซ้อน (window ก้าวละ 10) | `train_entry_model.py:53`, `features.py:13-18` | ~3.4k episodes ต่อ 108 ฟีเจอร์ |

### ตรวจแล้วไม่พบปัญหา
- **N-1 ของฟีเจอร์ TA:** แก้ OHLCV ของ bar i ×3 แล้วฟีเจอร์ของ bar i ไม่เปลี่ยน, bar i+1 เปลี่ยน, bar ก่อนหน้าไม่เปลี่ยน (3 ครั้ง, ZEC 4H). การทดสอบครั้งที่ 4 ล้มด้วยข้อผิดพลาดของสคริปต์ทดสอบเอง (bar ที่ถูก dropna) ไม่ใช่ผลของ leak
- **Macro lag:** VIX ปิด 2024-03-08 = 14.74 ปรากฏที่ 2024-03-09. FRED lag ต่อซีรีส์ (M2 60, CPI 45, FEDFUNDS 35, HY 2) สมเหตุสมผล
- **Rotation timing:** ตัดสินใจที่ close d เทรดที่ open d+1 (`W.shift(1)`), vol target `shift(2)`
- **Purge/embargo:** purge ตาม `Label_EndDate`, embargo 3 วัน, test เป็นบล็อกเวลาอนาคตร่วมกันทุกเหรียญ
- **ตัวชี้วัด:** ใช้ PR-AUC ไม่ใช่ accuracy

---

## ข้อควรระวังของตัวเลขชุดนี้
- Universe และ basket ใช้ OOF คนละโมเดลและคนละช่วงเวลา (universe เริ่ม 2020-12-31). ผมเทียบ basket เฉพาะ 2021+ แล้ว (+6.82% vs −2.80%) ข้อสรุปไม่เปลี่ยน
- OOF ของ universe ครอบคลุมเฉพาะแถวที่เหรียญอยู่ใน PIT top-120
- Ablation แต่ละรุ่นใช้ 1 seed. ยังไม่ได้ทำ CI ของ PR-AUC
- ตัวเลข "ต้นทุน" ใช้ flat cost เดิม ไม่ได้สร้าง slippage ตามสภาพคล่อง
- ไม่ได้ตรวจ `research_agent.py`, `universal_researcher.py`, `optimizer.py`, `labels.py`, `backtester.py` (pipeline เดี่ยวรุ่นเก่า) นอกจากข้อสังเกตข้างต้น
- ไม่ได้ตรวจ ZEC รายรุ่น (เป็นเรื่องของขั้นที่ 5)

## ข้อแนะนำขั้นต่อไป
1. **ส่ง `prompts/new_research.md`** เพื่อเริ่มขั้นที่ 3–8. common_eval ควรบังคับ universe PIT และมี benchmark `btc-trend` + ซื้อทุก bar + RANDOM เป็นค่าเริ่มต้น
2. ในโมเดลใหม่ **อย่าใส่ macro ที่เปลี่ยนช้า** เป็นฟีเจอร์ต่อเหรียญ. ถ้าจะใช้ ให้ใช้เป็น regime gate แยกและวัดแยกจากการเลือกเหรียญ
3. ใช้ **cross-sectional metric** (rank IC / PR-AUC หลัง demean ต่อวัน) เป็นเกณฑ์หลักแทน PR-AUC รวม
4. P&L ฝั่ง short ต้องเป็น `1 − exit/entry` และรวม funding/borrow ก่อนรายงานใดๆ
5. ตรวจรายการ perp ที่ delist ของ Binance เทียบกับ `universe_symbols.json` และตัด bar volume=0 ก่อนสร้าง universe ที่จะใช้เป็นสนามร่วม
