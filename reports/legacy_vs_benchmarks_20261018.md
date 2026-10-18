# LEGACY เทียบ benchmark บนสนามทดสอบร่วม `common_eval` (ข้อ 3) — 2026-10-18

ใช้เฉพาะช่วง development + walk-forward (2022-01-01 → 2025-06-30). **ไม่ได้แตะ final holdout (≥ 2025-07-01)**
รายละเอียดทุกตัวเลขอยู่ใน `legacy_vs_benchmarks_20261018.xlsx` กราฟอยู่ใน `reports/charts/`
ยังไม่มี NEW จึงยังไม่มีการเทียบ NEW vs LEGACY (ข้อ 2 ทำต่อจากนี้)

## ตอบคำถามหลัก

**LEGACY_FIXED ดีกว่า RANDOM ที่ percentile 95 หรือไม่?**

| รุ่น | โหมด RANDOM | exit native | exit unified |
|---|---|---|---|
| **LEGACY_FIXED long (m1)** | rate-matched | ไม่ผ่าน (expectancy percentile 0.3, CAGR 0.6) | ไม่ผ่าน (0.4 / 0.0) |
| | timing-matched | ไม่ผ่าน (9.7 / 7.8) | ไม่ผ่าน (1.6 / 0.0) |
| **LEGACY_FIXED short** | rate-matched | **ผ่าน** (expectancy 95.5, CAGR 99.7, Sharpe 99.2) | ไม่ผ่าน (84.5 / 93.2 / 80.9) |
| | timing-matched | expectancy ไม่ผ่าน (88.3) แต่ CAGR/Sharpe ผ่าน (98.2 / 97.8) | ไม่ผ่าน (60.2 / 56.8 / 53.9) |

- **ฝั่ง long: ไม่ดีกว่า RANDOM เลย และอยู่ในหาง "แย่กว่าสุ่ม"** (percentile 0–23) ทั้งสองโหมด exit และทั้งสองแบบการจับคู่
- **ฝั่ง short: ผ่าน P95 เฉพาะ exit แบบ native** และไม่ผ่านใน unified. แม้ผ่าน ก็ไม่ใช่รุ่นที่ทำกำไร: CAGR −3.2%, MaxDD −90.7% (BTC_HOLD +26.6% / −66.9%) จึงสรุปได้ว่ามีสัญญาณอ่อนในหางของการจัดอันดับ แต่ไม่มี edge ที่ใช้เทรดได้
- ตัวเลขนี้เป็นการรันครั้งเดียวต่อรุ่น (217–440 เทรด) ความไม่แน่นอนสูงและมีหลายการเปรียบเทียบ (4 โหมด × 3 เมตริก) การผ่านเพียงหนึ่งในสี่โหมดไม่ควรถือเป็นหลักฐานแข็ง

## สนามทดสอบ (common_eval)
- Universe: `data_all` 503 perp ที่มีข้อมูลในช่วงนี้ (เฉลี่ย **220 เหรียญ/แท่ง** ผ่าน MIN_QUOTE_VOL_30D 500k); basket (LEGACY_AS_IS เท่านั้น): 140 เหรียญ เฉลี่ย 89/แท่ง
- ข้อมูล: ตัด bar ที่ volume=0 (ใช้แทน "ขั้น B1" ซึ่งยังไม่มีนิยามจนกว่าจะได้ `new_research.md`)
- backtester ตัวเดียว: เข้า Open ของ bar ที่ signal ระบุ, fee 0.10%/ข้าง, slippage = 0.20% × clip(√(10M / qv30), 1, 5) ต่อข้าง (0.20–1.0%), MAX_POSITIONS 10, ไม้ละ equity/10, ไม่มี leverage, slot คืนหลัง bar ที่ออก
- exit: **native** = กฎเดิม (long TP +100%/SL −40%/60d; short mirror) และ **unified** = trailing 35% จาก peak + time stop 540 bar
- Walk-forward: expanding, test รายครึ่งปี 7 บล็อก (2022-01 → 2025-06), purge ตาม label end + embargo 540 bar (90 วัน) ฝึกด้วยข้อมูลก่อน test_start − 90 วันเท่านั้น. ข้อมูล/แถว panel ที่ ≥ 2025-07-01 ไม่ถูกโหลดเลย (มี guard ใน `common_eval/config.py`)
- **ตรวจสอบ harness:** ตั้ง common_eval ให้เหมือนกฎเดิมของ LEGACY (ไม่จำกัดไม้ slippage 0.05% ไม่กรองสภาพคล่อง) ได้ 1,488 เทรด / +11.12% เทียบกับ log เดิม 1,468 เทรด / +11.18% (ช่วงออกก่อน 2025-04-30) — ใช้ยืนยันว่า engine ไม่ได้ทำให้ผลต่างจาก LEGACY โดยไม่ตั้งใจ

## ผลบันได LEGACY (สะสม) — ช่วง walk-forward 2022-01 → 2025-06

### ฝั่ง long (m1, threshold 0.60)

| ขั้น | exit | เทรด | win | expectancy | CAGR | MaxDD | Sharpe | PR-AUC (base) |
|---|---|---|---|---|---|---|---|---|
| **L0 AS_IS** (basket + OOF เดิม) | native | 233 | 39.1% | +0.58% | −26.9% | −78.9% | −0.10 | 0.148 (0.136) |
| | unified | 302 | 33.1% | +3.16% | −12.1% | −73.5% | 0.19 | |
| **L1 +H1** (universe PIT, OOF m1_univ เดิม) | native | 271 | 36.2% | −0.06% | −35.3% | −86.1% | −0.08 | 0.081 (0.090) |
| | unified | 328 | 30.2% | −0.78% | −37.0% | −88.8% | −0.08 | |
| L1c ควบคุม (ฝึกใหม่ด้วยโครง fold ร่วม, 108 ฟีเจอร์) | native | 246 | 29.3% | −5.66% | −49.8% | −92.3% | −0.39 | 0.104 (0.090) |
| | unified | 331 | 28.1% | −5.61% | −55.3% | −94.9% | −0.46 | |
| **L2 +H2** (ตัด macro ช้า 7 ตัว) | native | 270 | 30.0% | −3.48% | −46.5% | −89.8% | −0.35 | **0.086 (0.090)** |
| | unified | 341 | 27.0% | −3.46% | −50.2% | −93.5% | −0.37 | |
| **L4 +M4** (ไม่ fillna(0)) = **LEGACY_FIXED** | native | 266 | 29.7% | **−4.88%** | **−50.3%** | −92.1% | −0.43 | 0.086 (0.090) |
| | unified | 341 | 26.1% | **−6.46%** | **−57.8%** | −96.4% | −0.58 | |

- H3 ไม่เกี่ยวกับฝั่ง long (ใช้กับ short เท่านั้น) จึงไม่มีแถวแยก
- **การเปลี่ยนแปลงต่อขั้น (native):** AS_IS→+H1: expectancy +0.58% → −0.06%, CAGR −26.9% → −35.3% (ผลของ survivorship; ผลต่างเล็กในตัวเลขนี้เพราะ cap 10 ไม้ทำให้ AS_IS ถูกบีบ ดูตาราง bridge). +H1→ควบคุม: −5.6pt expectancy (ผลของการฝึกใหม่ด้วย embargo 90 วันและ fold ร่วม ไม่ใช่ bug). +H2: ตัด macro ช้าแล้ว **PR-AUC ตกจาก 0.104 เป็น 0.086 ต่ำกว่า base rate 0.090 และ ROC 0.546 → 0.474** — ยืนยันผลตรวจ H2 ว่าทักษะของโมเดลมาจาก macro ที่เป็นตัวแทนเวลา. +M4: ผลเล็ก (expectancy −3.48% → −4.88% อยู่ในความแปรปรวนของสุ่ม)
- **ตัวชี้วัดโมเดล:** L2/L4 precision@1% 9.5% (lift 1.04–1.06), @10% lift 0.88–0.91 → ไม่มีทักษะ

### มุมมองตัดปี 2022 ออก (2023-01 → 2025-06) — ปี 2022 ลบหนักทุกรุ่น และเป็นบล็อกแรกที่ฝึกด้วยข้อมูลสั้นที่สุด

| รุ่น | exit | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|
| L0 AS_IS | native / unified | +14.0% / +25.8% | −74% / −74% | 0.53 / 0.68 |
| L1 +H1 | native / unified | +7.2% / +20.6% | −78% / −69% | 0.50 / 0.64 |
| L4 FIXED | native / unified | −22.8% / −30.4% | −75% / −77% | 0.04 / −0.05 |
| BTC_HOLD | | **+111.1%** | −28.1% | 1.78 |
| EQUAL_WEIGHT | | +4.3% | −76.4% | 0.44 |

แม้ตัดปี 2022 ลำดับก็เหมือนเดิม: ยิ่งแก้ bug ยิ่งแย่ลง และทุกรุ่นแพ้ BTC_HOLD

### ฝั่ง short (short_v2)

| ขั้น | exit | เทรด | win | expectancy | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|---|---|---|
| **S0 AS_IS** (basket, สูตร reciprocal ที่เป็น bug) | native | 141 | 61.7% | **+23.35%** | **+98.4%** | −47.5% | 1.51 |
| | unified | 296 | 48.0% | +9.25% | +75.4% | −48.5% | 1.29 |
| **S1 +H3** (กำไร short จริง 1−exit/entry) | native | 141 | 61.7% | +0.77% | −9.6% | −79.0% | 0.04 |
| | unified | 295 | 47.8% | +0.47% | −3.4% | −64.3% | 0.11 |
| S2 +H1 (universe + ฝึกใหม่) | native | 246 | 62.6% | +1.49% | −20.5% | −90.8% | 0.05 |
| | unified | 560 | 45.4% | +0.52% | −6.6% | −83.3% | 0.18 |
| S3 +H2 | native | 222 | 64.0% | +2.76% | −9.0% | −90.3% | 0.19 |
| | unified | 462 | 43.7% | −0.33% | −14.3% | −83.7% | −0.01 |
| **S4 +M4 = LEGACY_FIXED short** | native | 217 | 65.0% | +2.56% | −3.2% | −90.7% | 0.28 |
| | unified | 440 | 46.8% | +0.27% | −11.6% | −82.4% | 0.03 |

- **H3 คือการแก้ที่ใหญ่ที่สุดในทั้งงาน:** CAGR +98.4% → −9.6% จากการเปลี่ยนสูตรกำไรอย่างเดียว (ราคาเดียวกัน สัญญาณเดียวกัน)
- S2–S4 ใช้ threshold ที่ตั้งจากคะแนนช่วง development ให้ได้สัดส่วน signal เท่า S0 (3.24%) เพราะ 0.5 ตรงๆ ให้ signal 0.8k–7k หลังฝึกใหม่
- โมเดล short หลังตัด macro ช้า: ROC 0.439 (<0.5) แต่ precision@1% = 14.6% vs base 4.7% (lift 3.1) — มีสัญญาณเฉพาะหางบนสุด ซึ่งสอดคล้องกับที่ผ่าน P95 ของ RANDOM (native)

### Bridge: ทำไม LEGACY_AS_IS ใน common_eval จึงต่างจากตัวเลข README (long, basket, native)

| ขั้น | เทรด | expectancy |
|---|---|---|
| B0 กฎ LEGACY (2019+, slip 0.05%, ไม่จำกัดไม้) | 1,631 | +11.03% |
| B1 + ช่วง 2022-01 → 2025-06 | 1,223 | +6.12% |
| B2 + ต้นทุนร่วม (fee 0.10% + slippage ตามสภาพคล่อง ≥0.20%) | 1,223 | +5.37% |
| B3 + กรองสภาพคล่อง 500k + ตัด bar volume=0 | 1,187 | +5.03% |
| **B4 + MAX_POSITIONS=10 (เรียงตามคะแนน)** | **233** | **+0.58%** |
| B4r เหมือน B4 แต่เรียงลำดับแบบสุ่ม | 227 | +0.82% |
| B5 B4 แต่ exit unified | 302 | +3.16% |

ตัวทำลายผลใหญ่สุดคือเพดาน 10 ไม้ (คำนวณจริงแบบมีทุนจำกัด) ส่วนการเรียงตามคะแนนไม่ช่วยอะไร (สุ่มได้ผลเท่ากัน) ซึ่งสอดคล้องกับข้อสรุปว่าคะแนนไม่ได้บอกว่าเหรียญไหนดีกว่า ณ เวลาเดียวกัน

## Benchmark (2022-01 → 2025-06, ต้นทุนชุดเดียวกัน)

| รุ่น | CAGR | MaxDD | Sharpe | หมายเหตุ |
|---|---|---|---|---|
| BTC_HOLD | **+26.6%** | −66.9% | 0.71 | ปี 2022 −65%, 2023 +156%, 2024 +121%, 2025H1 +14% |
| EQUAL_WEIGHT (รายเดือน, ~215 เหรียญ) | −36.7% | −85.2% | −0.15 | alt เฉลี่ยแพ้ต้นทุนและ drawdown |
| LEGACY_AS_IS long (native) | −26.9% | −78.9% | −0.10 | |
| LEGACY_FIXED long (native / unified) | −50.3% / −57.8% | −92.1% / −96.4% | −0.43 / −0.58 | |
| LEGACY_FIXED short (native / unified) | −3.2% / −11.6% | −90.7% / −82.4% | 0.28 / 0.03 | |

RANDOM (1,000 ครั้ง, long, rate-matched): CAGR P5/P50/P95 = −46% / −36% / −22% (native), −48% / −38% / −24% (unified) ตลาดช่วงนี้ทำให้สุ่มติดลบหนักเอง LEGACY_FIXED long (−50%, −58%) แย่กว่า P5 ในโหมด unified

## ข้อควรระวัง
- **ความแปรปรวนสูง:** 140–560 เทรดที่ถือพร้อมกัน ≤10 ไม้ และผลถูกกำหนดโดยไม่กี่ช่วงเวลา (ปี 2022 ลดลง 76–88% ในทุกรุ่น long). ยังไม่ได้ทำ bootstrap CI ของ expectancy ในรายงานนี้ (อยู่ในขั้นที่ 5)
- **ยังไม่ได้ปรับ threshold ฝั่ง long หลังแก้ bug** (ใช้ 0.60 ตามเดิมเพื่อแยกผลของ bug) จึงไม่ได้ตอบว่า "ถ้าเลือก threshold ใหม่จะดีขึ้นหรือไม่"
- **ไม่มี funding** ในการ backtest (long perp ปกติเสีย, short ปกติได้) และ slippage เป็นสูตรตามสมมติฐานที่ตรึงใน `config.yaml` (SLIP_REF_QV, SLIP_MAX_MULT) ไม่ใช่ค่าที่วัดจากตลาดจริง
- **เกณฑ์เลือก threshold ฝั่ง short หลังฝึกใหม่** ใช้สัดส่วน signal ของ S0 (กฎที่ตั้งก่อนเห็นผล)
- AS_IS ใช้ basket และข้อมูล `data/` (spot-style เก่า) ส่วนรุ่นอื่นใช้ `data_all` — ตั้งใจเพราะ H1 คือการเปลี่ยน universe
- พารามิเตอร์/preset ของ LEGACY ถูกเลือกโดยผู้เขียนเดิมโดยเห็นช่วง 2025–26 มาแล้ว ผมไม่ได้ปรับใหม่
- rotation (`rotation_backtest.py`) ยังไม่ได้ผ่าน common_eval เพราะเป็นระบบถือ weight รายวัน ไม่ใช่สัญญาณรายแท่ง (ผลเดิมอยู่ในรายงาน audit)
- ผล RANDOM ของ short มี 1 จาก 4 โหมดที่ผ่าน P95 ซึ่งเสี่ยงต่อ multiple comparisons

## สิ่งที่ส่งมอบ
- `config.yaml`, `common_eval/` (`config.py`, `data.py`, `backtest.py`, `metrics.py`, `walkforward.py`, `benchmarks.py`, `legacy_adapter.py`, `run_long.py`, `run_short.py`, `run_bench.py`, `run_bridge.py`, `validate_vs_legacy.py`, `build_report.py`)
- `reports/legacy_vs_benchmarks_20261018.xlsx` (config, ladder, post-dev, bridge, benchmarks, random_vs_fixed, random_runs 8,000 แถว, yearly, ex_2022, trade logs)
- `reports/charts/`: `equity_drawdown_native|unified`, `random_hist_long|short`
- panel short บน universe: `data_all/model/entry_panel_univ_short.parquet` (git-ignored)

## ก่อนทำข้อ 2
กรุณาส่ง `new_research.md` ผมจะบันทึกที่ `prompts/new_research.md` แล้วสร้าง NEW บน `feat/new-research` โดยใช้ `config.yaml` และ `common_eval/` ตัวนี้โดยไม่แก้ ถ้าจำเป็นต้องแก้ ผมจะหยุดถามก่อน
