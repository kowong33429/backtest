# NEW Research Spec

คุณคือ quant researcher ที่เชี่ยวชาญ Python (pandas, numpy, xgboost, scikit-learn, shap)
เป้าหมาย: หา pattern ก่อนเหรียญคริปโตขึ้น >= 2 เท่า (UP) หรือลง >= 50% (DOWN)
และสร้าง scanner ที่ใช้ซ้ำได้ ทำงานบน branch `feat/new-research` เท่านั้น

---

## 0. บริบทจาก LEGACY (ต้องอ่านก่อนเริ่ม)

ผลจาก `reports/legacy_vs_benchmarks_20261018.md` (commit d7e94e7):
- ผลดีเดิมของ LEGACY มาจาก bug: survivorship (H1), macro ช้า (H2),
  สูตรกำไร short ผิด (H3 ทำให้ CAGR จาก +98% เหลือ −10%), fillna(0) (M4)
- LEGACY_FIXED ฝั่ง long แย่กว่า RANDOM (percentile 0.3–9.7)
- เพดาน 10 ไม้ทำให้ expectancy LEGACY จาก +5.0% เหลือ +0.6%
  แปลว่าผลเดิมพึ่งการถือหลายเหรียญพร้อมกันในตลาดขาขึ้น (beta) ไม่ใช่การเลือกเหรียญ
- ปี 2022 ทำให้ทุกรุ่นฝั่ง long ขาดทุนหนัก
- benchmark ที่ต้องชนะ: BTC_HOLD +26.6% / DD −67%

บทเรียนที่ NEW ต้องนำไปใช้:
1. สูตรกำไรทุกแบบต้องมี unit test ก่อนใช้ (บทเรียนจาก H3)
2. ต้องมีตัวกรองสภาวะตลาด (regime filter) ฝั่ง long
3. ผลต้องมาจากการเลือกเหรียญ ไม่ใช่จากการถือตลาดทั้งก้อน
4. วัดผลหลังเพดาน MAX_POSITIONS เสมอ

---

## 1. กฎที่ห้ามละเมิด

- ใช้ `config.yaml` และ `common_eval/` ตามที่มีอยู่ (commit d7e94e7) ห้ามแก้
  ถ้าจำเป็นต้องแก้ ให้หยุดและถามผู้ใช้ก่อน ค่าเฉพาะของ NEW ให้เพิ่มเป็น key ใหม่
  ใต้หัวข้อ `new:` ใน config.yaml เท่านั้น
- **Holdout ตั้งแต่ 2025-07-01 ล็อกไว้** ห้ามโหลด ห้ามดู ห้ามใช้ปรับอะไร
  จนกว่าผู้ใช้จะอนุญาตในขั้นที่ 10
- feature ใช้ข้อมูล ณ bar t และก่อนหน้าเท่านั้น label เท่านั้นที่มองอนาคตได้
- ข้อมูลรายวันต้องเลื่อนอย่างน้อย D+1 ก่อน merge เข้า 4H
- ห้าม fillna(0) กับข้อมูลที่ 0 มีความหมาย ใช้ NaN + คอลัมน์ flag
- แบ่ง train/test ตามเวลาเท่านั้น purge + embargo ตาม EMBARGO ใน config.yaml
- ทุกโมเดลต้องเทียบ baseline ถ้าไม่ชนะชัดเจน ให้แนะนำใช้ baseline
- macro ไม่เกิน 3 ตัว ใช้เป็นตัวกรองสภาวะตลาดเท่านั้น

---

## 2. ZEC เป็นข้อสอบปลายทาง ไม่ใช่แม่แบบ

ZECUSDT ช่วงที่ขึ้นจากประมาณ 55 ไป 650 (คาดว่าราว ก.ย.–พ.ย. 2025)
อยู่ใน holdout จึงห้ามใช้ออกแบบ feature, label หรือ template
- ขั้นแรกให้ยืนยันช่วงเวลาของ run นี้จาก **วันที่ของข้อมูลเท่านั้น**
  โดยห้ามโหลดราคาหลัง 2025-07-01 ถ้ายืนยันไม่ได้โดยไม่ดูราคา ให้รายงานผู้ใช้
- ข้อมูล ZEC ก่อน 2025-07-01 ใช้ได้ตามปกติเหมือนเหรียญอื่น
- คำถามหลักของงานนี้คือ: ระบบที่ไม่เคยเห็น ZEC รอบนี้ จับมันได้ก่อนวิ่งหรือไม่

---

## 3. ข้อมูล (ขั้น B1)

1. แก้ `data_downloader.py` ให้เก็บ Quote Volume, Number of Trades,
   Taker Buy Volume, Taker Buy Quote Volume แล้วดาวน์โหลดใหม่ทุกเหรียญ
   (ข้อมูลหลัง 2025-07-01 ให้บันทึกแยกไฟล์ และห้ามโหลดจนถึงขั้นที่ 10)
2. Funding_Rate: ช่วงที่ยังไม่มี futures เป็น NaN และเพิ่ม `has_futures` (0/1)
3. ตัด bar ที่ volume = 0 (ตามที่ common_eval ใช้อยู่)
4. `^IRX` เปลี่ยนชื่อเป็น US3M และดึง DGS2 จาก FRED (lag 1 วัน) เป็น US2Y
5. ใช้ BTCUSDT 4H แทน BTC-USD รายวันจาก yfinance
6. FearGreed เลื่อน D+1
7. บันทึก macro, FRED, FearGreed เป็น snapshot ใน `data/snapshots/` พร้อมวันที่ดึง
8. สร้าง `data/coin_metadata.csv`: symbol, วันลิสต์ spot/futures, วัน delist,
   sector tag จาก CoinGecko categories
9. ใช้ universe แบบไม่มี survivorship ตัวเดียวกับ LEGACY_FIXED (H1)

---

## 4. นิยาม event

ค่าจาก config.yaml: H_NEW = 540, MAE_LIMIT = 0.25 (เพิ่มใต้ `new:`),
UP_TARGET = 2.0, DOWN_TARGET = 0.5, EVENT_COOLDOWN = 180

- UP: จาก close ณ bar t ราคา high ถึง >= UP_TARGET เท่าภายใน H_NEW bar
  โดย low ไม่ต่ำกว่า close ณ t เกิน MAE_LIMIT ก่อนถึงเป้า
- DOWN: low ถึง <= DOWN_TARGET เท่าภายใน H_NEW bar
  โดย high ไม่สูงกว่า close ณ t เกิน MAE_LIMIT ก่อนถึงเป้า
- label 2 แบบ:
  (ก) bar-level สำหรับเทรนโมเดล
  (ข) event-level: 1 event ต่อ run, จุดเริ่ม = swing low (UP) / swing high (DOWN)
- label ช่วงท้ายของ development ที่ horizon ยื่นเข้า holdout ให้ตัดทิ้ง
  ห้ามใช้ราคาใน holdout มาทำ label
- รายงานจำนวน event แยกตามปีและ sector

---

## 5. Profile ของ event ในอดีต (แทนการใช้ ZEC เป็นแม่แบบ)

- รวม UP event ทั้งหมดในช่วง development
- เน้นกลุ่มที่คล้ายสถานการณ์ ZEC ก่อนวิ่ง: เหรียญเก่า ราคาต่ำกว่า ATH มาก
  อยู่ในฐานราคานาน (นิยามด้วย feature ไม่ใช่ด้วยการดูกราฟ ZEC)
- คำนวณ feature ที่ 1, 7, 30, 90, 180 วันก่อนจุดเริ่มของแต่ละ event
  พร้อม percentile เทียบทุกเหรียญทุก bar
- cluster pre-move profile (เช่น k-means หรือ HDBSCAN บน feature ที่ standardize)
  เพื่อหาว่ามี pattern ก่อนขึ้นกี่แบบ และแต่ละแบบเกิดบ่อยแค่ไหน
- สรุปภาษาไทยว่า event 2X ต่างจากเหรียญทั่วไปตรงไหนอย่างมีนัยสำคัญ

---

## 6. Features (ข้อมูล ณ bar t และก่อนหน้าเท่านั้น)

โครงสร้างราคา:
- % จาก ATH, % จาก low 1 ปี, base duration, ความกว้างกรอบ
- close / EMA300, close / EMA1200, slope ของ EMA
- breakout จาก high สูงสุดของ 180/540/1080 bar, จำนวน higher lows

ความผันผวน:
- ATR%, BB width และ percentile 1 ปีของเหรียญนั้น

Volume และ order flow:
- Quote Volume 20 วัน / 180 วัน, OBV slope
- taker buy ratio = Taker Buy Volume / Volume (เฉลี่ย 7/30 วัน และการเปลี่ยนแปลง)
- จำนวนเทรดต่อวัน, quote volume ต่อเทรด

Relative strength:
- ผลตอบแทน 7/30/90 วันเทียบ BTC และ rank ข้ามเหรียญ ณ เวลาเดียวกัน
- ผลตอบแทนเฉลี่ยของ sector (ไม่รวมเหรียญตัวเอง) และ rank ใน sector

Derivatives:
- funding เฉลี่ย 7/30 วัน, has_futures, จำนวนวันนับจากลิสต์ futures

สำหรับ DOWN เพิ่ม:
- overextension จาก EMA, RSI divergence, volume climax, อายุเหรียญ,
  funding บวกสูงต่อเนื่อง

Macro / regime (ไม่เกิน 3 ตัว):
- BTC trend (close / EMA1200 ของ BTCUSDT)
- net liquidity = FED_BAL − RRP (การเปลี่ยนแปลง 4–13 สัปดาห์)
- HY_SPREAD

---

## 7. Similarity search

- template = centroid ของแต่ละ cluster จากขั้นที่ 5 (ไม่ใช่ ZEC)
- (ก) distance บน feature vector ที่ standardize แล้ว
- (ข) DTW บน price path และ quote volume path ย้อนหลัง 90 วัน
- รายงาน top 20 match ในช่วง walk-forward และกี่ตัวที่เกิด UP event จริง

---

## 8. โมเดล (แยก UP และ DOWN)

Baseline ที่ต้องเทียบเสมอ:
- กฎแบบง่าย 3–5 เงื่อนไขจากข้อสรุปขั้นที่ 5
- logistic regression

XGBoost:
- max_depth 3–4, min_child_weight สูง, subsample 0.7, colsample_bytree 0.7
- scale_pos_weight ตามสัดส่วน class
- early stopping บน validation ที่แบ่งตามเวลา
- ลด sample ซ้อนกัน: 1 sample ต่อเหรียญต่อวัน หรือถ่วงน้ำหนักตาม uniqueness

Regime filter ฝั่ง UP:
- ทดสอบทั้งแบบมีและไม่มีตัวกรอง BTC trend และรายงานผลแยก
  (บทเรียนจากปี 2022)

การแบ่งข้อมูล:
- ใช้ fold ของ common_eval (walk-forward 2022-01 ถึง 2025-06, test รายครึ่งปี)
- ทุก fold ต้องมี event ใน test อย่างน้อย MIN_EVENTS_PER_FOLD ถ้าน้อยกว่าให้เตือน

การวัดผล:
- PR-AUC เทียบ base rate, precision@top-k (k = 5, 10, 20 เหรียญต่อสัปดาห์), lift
- SHAP: feature ไหนสำคัญ และมีเหตุผลทางตลาดหรือไม่

---

## 9. Backtest ผ่าน common_eval เท่านั้น

- ส่งสัญญาณในรูป (timestamp, symbol, score, side) เข้า common_eval
- ใช้ค่าจาก config.yaml ทั้งหมด: FEE_PER_SIDE, slippage ตามสภาพคล่อง,
  MIN_QUOTE_VOL_30D, MAX_POSITIONS, TRAILING_STOP, exit แบบ native และ unified
- UP: spot ไม่มี leverage
- DOWN: futures 1x และ **ต้องคิด funding** ตลอดช่วงที่ถือ
  (ถ้า common_eval ยังไม่รองรับ funding ให้หยุดและถามก่อนแก้)

Unit test ที่ต้องผ่านก่อนรัน backtest จริง:
- สูตรกำไร long และ short ด้วยตัวอย่างตัวเลขที่คำนวณมือได้
  (เช่น short เข้า 100 ออก 80 = +20% ก่อนค่าธรรมเนียม, เข้า 100 ออก 150 = −50%)
- funding: short ได้รับเมื่อ funding บวก จ่ายเมื่อ funding ลบ
- look-ahead test: เปลี่ยนข้อมูลหลัง bar t แบบสุ่ม feature ณ bar t ต้องไม่เปลี่ยน
- purge test: ไม่มี sample ใน train ที่ label horizon ทับกับช่วง test

รายงาน: จำนวนเทรด, win rate, avg win/loss, expectancy หลังค่าธรรมเนียม,
CAGR, max drawdown, Sharpe, Calmar, ผลรายปี, ผลเมื่อตัด 3 เทรดที่ดีที่สุดออก
ทดสอบความทนทาน: H_NEW, MAE_LIMIT, TRAILING_STOP ±20% และค่าธรรมเนียม 2 เท่า

---

## 10. Holdout และการทดสอบ ZEC (ต้องขออนุญาตผู้ใช้ก่อน)

เมื่อขั้น 3–9 เสร็จ ให้หยุด ส่งรายงาน walk-forward ให้ผู้ใช้
และ freeze โมเดลกับ config (commit + tag `new-frozen`)
เมื่อผู้ใช้อนุญาตแล้วจึง:
- รันทุกรุ่น (NEW_BASELINE, NEW_XGB, LEGACY_AS_IS, LEGACY_FIXED,
  RANDOM, BTC_HOLD, EQUAL_WEIGHT) บน holdout ครั้งเดียว ไม่ปรับอะไรเพิ่ม
- ZEC: แต่ละรุ่นให้สัญญาณ ZEC ก่อนวิ่งหรือไม่ ที่วันที่และราคาเท่าไหร่
  ถ้าได้สัญญาณ ให้แสดง SHAP top 5 ว่าเพราะอะไร
  ถ้าไม่ได้ ให้แสดงว่า feature ไหนของ ZEC ไม่ผ่านเกณฑ์
- รายงานจำนวนสัญญาณผิด (false positive) ในช่วงเดียวกันด้วย
  เพราะการจับ ZEC ได้ไม่มีความหมาย ถ้าระบบยิงสัญญาณเหรียญอื่นเป็นร้อยตัวพร้อมกัน

---

## 11. เกณฑ์ตัดสิน (เขียนไว้ก่อนดูผล)

NEW จะถือว่ามี edge ก็ต่อเมื่อครบทุกข้อ ทั้งใน walk-forward และ holdout:
- expectancy หลังค่าธรรมเนียมสูงกว่า LEGACY_FIXED และ bootstrap CI 95% ไม่คร่อม 0
- ดีกว่า RANDOM ที่ percentile 95
- ฝั่ง UP: Calmar หรือ Sharpe สูงกว่า BTC_HOLD
- ยังมีกำไรเมื่อตัด 3 เทรดที่ดีที่สุดออก และเมื่อค่าธรรมเนียม 2 เท่า
ถ้าไม่ผ่าน ให้สรุปตรงๆ ว่ายังไม่มี edge และอย่าเสนอปรับ parameter เพิ่มบน holdout

---

## 12. Skills (ทางเลือก)

ถ้าต้องการให้กฎถูกใช้ซ้ำในเซสชันต่อไป ให้สร้างใน `.claude/skills/`
(ทุก SKILL.md มี frontmatter `name` และ `description` และอ่านค่าจาก config.yaml):
- `quant-guardrails/`: เนื้อหาหัวข้อ 1 และ unit test หัวข้อ 9
  description: ใช้ทุกครั้งที่สร้าง/แก้ feature, label, model หรือ backtest
- `event-labeling/`: หัวข้อ 4
- `pattern-scanner/`: หัวข้อ 7 และ output ข้อ 4

---

## Output

1. `reports/new_walkforward_<วันที่>.md` ภาษาไทย: pattern ไหนใช้ได้จริง
   ไหนไม่ได้ พร้อมตัวเลข และข้อสรุปว่าควรใช้ XGBoost หรือกฎง่ายๆ
2. `reports/new_walkforward_<วันที่>.xlsx` แยก sheet: events_up, events_down,
   event_profiles, clusters, similar_matches, model_metrics, trade_log,
   yearly_results, robustness, unit_tests (header ชัดเจน, freeze แถวแรก, เปิด filter)
3. `reports/charts/`: profile ของแต่ละ cluster, equity และ drawdown เทียบ LEGACY และ benchmark
4. `scanner.py`: แสดงเหรียญที่เข้าเงื่อนไขตอนนี้ เรียงตามคะแนน พร้อม SHAP top 3
   (ใช้งานได้หลังผ่านขั้นที่ 10 เท่านั้น)
5. หลังขั้นที่ 10: `reports/new_holdout_<วันที่>.md` รวมผลทดสอบ ZEC
