import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

SCR = r"C:/Users/User/AppData/Local/Temp/claude/c--Project-backtest/c8133345-49e8-4a78-8722-bdcdd17b9945/scratchpad"

F = [  # id, sev, title, where, impact, fix
("H1", "สูง", "Survivorship / hindsight: basket 150 เหรียญถูกเลือกจากเหรียญที่ 'เคย rally >200% ใน 14-120 วัน' มาแล้วทุกตัว (150/150)",
 "core/trend_scanner.py:443 (+docstring บรรทัด 2-3); core/dataset.py:82-104 (discover_coins อ่าน basket json)",
 "m1 thr0.6 expectancy/เทรด: basket +8.35% (PF 1.53, 2,154 เทรด) vs universe PIT top-120 -2.80% (PF 0.85, 2,502 เทรด). ซื้อทุก bar ไม่ใช้โมเดล: basket +5.09% vs universe -1.93%. rotation b7 ช่วงเดียวกัน (2021-10-28→): basket +120.1% CAGR vs universe -8.7%; BTC hold +7.0%",
 "ใช้ universe point-in-time (data_all + volume_rank) เป็นค่าเริ่มต้นเสมอ ห้ามเลือกเหรียญด้วยผลลัพธ์ในอนาคต"),
("H2", "สูง", "'ทักษะ' ของโมเดลมาจากฟีเจอร์ macro ช้าๆ (CPI/M2/yield curve/FEDFUNDS/HY) ที่ทำหน้าที่เป็นตัวบอกช่วงเวลา ไม่ใช่การเลือกเหรียญ",
 "core/dataset.py:157-166 (merge macro/F&G หลังคัดฟีเจอร์); core/train_entry_model.py:134",
 "gain 64-66% มาจาก 18 ฟีเจอร์ non-TA (CPI_YoY เดียว 15-18%). Ablation: TA-only PR-AUC = 0.141 (=base rate 0.141) บน basket, 0.074 < base 0.081 บน universe; full 0.219 / 0.077. คะแนน 'ค่าเฉลี่ยรายวันของทุกเหรียญ' อย่างเดียวได้ PR-AUC 0.219 = เท่าโมเดลเต็ม. cross-sectional PR-AUC 0.156 (base 0.141). trade ที่ความถี่เท่ากัน: เต็ม +7.64%, TA-only +2.55%, สุ่ม +4.4~4.9%",
 "ตัด macro ที่เปลี่ยนช้า (CPI/M2/FEDFUNDS/HY) หรือใช้เฉพาะ rank ข้ามเหรียญ (demean ต่อวัน); วัดผลด้วย cross-sectional PR-AUC/IC"),
("H3", "สูง", "สูตรกำไรฝั่ง SHORT ใช้ reciprocal (entry/exit-1) ซึ่งไม่ใช่กำไรของ short จริง",
 "core/entry_backtest_short.py:21-23,100; core/labeler_short.py:13-25",
 "short_v2 thr0.5 (1,240 เทรด): log เดิม expectancy +22.54%, PF 2.91, +$27,947 -> คำนวณใหม่แบบ short จริง (1-exit/entry): +0.62%, PF 1.03, +$763. SL ขาดทุนจริง -66.7% (log บันทึก -40%). รายปีจริง: 2020 -19%, 2021 -6%, 2022 -12%, 2023 -8%, 2024 -3%, 2025 +15%, 2026 -2%",
 "คำนวณ P&L short = 1 - exit/entry; ปรับ barrier ของ label ให้สอดคล้อง (TP ราคาลง 50% = +50%); เพิ่ม funding/borrow"),
("M1", "กลาง", "ไม่มีเพดานเงินทุน/ความสัมพันธ์ระหว่างเทรด: ตัวเลข 2,154 เทรดไม่ใช่การทดลองอิสระ",
 "core/entry_backtest.py:58-98 (cooldown ต่อเหรียญเท่านั้น)",
 "ถือพร้อมกันสูงสุด 134 ไม้ เฉลี่ย 51; วันเดียวเข้า 73 ไม้ (2023-07-26) และ 72 ไม้ (2025-12-16). month-block bootstrap 95% CI ของ expectancy: basket [+1.1%, +15.9%], universe [-10.1%, +5.0%]. positive 146,413 bar = 3,418 episodes (universe 71,884 bar = 2,290 episodes)",
 "รายงานผลระดับพอร์ตที่จำกัดทุน (portfolio_backtest) และ CI แบบ block bootstrap เป็นค่าหลัก"),
("M2", "กลาง", "เลือก threshold/hyperparameter/preset บน OOF ชุดเดียวกับที่รายงานผล ไม่มี holdout ที่ไม่เคยแตะ",
 "core/train_entry_model.py:225-228; README (>=11 รุ่นโมเดลใน data/model/v2, sweep thr 0.5-0.9, exit sweeps, rotation 15 รอบ ~20 พารามิเตอร์)",
 "วัดตรงไม่ได้. ความไวต่อพารามิเตอร์ (basket, b7 CAGR 120.1%): lead-near-high 0.20/0.30 = 98.2/105.0; rebal 6/8 = 82.0/82.7; lookback 48/72 = 99.1/117.8; vol-target 0.56/0.84 = 107.4/123.9; k 8/12 = 130.4/119.4; core-btc-sma 80/120 = 121.0/118.6. บน universe ทุกค่าติดลบ (-4.2 ถึง -11.5)",
 "ล็อก holdout (เช่น 2025-07-01→) ใช้ครั้งเดียว; นับจำนวน trial เพื่อทำ deflated Sharpe"),
("M3", "กลาง", "ต้นทุนแบบคงที่ ไม่ผูกกับสภาพคล่อง และไม่คิด funding ของ perp",
 "core/entry_backtest.py:144-145; core/rotation_backtest.py:314; core/exits.py:96",
 "rotation turnover 38-47 เท่า/ปี. ต้นทุน 2 เท่า (0.30%/ข้าง): basket b7 +120.1% -> +107.1%; 4 เท่า (0.60%): +83.3%; universe b7 -8.7% -> -14.8% (DD -76% -> -80%). m1 universe expectancy -2.80% -> -3.10%. funding (perp, long): btc-trend CAGR 30.3% -> 24.8% (-5.5pt); b7 universe -0.3pt",
 "slippage เป็นฟังก์ชันของ quote volume/ขนาดไม้; บวก funding จริงเมื่อเทรด perp"),
("M4", "กลาง", "ฟีเจอร์ NaN ถูกแทนด้วย 0 ทำให้ 0 กลายเป็นธงช่วงเวลา",
 "core/train_entry_model.py:134; core/predict_entry.py:61; core/rotation_paper.py:122; core/data_loader.py:105,205",
 "HY_SPREAD NaN 35.2% ของ panel m1 (FRED ให้ BAMLH0A0HYM2 ตั้งแต่ 2023-10-17 เท่านั้น); Funding_Rate = 0 ใน 13.6% ของแถว (basket เฉลี่ย 14%, สูงสุด 75% ต่อเหรียญ = ช่วงก่อนมี perp). ablation: ตัด HY+Funding แล้ว PR-AUC 0.221 vs 0.219 (ไม่เปลี่ยน เพราะ macro ตัวอื่นแทนที่)",
 "ปล่อย NaN ให้ XGBoost จัดการเอง (หรือ drop ฟีเจอร์), อย่า fillna(0) กับ macro/funding"),
("M5", "กลาง", "Label ไม่ตรงกับวิธีเทรดจริง",
 "core/labeler.py:91-94 vs core/entry_backtest.py:61; core/exits.py:49",
 "label เข้า Open[i] และสแกนตั้งแต่ bar i+1 (ข้าม High/Low ของ bar i) ส่วน backtest เข้า Open[i+1] (ช้ากว่า 1 bar ซึ่งไม่ใช่ leak). label ไม่คิดค่าธรรมเนียม; 53% ของเทรดจบด้วย TIME stop (avg +1.6% basket / +0.9% universe) ซึ่ง label ถือเป็น 0 เท่ากับ SL (-40%). win rate 43% จึงถูกพยุงด้วย TIME exit. avg win +55.7% / avg loss -27.7%",
 "นิยาม label ด้วยผลตอบแทนสุทธิหลังต้นทุนที่ exit rule เดียวกับตอนเทรด หรือใช้ regression/ranking"),
("M6", "กลาง", "ความน่าจะเป็นไม่ได้ calibrate (scale_pos_weight 6-20) ทำให้ threshold 0.6/0.9 ไม่มีความหมายเชิงความน่าจะเป็น",
 "core/train_entry_model.py:152-157",
 "basket: prob>=0.6 -> precision จริง 20.1%, >=0.9 -> 38.2% (base 14.1%). universe: precision 7.4-7.9% ทุก threshold (<base 8.1%, ยิ่ง prob สูงยิ่งแย่ลง)",
 "calibrate (isotonic) บน fold ก่อนเลือก threshold หรือใช้ top-k ต่อวัน"),
("M7", "กลาง", "คุณภาพข้อมูล universe: bar แบน/ปริมาณ 0 และจำนวนเหรียญที่ delist ต่ำผิดปกติ",
 "core/universe_download.py; core/rotation_backtest.py:254-255,403",
 "data_all: 162/703 เหรียญมี bar แบน volume=0 รวม 341,385 bar (8.8%; BTCST, SC, FTT, RAY ฯลฯ แบนยาว 2-5 ปี); มีเพียง 31/703 (4.4%) ที่ข้อมูลจบก่อน 2026-10-01 -> อาจยังตกหล่น perp ที่ delist (ยังไม่ได้ตรวจกับรายการ delist ของ Binance). breadth นับเหรียญ delist เป็นต่ำกว่า SMA (README ระบุว่าเปิดอยู่)",
 "ตัด bar volume=0 ออกจาก panel/eligibility; ตรวจรายการ delist ของ Binance เทียบกับ universe_symbols.json"),
("M8", "กลาง", "ผลขึ้นกับสภาวะตลาด: rotation แพ้หนักเมื่อ BTC ต่ำกว่า SMA200",
 "core/rotation_backtest.py:331-354",
 "universe b7: วันที่ BTC<SMA200 (902 วัน) รวม -66.8%; BTC>SMA200 (909 วัน) +87.8%. m1 basket รายปี: 2022 -1.2%, 2025 -6.2%; universe: 2021 -18.6%, 2022 -7.0%, 2025 -6.0%, 2026 -4.2%. btc-trend (ไม่ใช้โมเดล) +30.3%/DD -24.6% ดีกว่าทุก preset บน universe",
 "ใช้ btc-trend เป็น benchmark ที่ต้องชนะ"),
("L1", "ต่ำ", "^IRX ถูกใช้เป็น US2Y (จริงคือ T-bill 13 สัปดาห์) ทำให้ 'YieldCurve_10Y_2Y' เป็น 10Y-3M",
 "core/entry_finder.py:254,335; core/data_loader.py:50",
 "ยืนยันแล้ว ไม่ใช่ leak แต่ชื่อ/ความหมายผิด; ฟีเจอร์นี้อยู่อันดับ 4-6 ของ gain (4.8% m1, 6.7% m1_univ, 9.6% short_v2)",
 "ใช้ FRED DGS2 หรือเปลี่ยนชื่อเป็น 10Y-3M"),
("L2", "ต่ำ", "Fear&Greed ไม่ถูกเลื่อน D+1 และหลุดจาก panel เงียบๆ ถ้า API timeout; FRED ใช้ค่าที่ revise แล้ว (ไม่ใช่ vintage แรก)",
 "core/dataset.py:157-159; core/entry_finder.py:387-395",
 "gain ของ FearGreed 0.4-1.7% ผลน้อย; lag FRED ตรวจแล้วเหมาะสม (VIX ปิด 2024-03-08 = 14.74 ปรากฏที่ 2024-03-09)",
 "เลื่อน F&G +1 วัน; ใช้ ALFRED vintage; assert จำนวนฟีเจอร์ตอน build"),
("L3", "ต่ำ", "ไม่มี leverage/การ liquidate ใน backtest ปัจจุบัน; backtester.py เก่ามี max_position=3.0 แต่ไม่มีโมเดล liquidation",
 "core/backtester.py:54",
 "entry/rotation/portfolio เป็น long ไม่ใช้ leverage -> ไม่มีความเสี่ยงถูก liquidate ในตัวเลขที่รายงาน (backtester.py ไม่ได้ใช้กับผลลัพธ์หลัก และอยู่นอกขอบเขตที่ตรวจ)",
 "หากจะใช้ leverage ต้องเพิ่มโมเดล liquidation/margin"),
("L4", "ต่ำ", "SL/TP เติมที่ราคาระดับเป๊ะ (barrier) ไม่คิด gap",
 "core/exits.py:62-66; core/entry_backtest.py:76-81",
 "ตรวจ SL 611 (long) + 315 (short): ไม่มีไม้ที่ Open ของ bar ออกทะลุระดับ stop -> ผลกระทบวัดได้ 0",
 "ใช้ min(Open, stop) เพื่อความปลอดภัย"),
("L5", "ต่ำ", "fold แรกฝึกบนข้อมูลน้อย; window ฟีเจอร์ก้าวทีละ 10 (ซ้ำซ้อน) ตาม user override",
 "core/train_entry_model.py:53; core/features.py:13-18",
 "fold แรกฝึกราว 11k แถว (README); 108 ฟีเจอร์ต่อ ~3.4k episodes ที่ซ้อนทับสูง",
 "ใช้ corr-filter/ลดจำนวนฟีเจอร์"),
("OK", "ผ่าน", "ตรวจแล้วไม่พบปัญหา: N-1 ของฟีเจอร์ TA, ไทม์มิ่ง rotation, purge/embargo, ค่า lag macro",
 "core/features.py:216; core/rotation_backtest.py:447,452; core/train_entry_model.py:39-68; core/entry_finder.py:261-266,296",
 "ทดสอบ perturbation 3 ครั้ง (แก้ OHLCV ของ bar i แล้วฟีเจอร์ของ bar i ไม่เปลี่ยน, bar i+1 เปลี่ยน, bar ก่อนหน้าไม่เปลี่ยน); weights ตัดสินที่ close d เทรดที่ open d+1; vol target shift(2); purge ตาม Label_EndDate + embargo 3 วัน; funding shift(1)",
 "-"),
]

wb = Workbook()
wb.remove(wb.active)


def sheet(name, header, rows, widths=None):
    ws = wb.create_sheet(name)
    ws.append(header)
    for r in rows:
        ws.append(list(r))
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F3864")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for i, h in enumerate(header, 1):
        w = (widths or {}).get(i)
        if w is None:
            w = max(12, min(60, max([len(str(h))] + [len(str(r[i - 1])) for r in rows]) + 2))
        ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    return ws


sheet("audit_legacy", ["ID", "ระดับ", "ปัญหา", "ไฟล์:บรรทัด", "ผลกระทบ (ตัวเลข)", "วิธีแก้ที่แนะนำ"], F,
      {1: 6, 2: 8, 3: 50, 4: 42, 5: 80, 6: 42})

sheet("legacy_summary", ["หัวข้อ", "รายละเอียด"], [
    ("เป้าหมาย", "entry model: P(rally +100% ภายใน 60 วัน ก่อนโดน -40%) ต่อ 4H bar (long, m1); short mirror (ราคา 'ลด' ถึง +100% reciprocal ภายใน 30 วัน ก่อน +66.7%); rotation: ถือ top-K ตาม OOF prob + leader/breakout sleeve + BTC trend fallback"),
    ("Label", "triple barrier TP +100% / SL -40% / horizon 60d (m1, 360 bar) หรือ 30d (original, short_v2); เข้า Open[i] สแกน High/Low จาก bar i+1; same-bar -> SL; sample weight = AFML average uniqueness"),
    ("ฟีเจอร์ (108)", "TA scale-free 90: RSI x10 (+lag RSI_10/20), ADX x10, BB_Width x9 (+lag), Dist_SMA x11, Return x8 windows (+Return_1/2/lag), Dist_High/Low x4+4, Volume_Surge x10, Pos_In x8, Dist_VWAP x8, SMA_Cross, ATR_Ratio; non-TA 18: Funding_Rate, FearGreed, SP500/DXY/GOLD/OIL ret20d/50d (8), YieldCurve_10Y_2Y(^TNX-^IRX), M2_YoY/MoM, CPI_YoY/MoM, FEDFUNDS, VIX, HY_SPREAD"),
    ("โมเดล", "XGBoost n_est 300, lr 0.03, depth 4, subsample 0.8, colsample 0.8, scale_pos_weight=neg/pos (m1, m1_univ); short_v2: 600 trees, depth 5, min_child_weight 50, colsample 0.5; Logistic baseline"),
    ("การแบ่งข้อมูล", "purged expanding walk-forward: timeline แบ่ง n+2 ช่วงเท่ากัน, fold 1..n เป็น test (n=6 long, 5 short), purge ตาม Label_EndDate < test_start, embargo 3 วัน; final model refit ทั้งหมดแต่ backtest ใช้ OOF"),
    ("กฎเข้า/ออก (entry)", "signal OOF prob >= 0.60 (baseline); เข้า Open ของ bar ถัดจาก signal row (s+1); TP +100% / SL -40% / time stop 60d ที่ Close; 1 ไม้ต่อเหรียญ; $100 ต่อไม้ ไม่จำกัดทุน"),
    ("กฎ rotation", "รายวัน, top-10 ตาม prob (smooth 7d), rebal 7d, hyst 2.0, breadth>0.5, leader sleeve 3 + young sleeve 2, BTC fallback (>SMA50&200), vol target 70%; ตัดสินใจที่ close d เทรดที่ open d+1"),
    ("ต้นทุน/leverage", "fee 0.10% + slippage 0.05% ต่อข้าง (rotation 0.15%/ข้างรวม); ไม่มี funding; ไม่มี leverage/liquidation; long-only spot-style (short backtest ใช้ reciprocal)"),
    ("Universe/ช่วงเวลา", "basket: 150 เหรียญ (144 ใน panel, ข้อมูล 2018-2026-10, OOF 2019-04→2026-10); universe: data_all 703 USDT-M perp (2020-01-01→2026-10-13), OOF m1_univ 347 เหรียญที่เคยอยู่ใน PIT top-120, 2020-12-31→2026-10-12"),
    ("ผลที่บันทึกไว้ (README)", "m1 thr0.6: 2,154 เทรด win 43.2% exp +8.35% PF 1.53 +$17,976; short_v2 thr0.5: 1,240 เทรด exp +22.5% PF 2.91; rotation b7 basket +198.8% CAGR DD -39.5%; universe b7 -7.9%; BTC hold +7.5%"),
    ("ผลที่ reproduce ได้ (รัน 2026-10-18)", "m1 basket 2154 เทรด/+8.35%/PF 1.53 ตรงกับ README; b7 basket +198.8%/-39.5% ตรง; b7 universe -8.7%/-76.1% (README -7.9% ข้อมูลใหม่กว่า)"),
], {1: 26, 2: 140})

sheet("measurements", ["การวัด", "ตัวเลข"], [
    ("m1 basket thr0.6 (2020-2026)", "2,154 เทรด win 43.2% exp +8.35% PF 1.53 | TP 409 / SL 611 / TIME 1,134 | avg win +55.7% avg loss -27.7%"),
    ("m1 basket เฉพาะ 2021+", "2,049 เทรด exp +6.82%"),
    ("m1 universe PIT top-120 thr0.6", "2,502 เทรด win 35.9% exp -2.80% PF 0.85 | thr0.5: 2,873 เทรด exp -2.06% PF 0.88"),
    ("ซื้อทุก bar (ไม่ใช้โมเดล) basket / universe", "+5.09% PF 1.31 / -1.93% PF 0.89"),
    ("สุ่มความถี่เท่า thr0.6 (3 seeds) basket / universe", "+4.44~+4.91% / -1.90~-2.28%"),
    ("month-block bootstrap 95% CI expectancy basket / universe", "[+1.08%, +15.92%] / [-10.05%, +5.00%]"),
    ("top-3 trade ออก (m1 basket)", "exp 8.35% -> 8.22% (ไม่พึ่ง outlier); ตัด top-30: 7.06%"),
    ("ต้นทุน x2/x3 (m1 basket)", "exp 8.05% / 7.75% (ไม่ไวเพราะไม้ใหญ่); universe x2: -3.10%"),
    ("OOF PR-AUC m1 basket (base 0.141)", "0.224 (reported); LR 0.147; date-mean-only 0.223; cross-sectional 0.160; ROC 0.584"),
    ("OOF PR-AUC m1_univ (base 0.081)", "0.077; date-mean-only 0.075; cross-sectional 0.092 (xsec ROC 0.531)"),
    ("PR-AUC ต่อปี basket (base)", "2020 0.439(0.369) 2021 0.438(0.279) 2022 0.099(0.100) 2023 0.160(0.123) 2024 0.163(0.198) 2025 0.115(0.067) 2026 0.145(0.135)"),
    ("short_v2 recompute (1,240 เทรด)", "log: exp +22.54% PF 2.91 $27,947 | short จริง: exp +0.62% PF 1.03 $763 | ตัด top3: +0.50%"),
    ("rotation b7 basket 2020-05-12→2026-10-08", "+198.8% CAGR, DD -39.5%, Sharpe 2.29, x1,110; ปี: 2020 +318 2021 +489 2022 -4 2023 +320 2024 +207 2025 +26 2026 +190"),
    ("rotation b7 basket 2021-10-28→", "+120.1% DD -39.5% Sharpe 1.80"),
    ("rotation b7 universe top-120", "-8.7% DD -76.1% Sharpe 0.05; ปี: 2021 -23 2022 -49 2023 +250 2024 +7 2025 -31 2026 -38; alt spells 894, >+100%: 5"),
    ("core-only universe (ไม่มี sleeve/fallback)", "-14.1% DD -72.8%"),
    ("btc-trend universe", "+30.3% DD -24.6% Sharpe 1.07 (funding perp -> +24.8%)"),
    ("BTC buy&hold 2021-10-28→", "+7.0% CAGR"),
    ("ความไว gap", "SL 611+315 ไม้: gap เกิน stop = 0"),
], {1: 48, 2: 130})

ab = [("basket",) + tuple(r) for r in pd.read_csv(SCR + '/abl_basket.csv').values.tolist()] + \
     [("universe",) + tuple(r) for r in pd.read_csv(SCR + '/abl_univ.csv').values.tolist()]
sheet("ablation", ["data", "variant", "n_feat", "base_rate", "pr_auc", "roc", "date_mean_only_pr", "xsec_pr", "xsec_roc"], ab)

sheet("trades_by_year", ["year", "m1_basket_n", "m1_basket_exp%", "m1_univ_exp%", "short_logged_exp%", "short_real_exp%", "short_n"], [
    (2020, 105, 38.1, "", 11.2, -19.1, 6), (2021, 297, 20.5, -18.6, 21.4, -6.2, 198), (2022, 291, -1.2, -7.0, 15.1, -11.6, 32),
    (2023, 246, 15.5, 5.9, 8.9, -7.6, 143), (2024, 466, 8.5, 3.6, 22.0, -3.2, 173), (2025, 379, -6.2, -6.0, 38.6, 15.3, 314),
    (2026, 370, 7.6, -4.2, 15.9, -1.8, 374)])

sheet("sensitivity", ["ชุด", "พารามิเตอร์", "CAGR %", "MaxDD %"], [
    ("basket b7 (2021-10-28→)", "base", 120.1, -39.5), ("basket", "near-high 0.20", 98.2, -43.8), ("basket", "near-high 0.30", 105.0, -44.2),
    ("basket", "core-btc-sma 80", 121.0, -41.2), ("basket", "core-btc-sma 120", 118.6, -39.5), ("basket", "vol-target 0.56", 107.4, -36.6),
    ("basket", "vol-target 0.84", 123.9, -42.8), ("basket", "k=8", 130.4, -39.7), ("basket", "k=12", 119.4, -35.5),
    ("basket", "rebal=6", 82.0, -52.2), ("basket", "rebal=8", 82.7, -42.0), ("basket", "lead-lookback 48", 99.1, -47.2),
    ("basket", "lead-lookback 72", 117.8, -39.9), ("basket", "cost 0.30%/ข้าง", 107.1, -40.9), ("basket", "cost 0.60%/ข้าง", 83.3, -44.0),
    ("universe b7", "base", -8.7, -76.1), ("universe", "near-high 0.20", -4.2, -75.1), ("universe", "near-high 0.30", -11.5, -79.2),
    ("universe", "cost 0.30%/ข้าง", -14.8, -80.2), ("universe btc-trend", "base", 30.3, -24.6), ("universe core-only", "base", -14.1, -72.8)])

wb.save('C:/Project/backtest/reports/legacy_audit_20261018.xlsx')
print('saved')
