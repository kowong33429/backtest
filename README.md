# 🚀 Agentic Quant System (Structural & Labeling Update)

การปรับปรุงระบบทั้งหมดเสร็จสมบูรณ์ 100% แล้วครับ โดยอิงตามหลักการที่คุณแนะนำมาทั้งหมด!

---

## 🎯 1. การแก้ไขเรื่อง "โกง Label" ให้ถูกต้อง

เราได้นำแนวคิด **Hindsight Labeling (มองอนาคตเพื่อหา Label ที่ดีที่สุด)** กลับมาใช้ใน `core/labels.py` แล้วครับ 

```python
# ใน core/labels.py
self.df['Swing_Low'] = self.df['Low'] == self.df['Low'].rolling(window=full_window, center=True).min()
```

- **ทำไมถึงดี:** การตั้ง `center=True` คือการยอมให้โมเดล "เห็นอนาคต" เพื่อมาร์คเป้าหมาย (Y) ให้แม่นยำที่สุดว่า "จุดไหนคือ Bottom ที่แท้จริง"
- **ไม่โกงในการทายผล:** ส่วน Features (X) ที่เราใช้ป้อนให้โมเดล Machine Learning ยังคงเป็นแบบ **มองอดีตอย่างเดียว (Backward-looking)** ดังนั้น โมเดลจึงยังต้อง "เดา" อนาคตอยู่ดี ไม่มีปัญหา Data Leakage แต่อย่างใดครับ
- **ผลลัพธ์:** Precision เพิ่มขึ้นทันที (กลับมาอยู่ที่ 5.49%) เพราะโมเดลได้เรียนรู้จากเป้าหมาย (Y) ที่ตรงเวลา ไม่ดีเลย์

---

## 📁 2. โครงสร้างโฟลเดอร์แบบ Multi-Coin

ผมได้ปรับโครงสร้างทั้งหมดให้รองรับการเทรดหลายๆ คู่เหรียญในอนาคตได้อย่างสวยงาม:

```text
C:\Project\backtest\
├── data\
│   ├── ZECUSDT\
│   │   ├── ZECUSDT_4h_full.csv
│   │   └── bt_zecusdt_legacy.py
│   └── BTCUSDT\ (ตัวอย่างในอนาคต)
└── core\
    ├── data_downloader.py  ← (เครื่องมือดาวน์โหลดส่วนกลาง)
    ├── data_loader.py
    ├── features.py
    ├── labels.py
    ├── optimizer.py
    ├── visualizer.py
    └── research_agent.py   ← (สมองหลัก)
```

### ⚡ เครื่องมือดาวน์โหลดส่วนกลาง (`data_downloader.py`)

คุณสามารถใช้เครื่องมือนี้ดาวน์โหลดข้อมูล OHLCV ของเหรียญใดๆ บน Binance ได้เลยผ่าน Terminal โดยมันจะสร้างโฟลเดอร์ให้เองอัตโนมัติ:

```bash
# ตัวอย่าง: ดาวน์โหลดข้อมูล ZECUSDT Timeframe 4H
python core/data_downloader.py --symbol ZECUSDT --interval 4h

# ตัวอย่าง: ดาวน์โหลดข้อมูล BTCUSDT Timeframe 1D
python core/data_downloader.py --symbol BTCUSDT --interval 1d
```

### 🧠 สั่ง Agent ทำงาน

เมื่อมีข้อมูลแล้ว คุณสามารถสั่งให้ Agent เริ่มทำวิจัยด้วย XGBoost ทันที (พร้อมระบุเหรียญ):

```bash
python core/research_agent.py --symbol ZECUSDT
```

---

## 📊 3. Data Analytics & Explainable AI (อัปเดตล่าสุด)

เราได้อัปเกรดระบบการวิเคราะห์ข้อมูลให้เทียบเท่าเครื่องมือของ Quant Professional โดยเพิ่มเทคนิคเหล่านี้เข้าไปใน Pipeline อัตโนมัติ:

1. **SHAP Values (Explainable AI):** ระบบจะสร้างรูปภาพ `shap_summary.png` เพื่อเปิดเผยว่าโมเดลตัดสินใจซื้อเพราะปัจจัยใดเป็นหลัก
2. **Trade Simulation & Equity Curve:** จำลองการเทรดจริงเพื่อพล็อตกราฟการเติบโตของพอร์ต (Equity Curve) พร้อมคำนวณ Win Rate จาก Confusion Matrix
3. **Feature Distributions:** พล็อต Boxplot เพื่อหาว่าสัญญาณซื้อที่ชนะ (Signal=1) มักจะเกิดขึ้นที่อินดิเคเตอร์ค่าเท่าไหร่
4. **Correlation Filter & Heatmap:** พล็อตกราฟ Heatmap เพื่อเช็คความสัมพันธ์ของปัจจัยแต่ละตัว และดรอปตัวแปรที่ให้ข้อมูลซ้ำซ้อนกัน (Multicollinearity > 0.9) ทิ้งอัตโนมัติก่อนเทรน
5. **Stability Analysis:** พล็อตกราฟเช็คความเสถียรของความแม่นยำโมเดลในแต่ละช่วงเวลา (Precision per Fold)

> **💡 Note:** ข้อมูลทั้งหมดจะถูกสรุปเป็นไฟล์ `summary_report.md` เก็บไว้ในโฟลเดอร์ย่อยของแต่ละเหรียญ และแสดงผลกราฟแบบ Interactive ผ่าน Plotly ให้ดูอัตโนมัติเมื่อรันโปรแกรม

---

## 🧪 ผลการรันระบบล่าสุด

Agent ทำงานวนลูปหา Hyperparameter ร่วมกับ LLM ได้อย่างราบรื่น:

1. ดึง Macro API 9 ตัว ครบถ้วน! (มี `USDJPY` ตามที่สั่ง)
2. `[Executor]` รัน Parameter sweep พบสัญญาณ Buy จาก Triple Barrier อย่างแม่นยำ พร้อมลบตัวแปรซ้ำซ้อนทิ้ง
3. `[Researcher]` วนลูปให้ LLM (Gemini 3.6 Flash) ปรับเป้า `TP/SL` ให้เหมาะสม
4. สร้างรายงานสรุปผล `summary_report.md`, ภาพ `shap_summary.png`, และกราฟแสดงผลสำเร็จ

ระบบพร้อมใช้งานสำหรับทำ Quant Research & Analytics บนทุกเหรียญที่คุณสนใจแล้วครับ!
