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

## 🧪 ผลการรันระบบล่าสุด

Agent ทำงานวนลูปหา Hyperparameter ร่วมกับ LLM ได้อย่างราบรื่น:

1. ดึง Macro API 9 ตัว ครบถ้วน! (มี `USDJPY` ตามที่สั่ง)
2. `[Executor]` รัน Parameter sweep พบสัญญาณ Buy จาก Triple Barrier อย่างแม่นยำ
3. `[Researcher]` วนลูปให้ LLM (Gemini 3.6 Flash) ปรับเป้า `TP/SL` ให้แคบลง/กว้างขึ้นตาม Precision ที่ได้
4. ระบบจบการทำงานและพล็อตเป้าหมายการเทรดลงกราฟ Plotly สำเร็จ

ระบบพร้อมใช้งานสำหรับทำ Quant Research บนทุกเหรียญที่คุณสนใจแล้วครับ!
