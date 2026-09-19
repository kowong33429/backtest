# 📊 Quant Research Report: ZECUSDT (Long/Short Dual Model)

## 1. ⚙️ การตั้งค่าที่ให้ผลลัพธ์ดีที่สุด (Best Configuration)
- **Take Profit (TP):** 12.0%
- **Stop Loss (SL):** 0.8%
- **ความแม่นยำรวม (Precision):** 2.84%

> **💡 ความหมาย:** เมื่อโมเดลสั่งเทรด ราคาจะมีโอกาสวิ่งไปชน TP ก่อน SL ด้วยความแม่นยำประมาณ 2.84%

## 2. 📉 ผลการจำลองเทรด (Trade Simulation & Confusion Matrix)
- **จำนวนไม้ทั้งหมดที่โมเดลบอกให้เทรด (Total Trades):** 1084 (Long: 720, Short: 364)
- **ชนะ (Wins):** 24
- **แพ้ (Losses):** 1060
- **Win Rate รวม:** 2.21%

## 3. 🧠 SHAP Values (Explainable AI)
วิเคราะห์ว่า Feature แต่ละตัวส่งผลอย่างไรต่อการตัดสินใจของโมเดล (จุดสีแดง = ค่าสูง, จุดสีน้ำเงิน = ค่าต่ำ)

### ฝั่ง Long
![SHAP Long](./shap_long.png)

### ฝั่ง Short
![SHAP Short](./shap_short.png)

## 4. 🔍 ปัจจัยที่มีผลต่อการตัดสินใจมากที่สุด (Top Feature Importances)
1. **RSI_14** (Score: 0.1370)
2. **FED_BAL** (Score: 0.0567)
3. **Pos_In_7D** (Score: 0.0452)
4. **RRP_PctChg** (Score: 0.0452)
5. **MACD_Hist** (Score: 0.0433)
6. **BB_Width** (Score: 0.0428)
7. **Pos_In_24H** (Score: 0.0370)
8. **RRP** (Score: 0.0359)
9. **OFI_Proxy** (Score: 0.0340)
10. **FearGreed** (Score: 0.0318)
