# 📊 Quant Research Report: ZECUSDT (Long/Short Dual Model)

## 1. ⚙️ การตั้งค่าที่ให้ผลลัพธ์ดีที่สุด (Best Configuration)
- **Take Profit (TP):** 35.0%
- **Stop Loss (SL):** 25.0%
- **ความแม่นยำรวม (Precision):** 5.49%

> **💡 ความหมาย:** เมื่อโมเดลสั่งเทรด ราคาจะมีโอกาสวิ่งไปชน TP ก่อน SL ด้วยความแม่นยำประมาณ 5.49%

## 2. 📉 ผลการจำลองเทรด (Trade Simulation & Confusion Matrix)
- **จำนวนไม้ทั้งหมดที่โมเดลบอกให้เทรด (Total Trades):** 2394 (Long: 1368, Short: 1026)
- **ชนะ (Wins):** 101
- **แพ้ (Losses):** 2293
- **Win Rate รวม:** 4.22%

## 3. 🧠 SHAP Values (Explainable AI)
วิเคราะห์ว่า Feature แต่ละตัวส่งผลอย่างไรต่อการตัดสินใจของโมเดล (จุดสีแดง = ค่าสูง, จุดสีน้ำเงิน = ค่าต่ำ)

### ฝั่ง Long
![SHAP Long](./shap_long.png)

### ฝั่ง Short
![SHAP Short](./shap_short.png)

## 4. 🔍 ปัจจัยที่มีผลต่อการตัดสินใจมากที่สุด (Top Feature Importances)
1. **RSI_14** (Score: 0.1101)
2. **Pos_In_24H** (Score: 0.0965)
3. **Pos_In_7D** (Score: 0.0407)
4. **MACD_Hist** (Score: 0.0390)
5. **RRP_PctChg** (Score: 0.0379)
6. **Return_1** (Score: 0.0373)
7. **BTC_Close** (Score: 0.0356)
8. **OFI_Proxy** (Score: 0.0343)
9. **SMA_50** (Score: 0.0342)
10. **ATR_14** (Score: 0.0313)
