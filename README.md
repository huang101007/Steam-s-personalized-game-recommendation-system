# Steam Game Recommendation System

## 專案簡介
本專案基於Python開發，針對UCSD實驗室提供的Steam平台真實玩家評論資料集（共59,305筆），建構一套個人化遊戲推薦系統。專案核心旨在解決真實數據中高達 88% 為正向評價的「資料不平衡」問題，並結合自然語言處理與潛在因子模型（Latent Factor Model），將機器學習理論轉化為解決實務數據問題的完整方案。

## 核心功能與技術實作

* **資料前處理與統計分析**
  * 導入約六萬筆 Steam 遊戲評論數據，採用 80/20 比例分割訓練集與測試集，確保模型評估之公正性。
  * 處理JSON格式的嵌套資料，萃取 user_id、item_id、recommend 及 review 等關鍵特徵。

* **分類任務：處理資料不平衡 (Classification & Imbalanced Data)**
  * 運用Logistic Regression 預測玩家推薦傾向。
  * 導入Balanced Error Rate (BER) 評估指標，取代傳統 Accuracy，以解決多數決偏差問題。
  * 最終分類模型取得Accuracy 0.7322 與 BER 0.4563 之成果。

* **回歸任務：情感分析與特徵萃取 (Regression & NLP)**
  * 使用NLTK函式庫（PorterStemmer）進行自然語言前處理。
  * 統計並萃取出現頻率最高的 100 個單字作為文字特徵向量。
  * 訓練 Ridge Regression 模型，量化評論文字與推薦傾向之關聯，取得 MSE 0.1108 的預測誤差。

* **個人化推薦系統優化**
  * 建構基於全域平均 (alpha)、使用者偏差 (beta_u) 與遊戲偏差 (beta_i) 的潛在因子預測模型。
  * 導入L2正規化防止模型過擬合。
  * 利用SciPy的L-BFGS-B最佳化演算法求解最佳參數。
  * 在情境測試中，精準預測目標玩家 (user: skyfii, item: 440) 具備 0.89 的高推薦機率。

## 開發環境與依賴套件
* 程式語言：Python 3
* 機器學習與科學計算：Scikit-learn, SciPy, NumPy
* 自然語言處理：NLTK (Natural Language Toolkit)
* 資料處理：JSON, Gzip, Collections

## 專案結構
* `/src`: 包含資料前處理、分類、回歸與推薦系統最佳化的 Jupyter Notebook 原始碼。
* `/docs`: 包含詳細的演算法分析與專案期末報告。

## 執行方式
1. 確認執行環境已安裝 Python 3.1x。
2. 透過指令 `pip install numpy scipy scikit-learn nltk` 安裝依賴套件。
3. 開啟 Jupyter Notebook 依序運行 `/src` 目錄下的腳本，即可重現各階段分析結果。
