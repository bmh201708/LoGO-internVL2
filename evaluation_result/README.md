# LOGO Evaluation Results Summary

This document summarizes and compares the evaluation results of **InternVL2-2B** and **Qwen2-VL-7B** on the LoGO benchmark.

**Summary Files Merged:**
- InternVL2-2B: `summary-intern.md` 
- Qwen2-VL-7B: `summary-qwen.md` 

---

## 🚀 Overall Performance Comparison

| Model | App-level Step Acc | App-level Episode Acc | Category-level Step Acc | Category-level Episode Acc |
| :--- | :---: | :---: | :---: | :---: |
| **InternVL2-2B** | 74.8% | 74.9% | 73.53% | 74.42% |
| **Qwen2-VL-7B** | **83.6%** | **83.81%** | **84.31%** | **82.56%** |

---

## 📱 App-level Detailed Comparison (14 Apps)

| App | InternVL2 Step | InternVL2 Episode | Qwen2-VL Step | Qwen2-VL Episode |
| :--- | :---: | :---: | :---: | :---: |
| adidas | 55.0% | 55.0% | 70.0% | 70.0% |
| amazon | 86.67% | 86.67% | 100.0% | 100.0% |
| calendar | 75.0% | 75.0% | 90.0% | 90.0% |
| clock | 100.0% | 100.0% | 86.67% | 86.67% |
| decathlon | 65.0% | 65.0% | 65.0% | 65.0% |
| ebay | 80.0% | 80.0% | 80.0% | 80.0% |
| etsy | 70.0% | 70.0% | 90.0% | 90.0% |
| flipkart | 60.0% | 60.0% | 60.0% | 60.0% |
| gmail | 100.0% | 100.0% | 100.0% | 100.0% |
| google_drive | 90.0% | 90.0% | 100.0% | 100.0% |
| google_maps | 80.0% | 80.0% | 90.0% | 90.0% |
| kitchen_stories | 50.0% | 50.0% | 65.0% | 65.0% |
| reminder | 93.33% | 92.86% | 100.0% | 100.0% |
| youtube | 60.0% | 60.0% | 80.0% | 80.0% |
| **Average** | **76.07%** | **76.04%** | **84.05%** | **84.05%** |
| **OVERALL** | **74.8%** | **74.9%** | **83.6%** | **83.81%** |

---

## 📂 Category-level Detailed Comparison (5 Categories)

| Category | InternVL2 Step | InternVL2 Episode | Qwen2-VL Step | Qwen2-VL Episode |
| :--- | :---: | :---: | :---: | :---: |
| entertainment | 66.67% | 69.23% | 80.0% | 76.92% |
| lives | 76.19% | 75.0% | 95.24% | 93.75% |
| office | 68.42% | 80.0% | 78.95% | 80.0% |
| shopping | 78.12% | 77.42% | 84.38% | 83.87% |
| traveling | 73.33% | 66.67% | 80.0% | 75.0% |
| **Average** | **72.55%** | **73.66%** | **83.71%** | **81.91%** |
| **OVERALL** | **73.53%** | **74.42%** | **84.31%** | **82.56%** |

