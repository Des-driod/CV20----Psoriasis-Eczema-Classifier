# GET 324 — Laboratory Exercise 10 (Mini-Project)
## Group CV20: Psoriasis vs Eczema — Binary Image Classification

Cloud-deployed CNN application that classifies a skin image as **Eczema** or **Psoriasis**, built with TensorFlow/Keras (MobileNetV2 transfer learning) and deployed as a Streamlit web app.

---

## 🔗 Live App
> **https://cv20----psoriasis-eczema-classifier.streamlit.app/**

##  Repository Contents
| File | Purpose |
|---|---|
| `app.py` | Streamlit application source code |
| `eczema_psoriasis_model.keras` | Trained CNN model |
| `requirements.txt` | Python dependencies for deployment |
| `CV20_Psoriasis_vs_Eczema_Training_Lab_10.ipynb` | Google Colab notebook (27 steps) used to train and evaluate the model |
| `README.md` | This file |

##  Environment / Package Versions
Verified in the notebook's Step 2 (Setup):

| Package | Version |
|---|---|
| Streamlit | 1.60.0 |
| TensorFlow | 2.20.0 |
| NumPy | 2.0.2 |
| Pandas | 2.2.2 |
| Matplotlib | 3.10.0 |
| Seaborn | 0.13.2 |
| scikit-learn | 1.6.1 |
| joblib | 1.5.3 |

##  Approach (27-step notebook)
1. **Dataset**: "Acne, Psoriasis, Eczema vs All Skin Diseases" dataset (Kaggle, `sufiahmad883/acne-psoriasis-eczema-vs-all-skin-diseases`), filtered to only the Eczema and Psoriasis classes.
2. **Split**: a genuine three-way split — **Training (70%)**, **Validation (15%)**, **Test (15%)** — kept as separate, non-overlapping folders, so the final evaluation is on data the model has never seen or been tuned against.
3. **Visual comparison (Step 16)**: side-by-side Eczema vs Psoriasis sample grids, plus an average colour-channel comparison (bar chart + tone swatches) to make the visual difference between the two conditions clearer.
4. **Model**: MobileNetV2 pre-trained on ImageNet, used as a frozen feature extractor with a custom classification head (global average pooling → dropout 0.3 → sigmoid output), then fine-tuned on the top 30 layers of the backbone.
5. **Training**: two-phase transfer learning — 20 epochs with the backbone frozen, followed by 15 fine-tuning epochs with the top layers unfrozen, both phases using early stopping on validation loss.
6. **Evaluation (Step 25)**: classification report, test accuracy, and confusion matrix, computed once on the held-out Test set.
7. **Precision/Recall/F1 (Step 26)**: extracted into a table and a per-class bar chart for a clear, at-a-glance comparison.
8. **Deployment**: Streamlit Community Cloud, reading the saved `.keras` model to serve predictions on uploaded images.

## 📊 Results

**Dataset split** (7,012 images total: 3,200 Eczema, 3,812 Psoriasis):
| Split | Eczema | Psoriasis | Total |
|---|---|---|---|
| Train (70%) | 2,240 | 2,668 | 4,908 |
| Validation (15%) | 480 | 571 | 1,051 |
| Test (15%) | 480 | 573 | 1,053 |

**Test set performance** (held-out, never seen during training or tuning):
| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Eczema | 0.733 | 0.808 | 0.769 | 480 |
| Psoriasis | 0.824 | 0.754 | 0.788 | 573 |
| **Accuracy** | | | **0.779** | 1,053 |
| Macro avg | 0.779 | 0.781 | 0.778 | 1,053 |
| Weighted avg | 0.783 | 0.779 | 0.779 | 1,053 |

Test loss: 0.4751 · Test accuracy: 77.87%

**Confusion matrix** (rows = true label, columns = predicted):
| | Predicted: Eczema | Predicted: Psoriasis |
|---|---|---|
| **True: Eczema** | 388 | 92 |
| **True: Psoriasis** | 141 | 432 |

## 📝 Project Report (100–150 words)

> This project addresses binary classification of Psoriasis and Eczema skin images using a MobileNetV2 transfer-learning CNN. The dataset (7,012 images total) was sourced from a public Kaggle skin-disease dataset, filtered to the two target classes and split into training (4,908), validation (1,051), and test (1,053) sets to ensure an honest, leakage-free evaluation. The base network was first trained with frozen weights, then fine-tuned on its top 30 layers with data augmentation (flip, rotation, zoom) to reduce overfitting. A dedicated visual comparison step highlighted typical colour and texture differences between the two conditions. On the held-out test set, the model achieved 77.9% accuracy, with a macro-averaged precision, recall, and F1-score of 0.78. Eczema was classified with higher recall (81%) while Psoriasis was classified with higher precision (82%), reflecting some visual overlap between the two conditions. The trained model was exported as a `.keras` file and served through a Streamlit web application for instant, confidence-scored predictions on uploaded images.

## 👥 Group Members (Group CV20)
| Name | Registration Number | GitHub Username |
|---|---|---|
| Asuquo, Destiny Bassey (Group Leader) | 23/EG/CV/069 | Des-driod |
| Johnny, Gideon Uko | 23/EG/CV/079 | gideon1500johnny-hash |
| Darby, Beulah Ephraim | 23/EG/CV/049 | beulahdarby1-hue |
| Mkpanam, Divine Philip | 23/EG/CV/019 | divinemkpanam142-glitch |
| Bright, Innocent Ekpe | 23/EG/CV/029 | bright18-ux |
| Paul, Alpha Francis | 23/EG/CV/009 | alphafrancis272-cyber |
| Frederick, Sagelesu Nwiuebari | 23/EG/CV/039 | Codedlygit |
| Akpan, Augustus Augustus | 23/EG/CV/089 | *(pending)* |


