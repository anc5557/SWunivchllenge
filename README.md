# 🧠 2022 소프트웨어중심대학 공동 딥러닝 챌린지

**OpenMax 기반 Open Set Image Classification 프로젝트**

본 저장소는 2022년 여름에 개최된 **소프트웨어중심대학 공동 딥러닝 챌린지** 참가 프로젝트를 정리한 포트폴리오용 리포지터리입니다.
학습 데이터에 존재하지 않는 클래스(Unknown)가 테스트 데이터에 포함되는 **Open Set 환경**에서 안정적인 이미지 분류 모델을 구현하는 것이 핵심 목표였습니다.

---

## 📅 대회 정보

* **대회 기간:** 2022.07.21 ~ 2022.08.20

* **평가 지표:** Categorization Accuracy

* **참가 현황:**

  * 45팀 참가
  * 28팀 제출
  * 총 1,281회 제출

* **성과:**

  * Public Leaderboard: **0.595**
  * Private Leaderboard: **0.620**
  * **최종 2위 수상 🥈**

---

## 🎯 프로젝트 목표

* 기존 Softmax 기반 분류 모델의 한계인 **Unknown Class 인식 불가 문제** 해결
* Open Set Recognition 기법인 **OpenMax 알고리즘**을 실제 코드에 적용
* 실전 대회 환경에서 Unknown 클래스를 안정적으로 판별할 수 있는 구조 구현

---

## 🛠 사용 기술 스택

| 구분                | 내용                               |
| ----------------- | -------------------------------- |
| Language          | Python                           |
| Framework         | PyTorch                          |
| Environment       | Google Colab, Kaggle             |
| Core Algorithm    | OpenMax (Open Set Recognition)   |
| Library           | libmr, numpy, torch, torchvision |
| Data Augmentation | Flip, Rotation, Color Jitter 등   |

---

## 🧩 OpenMax 알고리즘 개요

OpenMax는 기존 Softmax 분류기를 확장하여 **Unknown 클래스에 대한 확률을 명시적으로 계산**할 수 있도록 만든 Open Set Recognition 알고리즘입니다.

일반 Softmax는 다음과 같은 가정을 합니다.

> “입력 샘플은 반드시 학습된 클래스 중 하나에 속한다.”

따라서 전혀 다른 데이터가 들어와도 가장 확률이 높은 클래스 하나로 강제 분류됩니다.

Softmax 확률식:

$$
P(y=i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

이 구조에서는 Unknown 개념이 존재하지 않습니다.

---

## 🚀 OpenMax 핵심 아이디어

1. **클래스별 Activation Vector 평균 계산**

   * 학습 데이터 중 정답을 맞춘 샘플들의 마지막 레이어 출력 벡터를 모아
   * 클래스별 평균 벡터(Class Mean Vector)를 생성

2. **Weibull 분포 기반 거리 모델링 (libmr)**

   * 각 클래스별로 샘플과 평균 벡터 사이의 거리 분포를 계산
   * 상위 η개 거리값으로 Weibull 분포를 피팅
   * 해당 샘플이 “해당 클래스에 속할 신뢰도”를 확률적으로 모델링

3. **Activation Vector 재보정(Recalibration)**

   * 기존 Softmax score를 Weibull 기반 신뢰도로 감소
   * 감소된 확률을 **Unknown 클래스 확률로 재분배**

4. **OpenMax 확률 벡터 생성**

기존:

```text
[class1, class2, ..., class80]
```

OpenMax:

```text
[class1, class2, ..., class80, unknown]
```

5. **Thresholding 기반 Unknown 판별**

```python
if max(probabilities) < threshold:
    label = -1  # unknown
```

---

## 🧪 코드 구조 내 OpenMax 적용 흐름

1. 라이브러리 설치

```bash
pip install libmr
```

2. 클래스별 평균 Activation Vector 생성

```python
for c in np.unique(train_label):
    class_act_vec = ...
    class_mean = class_act_vec.mean(axis=0)
```

3. Weibull 분포 학습

```python
mr.fit_high(dist_to_mean[-eta:], eta)
```

4. OpenMax 확률 계산

```python
def compute_openmax(actvec):
    ...
    return np.exp(rev_actvec) / np.exp(rev_actvec).sum()
```

5. Unknown 포함 최종 예측

```python
def make_prediction(...):
    if max(score) < threshold:
        label = -1
```

---

## 📈 적용 효과

* Softmax 대비 **Unknown 클래스 탐지 가능**
* 오분류 강제 할당 문제 완화
* Private Leaderboard 성능 향상에 핵심 기여
* 실전 Open-set 환경에 적합한 구조

---

## 📂 저장소 구성

| 경로/파일                   | 설명                                                                          |
| ----------------------- | --------------------------------------------------------------------------- |
| `SWunivchallenge.ipynb` | 전체 실험이 담긴 단일 Jupyter Notebook. 데이터 전처리, 모델 학습, OpenMax 적용, 제출 파일 생성까지 모두 포함 |
| `README.md`             | 프로젝트 개요, 대회 정보, OpenMax 알고리즘 설명 문서                                          |

> 본 저장소는 **.ipynb 단일 파일 중심 포트폴리오 구조**로 구성되어 있습니다.

---

## 🔗 참고 자료

* Kaggle Competition
  [https://www.kaggle.com/competitions/2022swunivchallenge](https://www.kaggle.com/competitions/2022swunivchallenge)

* Google Colab Notebook


---
