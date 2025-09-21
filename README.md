네, 알겠습니다. `gyeonju_city.py`에서 `main.py`로 변경되면서 얻게 된 장점들을 중심으로 `README.md` 파일을 작성해 드리겠습니다.

아래 내용을 복사하여 `README.md` 파일에 붙여넣으시면 됩니다.

-----

# 경주 풍력 발전량 예측 프로젝트

본 프로젝트는 경주 지역의 풍력 발전소 SCADA(감시 제어 데이터 수집) 데이터와 NWP(수F치예보) 데이터를 활용하여 미래의 시간당 발전량을 예측하는 머신러닝 파이프라인입니다.

`LightGBM` 모델을 사용하여 기상 조건과 발전량 간의 복잡한 관계를 학습하고, 이를 통해 안정적인 에너지 수급 계획에 기여하는 것을 목표로 합니다.

## `main.py` 도입의 주요 개선점

기존의 `gyeonju_city.py` 스크립트 방식에서 `main.py`의 객체 지향 파이프라인 방식으로 코드를 재구성하며 얻은 장점은 다음과 같습니다.

  * **1. 체계적인 구조 (모듈화):**

      * **(Before)** `gyeonju_city.py`는 절차적으로 코드가 나열되어 있어 수정 및 관리가 어려웠습니다.
      * **(After)** `main.py`는 `WindPowerPredictor` 클래스를 중심으로 데이터 전처리, 피처 엔지니어링, 학습, 예측 기능이 **독립적인 함수(메서드)로 분리**되어 있습니다. 이로 인해 코드의 가독성이 향상되고 기능별 유지보수가 매우 용이해졌습니다.

  * **2. 자동화된 파이프라인:**

      * **(Before)** 각 단계별로 파일을 생성하고 다음 단계의 입력으로 사용하는 등 수동 개입이 필요했습니다.
      * **(After)** `python main.py --mode train` 명령어 하나로 **원본 데이터(xlsx) 로딩부터 최종 모델 저장까지 모든 과정이 자동으로 실행**됩니다. 수동 작업으로 인한 실수를 원천적으로 방지합니다.

  * **3. 재사용성 및 확장성:**

      * **(Before)** 다른 프로젝트에 코드 일부를 가져다 쓰기 어려운 구조였습니다.
      * **(After)** `WindPowerPredictor` 클래스는 다른 데이터나 모델에도 적용할 수 있도록 설계되어 **재사용성**이 높습니다. 새로운 피처를 추가하거나 다른 모델로 교체하는 등의 **확장** 작업이 훨씬 수월합니다.

  * **4. 명확한 실행 인터페이스:**

      * **(Before)** 코드의 어느 부분을 실행해야 하는지 파악하기 어려웠습니다.
      * **(After)** `--mode train`, `--mode predict` 와 같은 **명령어 기반의 명확한 실행 인터페이스**를 제공하여 사용자가 의도한 작업을 실수 없이 수행할 수 있도록 돕습니다.

## 프로젝트 구조

```
/gyeongju_city_prediction/
│
├── data/                 # 원본 데이터(.xlsx) 저장 위치
│   ├── scada_gyeongju_2020_10min.xlsx
│   └── ...
│
├── output/               # 결과물 저장 위치 (자동 생성)
│   ├── scada_processed_hourly.csv
│   ├── lgbm_model.pkl
│   └── power_predicted_2024_final.csv
│
├── main.py               # 메인 파이프라인 스크립트
├── requirements.txt      # 필요 패키지 목록
└── README.md             # 프로젝트 설명서
```

## 설치 및 환경 설정

1.  **가상 환경 생성 및 활성화**

    ```bash
    # 가상 환경 생성
    python -m venv venv

    # 가상 환경 활성화 (Windows)
    .\venv\Scripts\activate

    # 가상 환경 활성화 (macOS/Linux)
    source venv/bin/activate
    ```

2.  **필수 패키지 설치**

    ```bash
    pip install -r requirements.txt
    ```

## 실행 방법

프로젝트의 모든 기능은 `main.py`를 통해 실행됩니다.

1.  **모델 학습**
    `data` 폴더의 데이터를 이용하여 모델을 학습시키고 `output` 폴더에 결과물(전처리 데이터, 모델 파일)을 저장합니다.

    ```bash
    python main.py --mode train
    ```

2.  **발전량 예측**
    학습된 모델을 불러와 2024년의 발전량을 예측하고 `output` 폴더에 `power_predicted_2024_final.csv` 파일을 생성합니다.

    ```bash
    python main.py --mode predict
    ```

## 향후 개선 계획 (TODO)

  * **실제 NWP 데이터 연동:** 현재 가상으로 생성하고 있는 NWP(수치예보) 데이터를 기상청 API 등 실제 데이터와 연동하여 모델의 예측 정확도 향상.