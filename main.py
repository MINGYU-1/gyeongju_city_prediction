# -*- coding: utf-8 -*-
"""
main.py: 경주 풍력 발전량 예측 통합 파이프라인

PPT 고도화 전략(NWP 데이터 융합, LightGBM 모델)을 반영한 최종 코드입니다.
- xlsx 파일의 실제 변수명을 정확히 반영했습니다.
- 데이터 전처리부터 모델 학습, 예측까지 모든 과정을 클래스로 통합 관리합니다.

[실행 방법]
1. 모델 학습: python main.py --mode train
2. 2024년 발전량 예측: python main.py --mode predict
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import argparse
from tqdm import tqdm

class WindPowerPredictor:
    """경주 풍력 발전량 예측 모델 클래스"""

    def __init__(self, data_dir='./data', output_dir='./output'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processed_data_path = os.path.join(self.output_dir, 'scada_processed_hourly.csv')
        self.model_path = os.path.join(self.output_dir, 'lgbm_model.pkl')
        self.features_path = os.path.join(self.output_dir, 'features.pkl')

        os.makedirs(self.output_dir, exist_ok=True)

        # --- XLSX에서 직접 확인한 정확한 컬럼명 ---
        self.DATE_COL = 'Date/Time'
        self.ENERGY_COL = 'Energy Production\nActive Energy Production\n[KWh]'
        self.SENSOR_COLS = {
            'nacelle_wind_speed': 'Nacelle\nWind Speed\n[m/s]',
            'nacelle_wind_dir': 'Nacelle\nWind Direction\n[deg]',
            'rotor_speed': 'Rotor\nRotor Speed\n[rpm]',
            'air_density': 'Nacelle\nAir Density\n[kg/㎥]'
        }
        self.TARGET_COL = 'total_energy_kwh'

    def _preprocess_scada_data(self):
        """
        4개년치 SCADA xlsx 파일을 읽어 전처리하고 시간당 데이터로 집계합니다.
        PPT의 'Step 1, 2, 3' 데이터 준비 과정을 자동화합니다.
        """
        print("SCADA 원본 데이터 전처리를 시작합니다...")
        if os.path.exists(self.processed_data_path):
            print(f"이미 전처리된 파일이 존재합니다: {self.processed_data_path}")
            return pd.read_csv(self.processed_data_path, parse_dates=['timestamp'])

        xlsx_files = [f for f in os.listdir(self.data_dir) if f.endswith('.xlsx') and 'scada' in f]
        if not xlsx_files:
            raise FileNotFoundError(f"{self.data_dir} 에서 SCADA 파일을 찾을 수 없습니다.")

        all_dfs = []
        for file in sorted(xlsx_files):
            print(f"파일 처리 중: {file}")
            file_path = os.path.join(self.data_dir, file)
            xls = pd.ExcelFile(file_path)
            
            # 각 터빈(시트)의 데이터를 읽어옵니다.
            for sheet_name in tqdm(xls.sheet_names, desc=f"  - {file} 시트"):
                try:
                    cols_to_read = [self.DATE_COL, self.ENERGY_COL] + list(self.SENSOR_COLS.values())
                    df = xls.parse(sheet_name=sheet_name, usecols=cols_to_read, header=5)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"  - {sheet_name} 시트 처리 중 오류: {e}")

        df_total = pd.concat(all_dfs, ignore_index=True)
        df_total = df_total.rename(columns={v: k for k, v in self.SENSOR_COLS.items()})
        df_total = df_total.rename(columns={self.ENERGY_COL: 'energy_kwh'})

        # 시간 데이터 변환 및 불필요 행 제거
        df_total[self.DATE_COL] = pd.to_datetime(df_total[self.DATE_COL], errors='coerce')
        df_total = df_total.dropna(subset=[self.DATE_COL])
        df_total = df_total.sort_values(self.DATE_COL).set_index(self.DATE_COL)

        # 발전량은 터빈별로 합산, 센서는 평균
        df_energy = df_total[['energy_kwh']].resample('1H').sum(min_count=1).rename(columns={'energy_kwh': self.TARGET_COL})
        df_sensors = df_total[list(self.SENSOR_COLS.keys())].resample('1H').mean()

        # 데이터 병합
        df_hourly = pd.merge(df_energy, df_sensors, left_index=True, right_index=True, how='inner')
        
        # 결측치 처리 (선형 보간 후 앞/뒤 값으로 채우기)
        df_hourly = df_hourly.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 풍향(nacelle_wind_dir)을 sin/cos로 변환하여 순환성 표현
        df_hourly['wind_dir_sin'] = np.sin(np.deg2rad(df_hourly['nacelle_wind_dir']))
        df_hourly['wind_dir_cos'] = np.cos(np.deg2rad(df_hourly['nacelle_wind_dir']))
        df_hourly = df_hourly.drop(columns=['nacelle_wind_dir'])

        df_hourly.reset_index(inplace=True)
        df_hourly = df_hourly.rename(columns={self.DATE_COL: 'timestamp'})
        df_hourly.to_csv(self.processed_data_path, index=False)
        print(f"전처리 완료. 파일 저장: {self.processed_data_path}")
        return df_hourly

    def _create_features(self, df):
        """
        PPT 고도화 전략 ③: 피처 엔지니어링
        시간 관련 특성을 생성하여 모델 성능을 향상시킵니다.
        """
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # 시간의 순환성을 모델에 알려주기 위한 sin/cos 변환
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        
        return df

    def train(self):
        """PPT 고도화 전략 ①, ②, ④: 데이터 융합, 모델 재설계, 새로운 파이프라인"""
        print("--- 모델 학습 파이프라인 시작 ---")
        
        # 1. SCADA 데이터 전처리
        scada_df = self._preprocess_scada_data()

        # 2. 과거 NWP 데이터 융합 (현재는 가상 데이터 생성)
        # TODO: 이 부분은 실제 과거 기상 예보 데이터(기상청 제공)로 반드시 대체해야 합니다.
        print("가상으로 과거 NWP 데이터를 생성합니다. (실제 데이터로 대체 필요)")
        nwp_past_df = pd.DataFrame({
            'timestamp': pd.to_datetime(scada_df['timestamp']),
            'nwp_wind_speed': np.random.uniform(0, 25, len(scada_df)),
            'nwp_wind_dir': np.random.uniform(0, 360, len(scada_df)),
            'nwp_temperature': np.random.uniform(-10, 35, len(scada_df)),
        })
        
        # 3. 데이터 병합 및 피처 엔지니어링
        df_train = pd.merge(scada_df, nwp_past_df, on='timestamp', how='inner')
        df_train = self._create_features(df_train)
        
        # 4. 학습 데이터 준비
        FEATURES = [col for col in df_train.columns if col not in [self.TARGET_COL, 'timestamp']]
        TARGET = self.TARGET_COL
        
        X = df_train[FEATURES]
        y = df_train[TARGET]
        
        print(f"학습에 사용될 피처 ({len(FEATURES)}개): {FEATURES}")
        
        # 5. LightGBM 모델 학습
        lgbm = lgb.LGBMRegressor(
            objective='regression_l1',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        
        print("LightGBM 모델 학습 시작...")
        lgbm.fit(X, y, eval_set=[(X, y)], callbacks=[lgb.early_stopping(100, verbose=False)])
        
        # 6. 모델 및 피처 정보 저장
        joblib.dump(lgbm, self.model_path)
        joblib.dump(FEATURES, self.features_path)
        
        print(f"--- 모델 학습 완료 ---")
        print(f"모델 저장 경로: {self.model_path}")

    def predict(self):
        """학습된 모델을 사용하여 2024년 발전량을 예측합니다."""
        print("--- 2024년 발전량 예측 파이프라인 시작 ---")
        
        try:
            model = joblib.load(self.model_path)
            features = joblib.load(self.features_path)
        except FileNotFoundError:
            print("오류: 학습된 모델이 없습니다. 'train' 모드로 먼저 학습을 진행하세요.")
            return

        # 1. 2024년 예측을 위한 미래 NWP 데이터 생성
        # TODO: 기상청 API 등을 통해 실제 2024년 예보 데이터로 대체해야 합니다.
        print("2024년 예측을 위한 가상 NWP 데이터를 생성합니다. (실제 예보 데이터로 대체 필요)")
        future_dates = pd.date_range("2024-01-01 00:00:00", "2024-12-31 23:00:00", freq="h")
        df_future = pd.DataFrame({'timestamp': future_dates})
        
        # 가상 데이터 생성: 실제 데이터의 통계적 특성을 반영하면 더 좋습니다.
        for feature in features:
            if feature not in df_future.columns:
                if 'speed' in feature:
                    df_future[feature] = np.random.uniform(0, 25, len(future_dates))
                elif 'density' in feature:
                    df_future[feature] = np.random.uniform(1.1, 1.3, len(future_dates))
                elif 'dir' in feature or 'sin' in feature or 'cos' in feature:
                     df_future[feature] = np.random.uniform(-1, 1, len(future_dates))
                else: # 온도, rpm 등 기타
                    df_future[feature] = np.random.uniform(0, 20, len(future_dates))
        
        # 2. 피처 엔지니어링
        df_future = self._create_features(df_future)
        
        # 3. 학습 시 사용한 피처와 순서를 동일하게 맞춤
        X_future = df_future[features]
        
        # 4. 발전량 예측
        predictions = model.predict(X_future)
        
        # 5. 결과 저장
        df_result = pd.DataFrame({
            'timestamp': future_dates,
            'predicted_energy_kwh': predictions
        })
        df_result['predicted_energy_kwh'] = df_result['predicted_energy_kwh'].clip(lower=0)
        
        save_path = os.path.join(self.output_dir, "power_predicted_2024_final.csv")
        df_result.to_csv(save_path, index=False)
        
        print("--- 2024년 발전량 예측 완료 ---")
        print(f"결과 저장 경로: {save_path}")
        print("\n[예측 결과 샘플]")
        print(df_result.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="경주 풍력 발전량 예측 통합 파이프라인")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['train', 'predict'],
        help="실행 모드: 'train' (모델 학습), 'predict' (미래 예측)"
    )
    
    args = parser.parse_args()
    predictor = WindPowerPredictor()
    
    if args.mode == 'train':
        predictor.train()
    elif args.mode == 'predict':
        predictor.predict()