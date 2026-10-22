# Target Screening

## 1. Hugging Face 모델 가중치 다운로드 설정

앱은 모델 가중치를 함께 내려받지 않습니다. 처음 DTA 예측을 실행할 때 Hugging Face에서 필요한 파일을 자동으로 다운로드합니다. 사용할 공개 모델 저장소는 `Jjunyhyeon/target-screening-models`입니다.

프로젝트 폴더에서 환경 설정 파일을 복사합니다:

```bash
cp .env.example .env
```

`.env`에 아래 저장소 ID가 설정되어 있는지 확인합니다:

```dotenv
TARGET_SCREENING_HF_REPO=Jjunyhyeon/target-screening-models
```

저장소가 공개되어 있으므로 Hugging Face 계정이나 `HF_TOKEN`은 필요하지 않습니다. 앱에서 DTA 데이터셋을 선택하고 예측을 시작하면 해당 데이터셋의 5개 fold 체크포인트와 PLM 체크포인트가 자동 다운로드됩니다. 받은 파일은 Hugging Face 캐시에 보관되어 다음부터 재사용됩니다. 최초 다운로드 시 인터넷 연결과 모델 파일 용량만큼의 디스크 공간이 필요합니다.

모델 저장소 안의 파일 경로는 아래와 같아야 합니다 (`vendor/` 접두사 없음):

```text
PLM/checkpoint_step500000.pt
DTA_model/result/DAVIS-complete/fold1/best.pth
DTA_model/result/DAVIS-complete/fold2/best.pth
DTA_model/result/DAVIS-complete/fold3/best.pth
DTA_model/result/DAVIS-complete/fold4/best.pth
DTA_model/result/DAVIS-complete/fold5/best.pth
DTA_model/result/IC50/fold1/best.pth  (fold1~fold5)
DTA_model/result/Ki/fold1/best.pth    (fold1~fold5)
DTA_model/result/Kd/fold1/best.pth   (fold1~fold5)
```

예측 코드 자체는 프로젝트의 `vendor/PLM`과 `vendor/DTA_model`에 필요합니다. Hugging Face에서 내려받는 것은 가중치 파일입니다.

## 2. 앱 설치 및 실행

```bash
bash start.sh
```

처음 실행하면 스크립트가 필요한 Python 환경과 도구를 준비하고 Streamlit 앱을 엽니다. 브라우저가 자동으로 열리지 않으면 `http://localhost:8501`로 접속하세요.

## 다른 컴퓨터에서 접속

```bash
STREAMLIT_SERVER_ADDRESS=0.0.0.0 bash start.sh
```

같은 네트워크에서 실행 컴퓨터의 IP 주소와 포트 `8501`로 접속합니다.
