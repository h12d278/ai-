# 마음로그 AI

예시 문장으로 학습한 TF-IDF + 로지스틱 회귀 기반의 감정 일기 분석 데모입니다. 한글 문장과 상황을 입력하면 참고용 감정 분류, 회고 질문, 그리고 세션 내 기록 대시보드를 보여줍니다.

> **주의:** 학습 데이터는 연구자가 작성한 소량의 예시 문장으로, 실제 청소년 데이터가 아닙니다. 이 프로그램은 교육·시연용이며 감정 분석은 부정확할 수 있고 정신건강 진단이나 전문적인 도움을 대체하지 않습니다. 출력되는 모델 점수도 정확도나 보정된 확률을 의미하지 않습니다.

## 실행 방법

Python 3.9 이상을 권장합니다.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python mindlog_ai.py
```

실행하면 네 가지 예시 기록이 분석되고 대시보드 창이 열립니다. 창을 닫으면 프로그램이 종료됩니다.

## 기록 및 개인정보

`save=True`인 기록은 현재 프로세스의 메모리에만 보관되며 파일이나 외부 서비스로 전송되지 않습니다. 프로그램을 종료하면 기록은 사라집니다. `save=False`로 전달한 기록은 기록 목록에 추가되지 않습니다. 실제 사용자 데이터나 민감한 일기를 다룰 서비스로 사용하기 전에 보안·개인정보 보호 및 청소년 안전 요건을 별도로 검토해야 합니다.

## 코드에서 사용하기

```python
from mindlog_ai import MindLogAI, build_model

app = MindLogAI(build_model())
app.record_and_analyze("시험 결과가 걱정되어 불안하다.", "학업", save=False)
```

상황은 `학업`, `친구 관계`, `가족`, `생활습관`, `기타` 중 하나를 사용하세요. 감정 수정 값은 학습 데이터에 있는 감정인 `불안`, `기쁨`, `분노`, `슬픔` 중 하나여야 합니다.
