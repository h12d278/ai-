import warnings

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 이 예제는 작은 수기 데이터로 동작하는 데모입니다. 실제 진단이나 상담을 대체하지 않습니다.
warnings.filterwarnings("ignore")


def configure_font():
    """사용 가능한 한글 폰트를 선택합니다."""
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    preferred_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]

    for font in preferred_fonts:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_font()

# 비식별 예시 데이터: 실제 청소년 수집 데이터가 아닙니다.
SAMPLE_DATA = [
    # 학업
    {"text": "시험 결과가 생각보다 안 나와서 너무 불안하고 속상하다.", "emotion": "불안", "situation": "학업"},
    {"text": "다시 공부를 시작하려는데 어디서부터 해야 할지 막막하다.", "emotion": "불안", "situation": "학업"},
    {"text": "진로와 성적 걱정 때문에 밤에 잠이 잘 오지 않는다.", "emotion": "불안", "situation": "학업"},
    {"text": "원하던 모의고사 성적을 받아서 정말 기쁘고 보람차다.", "emotion": "기쁨", "situation": "학업"},
    {"text": "열심히 준비한 수행평가 발표를 무사히 마쳐서 후련하다.", "emotion": "기쁨", "situation": "학업"},
    # 생활습관
    {"text": "요즘 잠을 4시간밖에 못 자서 온몸이 피곤하고 짜증이 난다.", "emotion": "분노", "situation": "생활습관"},
    {"text": "늦게 자서 아침에 일어나기 힘들고 하루 종일 무기력하고 우울하다.", "emotion": "슬픔", "situation": "생활습관"},
    {"text": "일찍 자고 푹 쉬었더니 머리가 맑아지고 기분이 상쾌하다.", "emotion": "기쁨", "situation": "생활습관"},
    # 친구 관계
    {"text": "친구랑 사소한 일로 다투고 나서 하루 종일 마음이 답답하고 화가 난다.", "emotion": "분노", "situation": "친구 관계"},
    {"text": "친구들이 나를 오해하는 것 같아서 서운하고 외롭다.", "emotion": "슬픔", "situation": "친구 관계"},
    {"text": "친한 친구와 점심시간에 수다를 떨며 크게 웃어서 즐거웠다.", "emotion": "기쁨", "situation": "친구 관계"},
    # 가족 / 기타
    {"text": "부모님과 진로 문제로 대화가 통하지 않아 답답하고 화가 났다.", "emotion": "분노", "situation": "가족"},
    {"text": "가족들이 내 노력을 알아주고 응원해 주어서 마음이 따뜻하고 고맙다.", "emotion": "기쁨", "situation": "가족"},
    {"text": "아무 이유 없이 마음이 가라앉고 눈물이 날 것 같아 슬프다.", "emotion": "슬픔", "situation": "기타"},
]

EMOTIONS = {row["emotion"] for row in SAMPLE_DATA}
SITUATIONS = ["학업", "친구 관계", "가족", "생활습관", "기타"]


def build_model():
    """예시 문장으로 TF-IDF + 로지스틱 회귀 모델을 학습합니다."""
    train_df = pd.DataFrame(SAMPLE_DATA)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=1.0, random_state=42, max_iter=1000)),
        ]
    )
    model.fit(train_df["text"], train_df["emotion"])
    return model


class MindLogAI:
    """감정 일기를 분석하고, 저장을 선택한 기록을 세션 동안 보관합니다."""

    def __init__(self, trained_model):
        self.model = trained_model
        self.user_logs = []
        self.log_count = 0

    def record_and_analyze(self, text, situation, user_override_emotion=None, save=True):
        text = text.strip()
        if not text:
            raise ValueError("일기 문장을 입력해 주세요.")
        if situation not in SITUATIONS:
            raise ValueError(f"상황은 다음 중 하나여야 합니다: {', '.join(SITUATIONS)}")
        if user_override_emotion is not None and user_override_emotion not in EMOTIONS:
            raise ValueError(f"감정은 다음 중 하나여야 합니다: {', '.join(sorted(EMOTIONS))}")

        self.log_count += 1
        pred_emotion = self.model.predict([text])[0]
        probs = self.model.predict_proba([text])[0]
        confidence = float(np.max(probs)) * 100

        print("\n" + "=" * 60)
        print(f"📝 [마음로그 AI 분석 리포트 #{self.log_count}]")
        print(f"• 기록 문장 : \"{text}\"")
        print(f"• 상황 유형 : [{situation}]")
        print("-" * 60)
        print(f"💡 AI 참고 분석: 작성하신 글에서 [{pred_emotion}] 감정이 느껴집니다. (모델 점수: {confidence:.1f}%)")
        print("※ 예시 데이터만 학습한 데모이며, 모델 점수는 정확도나 확률을 보장하지 않습니다. 진단을 대신하지 않습니다.")

        final_emotion = user_override_emotion or pred_emotion
        if user_override_emotion:
            print(f"✏️ 사용자 수정 반영: [{pred_emotion}] → [{final_emotion}]")

        self._suggest_reflection(final_emotion, situation)

        if save:
            self.user_logs.append(
                {
                    "sentence_id": self.log_count,
                    "text": text,
                    "emotion": final_emotion,
                    "situation": situation,
                }
            )
            print("🗂️ [저장 완료] 기록은 현재 프로그램 실행 중에만 메모리에 보관됩니다.")
        else:
            print("🗑️ [미저장] 이 기록은 목록에 추가하지 않았습니다.")

    @staticmethod
    def _suggest_reflection(emotion, situation):
        print("\n🌱 [스스로 돌아보는 질문 & 가벼운 제안]")
        if emotion == "불안" and situation == "학업":
            print("• 질문: '지금 내가 통제할 수 없는 결과까지 미리 걱정하고 있지는 않나요?'")
            print("• 제안: 잠시 스트레칭을 하며 호흡을 정리해 보세요.")
        elif situation == "생활습관" or emotion == "분노":
            print("• 질문: '최근 수면 시간이나 휴식이 부족해 예민해진 상태는 아닐까요?'")
            print("• 제안: 물을 마시고, 오늘은 충분히 쉴 수 있는 시간을 마련해 보세요.")
        elif situation == "친구 관계":
            print("• 질문: '상대방의 말이나 행동을 내 입장에서만 해석한 부분은 없을까요?'")
            print("• 제안: 감정이 가라앉을 때까지 잠시 쉬며 좋아하는 음악을 들어보세요.")
        else:
            print("• 질문: '오늘 경험한 이 감정이 나에게 어떤 의미를 남겼나요?'")
            print("• 제안: 짧은 산책이나 휴식 시간을 가져보세요.")

    def show_dashboard(self):
        if not self.user_logs:
            print("\n📊 시각화할 저장된 기록이 없습니다.")
            return

        df_user = pd.DataFrame(self.user_logs)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        emotion_counts = df_user["emotion"].value_counts()
        axes[0].bar(emotion_counts.index, emotion_counts.values, color="#6C9BCF", width=0.45)
        axes[0].set_title("Emotion Frequency (감정 유형별 기록 수)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Emotion (감정)")
        axes[0].set_ylabel("Count (기록 횟수)")
        axes[0].set_yticks(range(0, int(emotion_counts.max()) + 2))

        situation_counts = df_user["situation"].value_counts()
        axes[1].pie(
            situation_counts.values,
            labels=situation_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#A0C4FF", "#BDB2FF", "#FFADAD", "#FDFFB6", "#CAFFBF"],
        )
        axes[1].set_title("Situation Ratio (감정 발생 상황 비율)", fontsize=12, fontweight="bold")

        plt.suptitle("MindLog AI - Personal Stress & Emotion Report", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()


def run_demo():
    print("=" * 60)
    print("       🧠 마음로그 AI (MindLog AI) 실행 시뮬레이션")
    print("=" * 60)

    app = MindLogAI(build_model())
    app.record_and_analyze(
        text="시험은 끝났는데 성적표 나올까 봐 계속 마음이 불안하고 초조하다.",
        situation="학업",
    )
    app.record_and_analyze(
        text="어제 늦게까지 폰을 보느라 잠을 못 자서 오늘 하루 종일 예민하고 짜증이 났다.",
        situation="생활습관",
        user_override_emotion="분노",
    )
    app.record_and_analyze(
        text="친한 친구와 진로 고민을 나누고 서로 응원해 주니 마음이 한결 가볍고 기뻤다.",
        situation="친구 관계",
    )
    app.record_and_analyze(
        text="오늘 친구랑 사소한 오해가 있었는데 개인적인 내용이라 저장은 안 하고 싶다.",
        situation="친구 관계",
        save=False,
    )
    app.show_dashboard()


if __name__ == "__main__":
    run_demo()
