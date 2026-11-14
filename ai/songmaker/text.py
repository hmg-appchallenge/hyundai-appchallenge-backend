"""음성 인식 결과에서 명사(가사용)와 의성어(비트용)를 고른다."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

HANGUL = re.compile(r"[가-힣]+")
TITLE_MAX = 12  # 앱 화면에 들어가는 노래 제목 길이 (띄어쓰기 포함 글자 수)

# 의성어로 자주 쓰는 소리 음절
ONO_SYLLABLES = set("둥둠쿵붐붕뿡퉁덩뚱통탕짝탁딱착팍짠쾅펑팡땅쨍빵빰칙치츠쉿슥싹찰랄라룰루나따띠딩동댕뚜삐뽀멍꽥퐁띵똑콩팅뚝뿌"
                    "슝쌩씽쓩휙휭윙뿅짹찍꿀")
ONO_WORDS = {
    "랄랄라", "룰루랄라", "랄라", "라라라", "나나나", "따라라", "띠리리", "야옹", "음매", "꼬끼오", "개굴개굴",
    "부릉부릉", "칙칙폭폭", "딩동댕", "딩동", "짝짝짝", "빵빵", "멍멍", "삐뽀삐뽀", "뿡뿡", "쿵짝", "쿵짝쿵짝",
    "둠칫", "둠칫둠칫", "반짝반짝", "깡충깡충", "뒤뚱뒤뚱", "쿵덕쿵", "꿀꿀", "어흥", "짹짹", "삐약삐약",
    "야옹야옹", "부엉부엉", "히힝", "꽥꽥", "찍찍",
}
# 두 음절이 되풀이되는 의성어·의태어 (야옹야옹, 데굴데굴). 형태소 분석기가 명사로 보는 말(야옹)도 여기 있으면 의성어로 본다
ONO_UNITS = {
    "부릉", "딩동", "삐뽀", "쿵짝", "둠칫", "반짝", "깡충", "개굴", "삐약", "뒤뚱",
    "야옹", "부엉", "음매", "어흥", "히힝", "깡총", "폴짝", "팔짝", "아장", "엉금", "데굴", "빙글", "뱅글", "보글",
    "뽀글", "살랑", "펄럭", "출렁", "첨벙", "철썩", "퐁당", "꿈틀", "꼬물", "토닥", "흔들",
}
MELODY_START = set("랄라룰루나냐리로레띠따")
KICK_START = set("둥둠쿵붐붕뿡퉁덩뚱통둔")
SNARE_START = set("짝탁딱착팍짠쾅펑팡땅쨍빵빰")
HAT_START = set("칙치츠쉿슥싹삭찰")

ANIMALS = set(
    "강아지 고양이 토끼 사자 호랑이 코끼리 기린 원숭이 곰 판다 여우 늑대 돼지 소 말 양 닭 오리 병아리 거북이 거북 "
    "개구리 물고기 상어 고래 돌고래 펭귄 공룡 나비 벌 꿀벌 개미 새 참새 부엉이 독수리 다람쥐 햄스터 하마 악어 얼룩말 "
    "사슴 염소 젖소 멍멍이 야옹이 티라노 문어 오징어 게 꽃게 달팽이 무당벌레 잠자리 코뿔소 고릴라 앵무새 비둘기 "
    "까치 백조 홍학 낙타 캥거루 코알라 수달 해마 불가사리 강아지풀".split()
)
PEOPLE = set(
    "엄마 아빠 할머니 할아버지 언니 오빠 누나 형 동생 아기 친구 선생님 이모 고모 삼촌 외삼촌 사촌 가족 아저씨 "
    "아줌마 경찰관 경찰 소방관 의사 간호사 공주 왕자 왕 여왕 요정 마법사 기사 로봇 아가 외할머니 외할아버지 "
    "할미 할비 동생들 형아 누나들 언니들".split()
)
PLACES = set(
    "바다 산 강 공원 학교 유치원 어린이집 집 놀이터 동물원 수족관 시장 마트 병원 도서관 시골 도시 하늘 숲 섬 "
    "캠핑장 수영장 놀이공원 키즈카페 휴게소 터널 다리 길 우주 달 별나라 해변 바닷가 계곡 호수 농장 목장 "
    "할머니댁 할아버지댁 외갓집 친가 놀이동산 박물관 미술관 기차역 공항 정류장 주차장 빵집 식당 "
    "화장실 주유소 고속도로 톨게이트".split()
)
STOP_NOUNS = set(
    "진짜 정말 이번 다음 오늘 내일 어제 지금 여기 거기 저기 이것 저것 그것 무엇 노래 사람 우리 생각 소리 "
    "마음 시간 정도 하나 둘 셋 처음 이름 친구들 얘기 말씀 조금 "
    # Whisper가 조용한 곳에서 지어내는 말 (시청해 주셔서 감사합니다, 구독과 좋아요 등)
    "시청 시청자 구독 영상 자막 뉴스 기자 감사 채널 알림 여러분 제작 후원 광고 방송".split()
)
# 아이들이 자주 말하는 사물·음식. 동물·사람·장소 목록과 함께 '아는 말'로 본다
KID_OBJECTS = set(
    "사과 바나나 딸기 포도 수박 복숭아 참외 키위 레몬 체리 토마토 당근 감자 고구마 옥수수 우유 주스 과자 사탕 "
    "초콜릿 젤리 아이스크림 케이크 쿠키 김밥 피자 햄버거 치킨 라면 국수 짜장면 계란 달걀 풍선 인형 장난감 블록 "
    "퍼즐 자동차 기차 비행기 헬리콥터 버스 트럭 택시 소방차 경찰차 구급차 포클레인 자전거 킥보드 로켓 우주선 "
    "나무 해님 달님 별님 구름 무지개 눈사람 바람 모래 조개 우산 모자 신발 양말 가방 그림 크레파스 색연필 그네 "
    "미끄럼틀 시소 선물 생일 사랑 행복 웃음 방귀 뽀뽀 박수 배꼽 꽃밭 비눗방울 뽀로로 핑크퐁 상어가족 "
    # 차 안에서 자주 나오는 말 (이 앱은 차에서 쓴다)
    "바퀴 창문 신호등 안전벨트 카시트 휴지 거울 선풍기 에어컨 간식 도시락 과일".split()
)
# 아이 노래에 넣지 않는 말 (욕, 어른 말, 무서운 말). 인식이 틀려서 엉뚱한 말이 나와도 여기서 걸러진다
BLOCK_NOUNS = set(
    "새끼 시발 졸라 미친놈 미친년 애미 애비 니미 느금 년놈 변태 음란 야동 포르노 섹스 성기 성교 자위 콘돔 "
    "강간 성폭행 매춘 창녀 파과 불륜 자살 살인 살해 시체 학살 테러 총살 마약 대마 필로폰 도박 담배 소주 맥주 "
    "양주 술집".split()
)
BLOCK_PARTS = ("씨발", "씨바", "병신", "븅신", "개새", "좆", "존나", "지랄", "염병", "섹스", "자살", "살인", "마약")
# 이보다 자신 없게 들린 명사는 아는 말(동물·사람·장소·사물 목록)일 때만 가사에 쓴다
MIN_NOUN_PROB = 0.5
CATEGORY_LABELS = {"object": "사물", "person": "사람", "animal": "동물", "place": "장소"}


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Noun:
    text: str
    category: str
    member: str
    start: float
    end: float
    frac_start: float = 0.0  # 인식된 어절 안에서 명사가 차지하는 위치 (0~1)
    frac_end: float = 1.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Ono:
    text: str
    role: str  # kick / snare / hat / melody / perc
    member: str
    start: float
    end: float
    frac_start: float = 0.0
    frac_end: float = 1.0

    def as_dict(self) -> dict:
        return asdict(self)


def fit_title(title: str, limit: int = TITLE_MAX) -> str:
    """제목을 limit 글자 안으로 맞춘다. 넘치면 그 안의 마지막 띄어쓰기에서 끊고, 띄어쓰기가 없으면 그냥 자른다."""
    t = " ".join(str(title or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit + 1].rfind(" ")
    return (t[:cut] if cut >= 2 else t[:limit]).strip()


def hangul_only(s: str) -> str:
    return "".join(HANGUL.findall(s))


def _vowel(syl: str) -> int:
    code = ord(syl) - 0xAC00
    return (code % 588) // 28 if 0 <= code < 11172 else -1


def is_onomatopoeia(w: str, kiwi=None) -> bool:
    """반복되는 소리 말(둥둥, 부릉부릉, 랄랄라, 칙칙폭폭)이면 의성어로 본다."""
    if not w:
        return False
    if w in ONO_WORDS:
        return True
    n = len(w)
    if n < 2 or n > 8:
        return False
    # 같은 음절 쌍이 두 번 (칙칙폭폭)
    if n == 4 and w[0] == w[1] and w[2] == w[3] and w[0] in ONO_SYLLABLES:
        return True
    for u in (1, 2):
        if n < 2 * u:
            continue
        unit = w[:u]
        k = 0
        while w[k * u:(k + 1) * u] == unit:
            k += 1
        rest = w[k * u:]
        if k < 2 or len(rest) > u:
            continue
        if rest and rest[0] not in ONO_SYLLABLES:
            continue
        if u == 2 and unit not in ONO_UNITS:
            # 두 음절 반복은 "엄마엄마" 같은 명사 반복일 수 있다
            if not all(c in ONO_SYLLABLES for c in unit):
                if kiwi is not None and any(t.tag.startswith("NN") for t in kiwi.tokenize(unit)):
                    return False
        return True
    return False


def ono_role(w: str) -> str:
    s = w[0]
    if w in {"랄랄라", "룰루랄라", "랄라", "라라라", "나나나", "따라라", "띠리리"} or s in MELODY_START:
        return "melody"
    if s in KICK_START:
        return "kick"
    if s in SNARE_START:
        return "snare"
    if s in HAT_START:
        return "hat"
    v = _vowel(s)
    # ㅗ(8) ㅜ(13) ㅓ(4) 같은 어두운 모음은 낮은 소리 쪽으로
    return "kick" if v in (4, 8, 13, 18) else "perc"


def is_known(noun: str) -> bool:
    return noun in ANIMALS or noun in PEOPLE or noun in PLACES or noun in KID_OBJECTS


def is_blocked(word: str) -> bool:
    return word in BLOCK_NOUNS or any(p in word for p in BLOCK_PARTS)


def usable_noun(noun: str, prob: float = 1.0) -> bool:
    """가사·들린 단어로 써도 되는 명사인지. 두 글자 이상 한글, 빼는 말·막는 말이 아니고, 자신 있게 들렸거나 아는 말."""
    if not HANGUL.fullmatch(noun) or len(noun) < 2 or noun in STOP_NOUNS or is_blocked(noun):
        return False
    return prob >= MIN_NOUN_PROB or is_known(noun)


def category_of(noun: str) -> str:
    if noun in ANIMALS:
        return "animal"
    if noun in PEOPLE:
        return "person"
    if noun in PLACES:
        return "place"
    for p in PLACES:
        if len(p) >= 2 and noun.endswith(p):
            return "place"
    return "object"


@dataclass
class Analysis:
    member: str
    transcript: str
    nouns: list[Noun]
    onos: list[Ono]
    meta_only: list[str]


def _sure_ono(w: str) -> bool:
    """잘못 들은 반복어(등등, 릉릉)가 아니라 정말 의성어인지: 아는 의성어, 소리 음절로만 된 말, 두 음절 되풀이."""
    if w in ONO_WORDS or all(c in ONO_SYLLABLES for c in w):
        return True
    return len(w) >= 4 and len(w) % 2 == 0 and w[:2] in ONO_UNITS and w == w[:2] * (len(w) // 2)


def analyze(words: list[Word], member: str, kiwi, strict: bool = False) -> Analysis:
    """strict: 실시간 받아쓰기처럼 덜 정확한 글(띄어쓰기 없음, 잘못 들은 말 섞임)에서 고를 때.
    명사는 아이들이 자주 쓰는 '아는 말'만(토끼를 토기로, 공룡을 공용으로 잘못 들은 말은 뺀다),
    의성어는 _sure_ono를 지나는 말만 쓴다. 나머지 단어는 Whisper가 다시 들은 뒤에 간다."""
    nouns: list[Noun] = []
    onos: list[Ono] = []
    meta: list[str] = []
    for w in words:
        clean = hangul_only(w.text)
        raw_letters = re.sub(r"[\s\W_]+", "", w.text)
        if not clean:
            if raw_letters:
                meta.append(raw_letters)
            continue
        if is_blocked(clean):
            continue
        if is_onomatopoeia(clean, kiwi) and (not strict or _sure_ono(clean)):
            onos.append(Ono(clean, ono_role(clean), member, w.start, w.end))
            continue
        nsyl = len(clean)
        for t in kiwi.tokenize(clean):
            form = t.form
            fs, fe = t.start / nsyl, (t.start + t.len) / nsyl
            if is_onomatopoeia(form, kiwi) and len(form) >= 2 and (not strict or _sure_ono(form)):
                onos.append(Ono(form, ono_role(form), member, w.start, w.end, fs, fe))
            elif t.tag in ("NNG", "NNP"):
                if strict and not is_known(form):
                    continue
                if usable_noun(form, w.prob):
                    nouns.append(Noun(form, category_of(form), member, w.start, w.end, fs, fe))
            elif t.tag in ("SL", "SH", "UN"):
                meta.append(form)
    transcript = " ".join(w.text.strip() for w in words).strip()
    return Analysis(member, transcript, nouns, onos, meta)


def pick_nouns(analyses: list[Analysis], limit: int = 4) -> list[Noun]:
    """가족마다 번갈아 가며 명사를 고른다. 같은 단어는 한 번만, 최대 limit개."""
    queues = [list(a.nouns) for a in analyses]
    out: list[Noun] = []
    seen: set[str] = set()
    while len(out) < limit and any(queues):
        for q in queues:
            while q:
                n = q.pop(0)
                if n.text not in seen:
                    seen.add(n.text)
                    out.append(n)
                    break
            if len(out) >= limit:
                break
    return out


def pick_onos(analyses: list[Analysis]) -> dict[str, Ono]:
    """역할별로 대표 의성어 하나씩: main(킥 자리), melody(멜로디 자리), extra(그 밖)."""
    allo = [o for a in analyses for o in a.onos]
    out: dict[str, Ono] = {}
    for o in allo:
        if o.role == "kick" and "main" not in out:
            out["main"] = o
        elif o.role == "melody" and "melody" not in out:
            out["melody"] = o
    for o in allo:
        if o is out.get("main") or o is out.get("melody"):
            continue
        if "extra" not in out and (out.get("main") is None or o.text != out["main"].text):
            out["extra"] = o
            break
    if "main" not in out:
        rest = [o for o in allo if o is not out.get("melody")]
        if rest:
            out["main"] = rest[0]
            if out.get("extra") is rest[0]:
                out.pop("extra")
    return out
