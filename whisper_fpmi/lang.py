"""Токены, служебные слова и леммы для датасета, облака и поиска."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё+]{1,}")
_CYR = re.compile(r"^[а-я]+$")
_LATIN = re.compile(r"^[a-z0-9+]+$")
_VOWELS = set("аеиоуыэюя")

# Окончания от длинных к коротким. Стем короче 4 букв не отрезаем.
_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "ее",
    "ие",
    "ые",
    "ое",
    "ая",
    "яя",
    "ой",
    "ий",
    "ый",
    "ую",
    "юю",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ам",
    "ям",
    "ом",
    "ем",
)

_STOP_TEXT = """
а ага без был была были было быть в во вот все всё всего всем всему всех всю вся вы где да даже для до его едва ее ей ему если есть еще ж же за из или им именно их к ко когда кого кто куда ли либо мне мной мог может можно мое мой моя мы на над нам нами нас нашему не него нее ней нем нему нет ни них ничего но ну о об обо он она они оно от ото перед по подо после потом потому почему почти при про пусть разве с со свой своя свое свои себе себя собой та так такая такие таким таких такого такое таком такую те тем теми тех то тогда того тоже только том тому тут ты у уже вам вас ведь весь вся что чтобы чтоб чем чему чему это эта эти этим этих этого этом этому этот я
или же бы ли ль вот вон ну да нет уже еще тоже также только даже именно просто вообще конечно например здесь там тут сегодня сейчас теперь потом тогда спасибо пожалуйста здравствуйте добрый
который которая которое которые которого которой котором которую которых которыми какой какая какое какие какого какой какую каких каким какими чей чья чье чьи
меня мне мной тебя тебе тобой вас вам вами его ему им ее ей ею их ими нас нам нами себя себе собой
меня тебя него нее них ней ним
этот эта это эти этого этому этим этом эту этой этих этими
тот та то те того тому тем том ту той тех теми
весь вся все всего всему всем всю всех всеми
мой моя мое мои моего моему моим моем мою моей моих моими
твой твоя твое твои твоего твоему твоим твоем твою твоей твоих твоими
наш наша наше наши нашего нашему нашим нашем нашу нашей наших нашими
ваш ваша ваше ваши вашего вашему вашим вашем вашу вашей ваших вашими
свой своя свое свои своего своему своим своем свою своей своих своими
такой такая такое такие такого такому таким таком такую такой таких такими
таков такова таково таковы
сей сия сие сии
чей чья чье чьи чьего чьему чьим чьем чью чьей чьих чьими
сам сама само сами самого самому самим самом саму самой самих самими
этот эта этот
будет будут есть был была было были быть
можно нужно надо нельзя
очень уже еще тоже также только даже именно просто вообще
потому поэтому однако хотя если чтобы чтоб будто
между через около изза
"""

_FILLER_TEXT = """
значит давайте давай итак короче типа собственно соответственно ладно понятно ясно хорошо
так ну вот ага угу слушайте смотрите допустим скажем видите повторюсь повторяю короче
собственно итак следовательно
"""

_FILLER_BIGRAMS = {
    ("как", "бы"),
    ("то", "есть"),
    ("в", "общем"),
    ("на", "самом"),
    ("в", "принципе"),
    ("ну", "вот"),
    ("ну", "так"),
}


def fold(token: str) -> str:
    return token.casefold().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def stem_token(token: str) -> str:
    word = fold(token)
    cached = _STEM_CACHE.get(word)
    if cached is not None:
        return cached
    _STEM_CACHE[word] = _strip(word)
    return _STEM_CACHE[word]


def _strip(word: str) -> str:
    if len(word) <= 4 or _CYR.fullmatch(word) is None:
        return word
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            return word[: -len(ending)]
    return word


_STEM_CACHE: dict[str, str] = {}

STOPWORDS = {fold(word) for word in _STOP_TEXT.split() if word}
FILLER_UNIGRAMS = {fold(word) for word in _FILLER_TEXT.split() if word}


def _closed(words: set[str], lemmatize) -> set[str]:
    closed = set(words)
    for word in words:
        closed.add(lemmatize(word))
    return closed


@dataclass(frozen=True)
class TextStats:
    tokens: int
    function_words: int
    fillers: int
    content: int
    counts: dict[str, int]
    filler_counts: dict[str, int]

    @property
    def lexical_density(self) -> float:
        if not self.tokens:
            return 0.0
        return self.content / self.tokens

    def per_1k(self, count: int) -> float | None:
        if not self.tokens:
            return None
        return 1000.0 * count / self.tokens


def _is_short_term(word: str) -> bool:
    if len(word) < 2 or word.isdigit():
        return False
    if _LATIN.fullmatch(word) is not None:
        return True
    if _CYR.fullmatch(word) is not None and len(word) >= 3:
        return not any(char in _VOWELS for char in word)
    return False


def analyze(text: str, lemmatize=stem_token) -> TextStats:
    """Считает слова, служебные, слова-паразиты и частоты содержательных лемм."""
    surfaces = tokenize(text)
    folded = [fold(token) for token in surfaces]
    stop = _closed(STOPWORDS, lemmatize)
    filler_lemmas = _closed(FILLER_UNIGRAMS, lemmatize)
    consumed: set[int] = set()
    filler_counts: Counter[str] = Counter()
    index = 0
    while index < len(folded) - 1:
        pair = (folded[index], folded[index + 1])
        if pair in _FILLER_BIGRAMS:
            filler_counts[" ".join(pair)] += 1
            consumed.add(index)
            consumed.add(index + 1)
            index += 2
            continue
        index += 1

    function_count = 0
    counts: Counter[str] = Counter()
    for index, word in enumerate(folded):
        if index in consumed or not word or len(word) > 40:
            continue
        lemma = lemmatize(word)
        if word in FILLER_UNIGRAMS or lemma in filler_lemmas:
            filler_counts[word if word in FILLER_UNIGRAMS else lemma] += 1
            continue
        if word in STOPWORDS or lemma in stop:
            function_count += 1
            continue
        if word.isdigit():
            continue
        if len(lemma) >= 4 or _is_short_term(word):
            key = word if _is_short_term(word) and len(lemma) < 4 else lemma
            counts[key] += 1
    content = int(sum(counts.values()))
    return TextStats(
        tokens=len(folded),
        function_words=function_count,
        fillers=int(sum(filler_counts.values())),
        content=content,
        counts=dict(counts),
        filler_counts=dict(filler_counts),
    )


def get_lemmatizer():
    """Возвращает (функция, имя). pymorphy3, если пакет установлен."""
    try:
        import pymorphy3
    except ImportError:
        return stem_token, "suffix"
    morph = pymorphy3.MorphAnalyzer()
    cache: dict[str, str] = {}

    def lemmatize(token: str) -> str:
        word = fold(token)
        cached = cache.get(word)
        if cached is not None:
            return cached
        if _CYR.fullmatch(word) is None:
            cache[word] = word
            return word
        parsed = morph.parse(word)
        lemma = fold(parsed[0].normal_form) if parsed else stem_token(word)
        cache[word] = lemma
        return lemma

    return lemmatize, "pymorphy3"


_GENERIC = {
    "начало",
    "начал",
    "конец",
    "вступление",
    "вступлен",
    "перерыв",
    "вопрос",
    "вопросы",
    "заключение",
    "заключен",
    "титры",
    "пауза",
    "пауз",
    "антракт",
}


def usable_phrase(text: str, lemmatize=stem_token) -> bool:
    """Фраза годится как запрос: не пустая и не одно короткое слово."""
    if fold(text.strip(" .:-–—")) in _GENERIC:
        return False
    stats = analyze(text, lemmatize)
    lemmas = [lemma for lemma, count in stats.counts.items() for _ in range(count)]
    unique = list(dict.fromkeys(lemmas))
    unique = [
        lemma
        for lemma in unique
        if lemma not in _GENERIC and not lemma.isdigit()
    ]
    if not unique:
        return False
    if len(unique) >= 2:
        return True
    return len(unique[0]) >= 6
