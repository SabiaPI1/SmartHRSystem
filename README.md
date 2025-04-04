# SmartHRSystem: Интеллектуальная система подбора кандидатов 

**SmartHRSystem** — проект, предназначенный для автоматического сопоставления вакансий и кандидатов (специалистов) на основе анализа текстовых данных. Система использует современные методы обработки естественного языка (NLP) и векторного поиска для определения наиболее релевантных кандидатов для каждой вакансии, учитывая навыки, опыт и семантическую близость описаний.

Цель проекта — предоставить HR-специалистам эффективный инструмент для первичного отбора кандидатов, экономя время и повышая точность подбора.

## Контекст проекта: Разработка для Iconicompany

Проект SmartHRSystem был разработан по заказу компании **ООО 'ЯКомпания' (Iconicompany)**, платформы умного аутстаффинга с ИИ-ассистентом.

## Ключевые возможности 

*   **Загрузка данных:** Автоматическая загрузка и парсинг данных о навыках, вакансиях и специалистах из JSON-файлов.
*   **Нормализация текста:** Приведение текстовых данных (навыки, описания) к единому формату, включая удаление лишних символов, приведение к нижнему регистру и стемминг (для русского и английского языков).
*   **Обработка синонимов:** Учет различных вариантов написания и синонимов для одних и тех же навыков (например, "postgresql", "postgres" -> "sql").
*   **Извлечение требований к опыту:** Автоматическое распознавание и извлечение требований к опыту работы с определенными навыками из текста вакансии (например, "python от 3 лет").
*   **Векторизация текста:** Преобразование текстовых описаний вакансий, резюме и навыков в векторные представления (эмбеддинги) с помощью модели Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`).
*   **Эффективный поиск:** Использование библиотеки FAISS для быстрого поиска семантически близких кандидатов по векторному представлению вакансии.
*   **Извлечение навыков из вакансий:** Идентификация релевантных навыков в тексте вакансии с использованием комбинации методов:
    *   Прямое совпадение с базой данных навыков.
    *   Семантический поиск похожих навыков на основе эмбеддингов.
    *   Распознавание именованных сущностей (NER) с помощью spaCy (опционально).
*   **Комплексный скоринг:** Расчет итоговой оценки релевантности кандидата на основе взвешенной суммы факторов:
    *   Семантическая близость (FAISS score).
    *   Процент совпадения навыков (с учетом прямых совпадений, синонимов и семантически близких навыков).
    *   Соответствие требованиям к опыту.
*   **Детализация совпадений:** Предоставление подробной информации о совпавших (прямо, по синонимам, семантически) и отсутствующих навыках, а также о соответствии опыта.
*   **Кэширование:** Использование кэширования для ускорения вычислений (эмбеддинги, нормализация навыков).

## Технологический стек

*   **Python 3.x**
*   **Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)**
*   **FAISS**
*   **KeyBERT** (для извлечения ключевых слов/навыков на основе эмбеддингов)
*   **spaCy (`ru_core_news_sm`)**
*   **NLTK** (**SnowballStemmer** для стемминга, **`punkt`** для токенизации)
*   **Scikit-learn (`cosine_similarity`)**
*   **NumPy**
*   **python-dateutil**

## Использование бота

### **Команды:**
- **/start** - Начало работы
- **/find** - Поиск кандидатов
- **/help** - Справка

## Возможные улучшения 

*   Разработка веб-интерфейса для более удобного взаимодействия.
*   Добавление возможности тонкой настройки весовых коэффициентов через интерфейс или конфигурационный файл.
*   Дообучение моделей Sentence Transformers и spaCy на специфичных для домена данных для повышения точности.
*   Оптимизация производительности для обработки очень больших объемов данных.
*   Интеграция с внешними HR-платформами.
