#!pip install aiogram python-dotenv

import os
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import CommandStart, Command
from aiogram.types import BotCommand, ReplyKeyboardRemove
from dotenv import find_dotenv, load_dotenv

import nest_asyncio
import asyncio
import json
import numpy as np
import faiss

# Применяем nest_asyncio
nest_asyncio.apply()

# Загрузка переменных окружения
load_dotenv(find_dotenv())

# Инициализация бота и диспетчера
bot = Bot(token="BOT_TOKEN")
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Загрузка данных и моделей
job_data = None
specialists_data = None
skill_db = None
normalized_skill_map = None
original_skill_map_norm = None
model = None  # SentenceTransformer модель
faiss_index = None  # FAISS индекс

# Состояния для FSM (Finite State Machine)
class Form(StatesGroup):
    waiting_for_requirements = State()

# Команды бота
private_commands = [
    BotCommand(command='start', description='Старт'),
    BotCommand(command='help', description='Помощь'),
    BotCommand(command='find', description='Найти кандидатов'),
]

async def initialize_system():
    """Инициализация системы - загрузка данных и моделей"""
    global job_data, specialists_data, skill_db, normalized_skill_map, original_skill_map_norm, model, faiss_index

    # Загрузка данных
    job_data, specialists_data, skill_db, normalized_skill_map, original_skill_map_norm = load_data()

    # Инициализация моделей
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    # Подготовка FAISS индекса
    faiss_index = prepare_faiss_index(specialists_data)

# Обработчик команды /start
@dp.message(CommandStart())
async def start_cmd(message: types.Message):
    await message.answer(
        "Привет! Я бот для поиска кандидатов. "
        "Отправь мне требования вакансии, и я найду подходящих специалистов.\n"
        "Используй команду /find чтобы начать поиск."
    )

# Обработчик команды /help
@dp.message(Command('help'))
async def help_cmd(message: types.Message):
    commands_list = "\n".join([f"/{cmd.command} - {cmd.description}" for cmd in private_commands])
    await message.answer(f"Доступные команды:\n{commands_list}")

# Обработчик команды /find
@dp.message(Command('find'))
async def find_cmd(message: types.Message, state: FSMContext):
    await state.set_state(Form.waiting_for_requirements)
    await message.answer(
        "Пожалуйста, введите требования вакансии. "
        "Например: 'Опыт работы с Python, знание SQL.'"
    )

def format_candidate_info(candidate):
    """Форматирование информации о кандидате для вывода"""
    info = (
        f"👤 <b>{candidate['name']}</b>\n"
        f"📊 Общий балл: {candidate['combined_score']:.2f}\n"
        f"🔍 Совпадение навыков: {candidate['_scores']['skill']}%\n"
        f"📈 FAISS score: {candidate['_scores']['faiss']:.4f}\n"
    )

    if candidate['matching_skills']['direct']:
        info += f"✅ Прямые совпадения: {', '.join(candidate['matching_skills']['direct'])}\n"

    if candidate['matching_skills']['synonym']:
        syns = [f"{k}→{v}" for k, v in candidate['matching_skills']['synonym'].items()]
        info += f"🔄 Синонимы: {', '.join(syns)}\n"

    if candidate['missing_skills']:
        info += f"❌ Отсутствующие навыки: {', '.join(candidate['missing_skills'])}\n"

    return info

# Обработчик текстовых сообщений (требований вакансии)
@dp.message(Form.waiting_for_requirements)
async def process_requirements(message: types.Message, state: FSMContext):
    requirements = message.text

    # Проверяем инициализацию системы
    if not all([job_data, specialists_data, model, faiss_index]):
        await message.answer("Система еще не готова. Пожалуйста, попробуйте позже.")
        await state.clear()
        return

    try:
        # 1. Создаем временную запись вакансии
        job_id = f"user_{message.from_user.id}_{int(time.time())}"
        job_info = {
            'name': "Пользовательская вакансия",
            'full_text': requirements,
            'required_experience': parse_experience_requirements_from_text(requirements, normalized_skill_map, skill_db)
        }

        # 2. Извлекаем навыки из требований
        req_skills = extract_job_skills_advanced(
            requirements,
            skill_db,
            normalized_skill_map,
            nlp,
            get_cached_skill_embeddings(tuple(sorted(skill_db.items())), model)

        )

        # 3. Поиск кандидатов через FAISS
        faiss_scores, faiss_indices = search_candidates_faiss(faiss_index, requirements, k=100)

        if not faiss_indices:
            await message.answer("Подходящих кандидатов не найдено.")
            await state.clear()
            return

        # 4. Обработка кандидатов
        processed_candidates = []
        for i in range(len(faiss_indices)):
            idx = int(faiss_indices[i])
            if idx < 0 or idx >= len(specialists_data):
                continue

            spec = specialists_data[idx]

            # Проверка соответствия навыков
            skill_details = calculate_match_details(
                req_skills,
                spec['skills_list_norm'],
                list(job_info['required_experience'].keys()),
                original_skill_map_norm)

            # Проверка соответствия опыта
            exp_score, met_exp, unmet_exp = calculate_experience_match(
                job_info['required_experience'],
                spec['experience_per_skill_months'])

            # Расчет общего балла
            skill_score_norm = skill_details['match_percent'] / 100.0
            safe_f_score = max(0.0, float(faiss_scores[i]))
            comb_score = (W_FAISS * safe_f_score + W_SKILL * skill_score_norm + W_EXPERIENCE * exp_score)

            # Форматирование данных для вывода
            def fmt_s_dict(md):
                return {original_skill_map_norm.get(k,{}).get('ru',k):
                        original_skill_map_norm.get(v,{}).get('ru',v)
                        for k,v in md.items()}

            def fmt_s_list(nl):
                return sorted(list(set(original_skill_map_norm.get(n,{}).get('ru', n)
                                   for n in nl if n)))

            processed_candidates.append({
                'id': spec['id'],
                'name': spec['name'],
                '_scores': {
                    'faiss': round(safe_f_score, 4),
                    'skill': skill_details['match_percent'],
                    'experience': exp_score
                },
                'combined_score': round(comb_score, 4),
                'matching_skills': {
                    'direct': fmt_s_list(skill_details['matched_direct']),
                    'synonym': fmt_s_dict(skill_details['matched_synonym']),
                    'semantic': fmt_s_dict(skill_details['matched_semantic'])
                },
                'missing_skills': fmt_s_list(skill_details['missing']),
            })

        # 5. Сортировка и выбор топ-5 кандидатов
        top_candidates = sorted(processed_candidates,
                               key=lambda x: x['combined_score'],
                               reverse=True)[:5]

        # 6. Формирование ответа
        if not top_candidates:
            await message.answer("Подходящих кандидатов не найдено.")
        else:
            response = "🔍 <b>Найденные кандидаты:</b>\n\n"
            for candidate in top_candidates:
                response += format_candidate_info(candidate) + "\n"

            await message.answer(response, parse_mode="HTML")

    except Exception as e:
        logging.error(f"Ошибка обработки требований: {e}")
        await message.answer("Произошла ошибка при обработке ваших требований. Пожалуйста, попробуйте еще раз.")

    await state.clear()

# Запуск бота
async def main():
    # Инициализация системы перед запуском бота
    await initialize_system()

    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_my_commands(commands=private_commands,
                             scope=types.BotCommandScopeAllPrivateChats())
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())