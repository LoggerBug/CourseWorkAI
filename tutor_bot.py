import logging
import re
import pymorphy3
import json
from typing import List, Dict, Optional, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, Message
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    CallbackContext,
    CommandHandler,
    CallbackQueryHandler
)
from g4f.client import Client
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import os

# Инициализация логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Инициализация клиента g4f и морфологического анализатора
client = Client()
morph = pymorphy3.MorphAnalyzer()

# История сообщений, контекст и корзина
user_history: Dict[int, List[Dict]] = {}
user_context: Dict[int, Dict[str, Optional[str]]] = {}
user_cart: Dict[int, List[int]] = {}
user_menu_context: Dict[int, str] = {}

# Принудительное использование CPU
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Загрузка репетиторов
def load_tutors(filename: str) -> List[Dict]:
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"Файл {filename} не найден")
        return []

tutors = load_tutors("tutors.json")

def load_dialogs(filename: str) -> Dict[str, str]:
    dialogues = {}
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line and '#' in line:
                    user_text, bot_text = line.split('#', 1)
                    user_text = user_text.strip()
                    bot_text = bot_text.strip()
                    if bot_text:
                        dialogues[user_text] = bot_text
                        logger.info(f"Loaded dialogue: {user_text} -> {bot_text}")
    except FileNotFoundError:
        logger.error(f"Файл {filename} не найден")
    return dialogues

dialogues = load_dialogs("dialogues.txt")

# Функция предобработки текста
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    lemmatized = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized)

# Подготовка данных для классификации
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
X = list(dialogues.keys())
X_embeddings = embedder.encode(X, convert_to_tensor=True, device=device)
logger.info(f"X_embeddings device: {X_embeddings.device}")

for orig in X:
    proc = preprocess_text(orig)
    logger.info(f"Original: {orig}, Preprocessed: {proc}")

# Функция классификации намерений
def classify_intent(user_message: str, confidence_threshold: float = 0.7) -> Tuple[Optional[str], float, Optional[str]]:
    try:
        preprocessed_message = preprocess_text(user_message)
        logger.info(f"User message: {user_message}, Preprocessed: {preprocessed_message}")
        user_embedding = embedder.encode(user_message, convert_to_tensor=True, device=device)
        cos_scores = util.pytorch_cos_sim(user_embedding, X_embeddings)[0]
        best_score = np.max(cos_scores.cpu().numpy())
        best_match_idx = np.argmax(cos_scores.cpu().numpy())
        if best_score >= confidence_threshold:
            return list(dialogues.values())[best_match_idx], best_score, list(dialogues.keys())[best_match_idx]
        return None, best_score, None
    except Exception as e:
        logger.error(f"Ошибка в classify_intent: {e}")
        return None, 0.0, None

# Загрузка начальной подсказки
try:
    with open('base.txt', 'r', encoding="UTF-8") as f:
        initial_prompt = f.read()
except FileNotFoundError:
    logger.error("Файл base.txt не найден")
    initial_prompt = "Ты бот, помогающий выбрать репетиторов. Отвечай кратко, профессионально, учитывая предыдущие сообщения в диалоге. После уточнения предмета и уровня показывай подходящих репетиторов автоматически. Предлагай добавить их в корзину через кнопки. Для работы с корзиной предлагай кнопку 'Работа с корзиной' в меню. Используй ссылки в формате [название](ссылка). Не используй эмодзи и избегай лишних деталей."

# Функция фильтрации репетиторов
def filter_tutors(subject: Optional[str] = None, level: Optional[str] = None, format: Optional[str] = None) -> List[Dict]:
    filtered = []
    for tutor in tutors:
        if subject and tutor["subject"] != subject:
            continue
        if level and level.lower() not in tutor["level"].lower():
            continue
        if format and tutor["format"] != format:
            continue
        filtered.append(tutor)
    return filtered

# Функция отображения репетиторов
def display_tutors(tutors_list: List[Dict]) -> str:
    if not tutors_list:
        return "Подходящих репетиторов не найдено. Попробуйте изменить запрос."
    response = "Доступные репетиторы:\n\n"
    for tutor in tutors_list:
        response += f"ID: {tutor['id']}\n"
        response += f"Имя: {tutor['name']}\n"
        response += f"Предмет: {tutor['subject']}\n"
        response += f"Уровень: {tutor['level']}\n"
        response += f"Формат: {tutor['format']}\n"
        response += f"Стоимость: {tutor['price_per_hour']} руб./час\n\n"
    return response

# Функция создания клавиатуры для репетиторов
def create_tutors_keyboard(tutors_list: List[Dict]) -> InlineKeyboardMarkup:
    buttons = [[InlineKeyboardButton(f"{tutor['name']} ({tutor['subject']})", callback_data=f"add_tutor_{tutor['id']}")] for tutor in tutors_list]
    buttons.append([InlineKeyboardButton("Скрыть меню", callback_data="hide_menu")])
    return InlineKeyboardMarkup(buttons)

# Функция создания единой кнопки "Меню" с "Подобрать репетитора"
def create_menu_button() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("Меню", callback_data="show_menu")],
        [InlineKeyboardButton("Подобрать репетитора", callback_data="start_tutor_selection")]
    ]
    return InlineKeyboardMarkup(keyboard)

# Функция создания главного меню
def create_base_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("Показать репетиторов", callback_data="show_tutors")],
        [InlineKeyboardButton("Работа с корзиной", callback_data="cart_menu")],
        [InlineKeyboardButton("Сохранить диалог", callback_data="save_dialog")],
        [InlineKeyboardButton("Советы по обучению", callback_data="learning_tips")],
        [InlineKeyboardButton("Скрыть меню", callback_data="hide_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

# Функция создания меню корзины
def create_cart_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("Посмотреть корзину", callback_data="view_cart")],
        [InlineKeyboardButton("Очистить корзину", callback_data="clear_cart")],
        [InlineKeyboardButton("Оплатить корзину", callback_data="pay_cart")],
        [InlineKeyboardButton("Вернуться в главное меню", callback_data="back_to_main")],
        [InlineKeyboardButton("Скрыть меню", callback_data="hide_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

# Функция создания клавиатуры с предметами
def create_subjects_keyboard() -> InlineKeyboardMarkup:
    unique_subjects = sorted(set(tutor["subject"] for tutor in tutors))
    buttons = [[InlineKeyboardButton(subject, callback_data=f"select_subject_{subject}")] for subject in unique_subjects]
    buttons.append([InlineKeyboardButton("Скрыть меню", callback_data="hide_menu")])
    return InlineKeyboardMarkup(buttons)

# Функция создания клавиатуры с уровнями
def create_levels_keyboard(subject: str) -> InlineKeyboardMarkup:
    unique_levels = sorted(set(tutor["level"] for tutor in tutors if tutor["subject"] == subject))
    buttons = [[InlineKeyboardButton(level, callback_data=f"select_level_{level}")] for level in unique_levels]
    buttons.append([InlineKeyboardButton("Скрыть меню", callback_data="hide_menu")])
    return InlineKeyboardMarkup(buttons)

# Функция для показа репетиторов на основе контекста
async def show_tutors_from_context(user_id: int, query: CallbackQuery) -> None:
    subject = user_context.get(user_id, {}).get("subject")
    level = user_context.get(user_id, {}).get("level")
    format = None
    filtered_tutors = filter_tutors(subject, level, format)
    bot_reply = display_tutors(filtered_tutors)
    user_history[user_id].append({"role": "assistant", "content": bot_reply})
    await query.message.reply_text(bot_reply, reply_markup=create_tutors_keyboard(filtered_tutors))

# Функция обработки кнопок меню
async def handle_menu_buttons(user_id: int, data: str, query: CallbackQuery) -> None:
    user_menu_context[user_id] = "main"
    if data == "show_menu":
        menu_context = user_menu_context.get(user_id, "main")
        reply_markup = create_base_keyboard() if menu_context == "main" else create_cart_keyboard()
        bot_reply = "Выберите действие:" if menu_context == "main" else "Выберите действие с корзиной:"
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=reply_markup)
    elif data == "hide_menu":
        bot_reply = "Меню скрыто. Нажмите 'Меню' для просмотра действий."
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_menu_button())

# Функция обработки кнопок главного меню
async def handle_main_menu_buttons(user_id: int, data: str, query: CallbackQuery) -> None:
    user_menu_context[user_id] = "main"
    if data == "show_tutors":
        await show_tutors_from_context(user_id, query)
    elif data == "cart_menu":
        user_menu_context[user_id] = "cart"
        bot_reply = "Выберите действие с корзиной:"
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_cart_keyboard())
    elif data == "save_dialog":
        if user_id in user_history and len(user_history[user_id]) > 1:
            filename = f"conversation_{user_id}.txt"
            with open(filename, "w", encoding="utf-8") as file:
                for message in user_history[user_id]:
                    if message['role'] != "system":
                        role = message['role'].capitalize()
                        content = message['content']
                        file.write(f"{role}: {content}\n\n")
            with open(filename, 'rb') as file:
                await query.message.reply_document(document=file, filename=filename)
            os.remove(filename)
            bot_reply = "Диалог сохранен."
        else:
            bot_reply = "История диалога пуста."
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_menu_button())
    elif data == "learning_tips":
        bot_reply = "Вот несколько советов по обучению: планируйте занятия, делайте заметки, повторяйте материал регулярно."
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_menu_button())

# Функция обработки кнопок меню корзины
async def handle_cart_menu_buttons(user_id: int, data: str, query: CallbackQuery) -> None:
    user_menu_context[user_id] = "main"
    if data == "view_cart":
        response = "Ваша корзина:\n\n"
        if user_id not in user_cart or not user_cart[user_id]:
            response = "Ваша корзина пуста."
        else:
            total_price = 0
            for tutor_id in user_cart[user_id]:
                tutor = next((t for t in tutors if t["id"] == tutor_id), None)
                if tutor:
                    response += f"Имя: {tutor['name']}\n"
                    response += f"Предмет: {tutor['subject']}\n"
                    response += f"Уровень: {tutor['level']}\n"
                    response += f"Формат: {tutor['format']}\n"
                    response += f"Стоимость: {tutor['price_per_hour']} руб./час\n\n"
                    total_price += tutor["price_per_hour"]
            response += f"Общая стоимость: {total_price} руб.\n"
        user_history[user_id].append({"role": "assistant", "content": response})
        await query.message.reply_text(response, reply_markup=create_menu_button())
    elif data == "clear_cart":
        user_cart[user_id] = []
        bot_reply = "Корзина очищена."
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_menu_button())
    elif data == "pay_cart":
        if user_id not in user_cart or not user_cart[user_id]:
            bot_reply = "Ваша корзина пуста."
        else:
            user_cart[user_id] = []
            bot_reply = "Оплата прошла успешно! Корзина очищена."
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_menu_button())
    elif data == "back_to_main":
        user_menu_context[user_id] = "main"
        bot_reply = "Вернулись в главное меню."
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_menu_button())

# Функция обработки добавления репетитора
async def handle_tutor_addition(user_id: int, data: str, query: CallbackQuery) -> None:
    user_menu_context[user_id] = "main"
    tutor_id = int(data.split("_")[-1])
    if user_id not in user_cart:
        user_cart[user_id] = []
    if tutor_id not in user_cart[user_id]:
        user_cart[user_id].append(tutor_id)
        tutor = next((t for t in tutors if t["id"] == tutor_id), None)
        bot_reply = f"Репетитор {tutor['name']} ({tutor['subject']}) добавлен в корзину."
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_menu_button())
    else:
        await query.message.reply_text("Этот репетитор уже в корзине.", reply_markup=create_menu_button())

# Функция обработки выбора предмета и уровня
async def handle_tutor_selection(user_id: int, data: str, query: CallbackQuery) -> None:
    user_menu_context[user_id] = "main"
    if data == "start_tutor_selection":
        user_context[user_id]["selection_step"] = "subject"
        user_context[user_id]["selected_subject"] = None
        bot_reply = "Выберите предмет:"
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_subjects_keyboard())
    elif data.startswith("select_subject_"):
        subject = data[len("select_subject_"):]
        user_context[user_id]["selected_subject"] = subject
        user_context[user_id]["selection_step"] = "level"
        bot_reply = f"Вы выбрали {subject}. Выберите уровень:"
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_levels_keyboard(subject))
    elif data.startswith("select_level_"):
        level = data[len("select_level_"):]
        subject = user_context[user_id].get("selected_subject")
        user_context[user_id]["selection_step"] = None
        user_context[user_id]["subject"] = subject
        user_context[user_id]["level"] = level
        filtered_tutors = filter_tutors(subject, level)
        bot_reply = display_tutors(filtered_tutors)
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await query.message.reply_text(bot_reply, reply_markup=create_tutors_keyboard(filtered_tutors))

# Инициализация пользователя
def initialize_user(user_id: int) -> None:
    user_history[user_id] = [{"role": "system", "content": initial_prompt}]
    user_context[user_id] = {"subject": None, "level": None, "selection_step": None, "selected_subject": None}
    user_cart[user_id] = []
    user_menu_context[user_id] = "main"

# Обновление контекста
def update_context(user_id: int, subject: Optional[str], level: Optional[str]) -> None:
    if user_id not in user_context:
        user_context[user_id] = {"subject": None, "level": None, "selection_step": None, "selected_subject": None}
    if subject:
        user_context[user_id]["subject"] = subject
    if level:
        user_context[user_id]["level"] = level
    logger.info(f"Updated context for user {user_id}: {user_context[user_id]}")

# Проверка связи с контекстом
def is_related_to_previous_context(user_message: str, user_id: int) -> Tuple[bool, Optional[str], Optional[str]]:
    user_message_lower = user_message.lower()
    subjects = ['физика', 'математика', 'химия', 'биология', 'география', 'история', 'английский', 'русский', 'литература', 'программирование', 'музыка', 'шахматы', 'французский', 'немецкий']
    levels = ['егэ', 'огэ', 'старшая школа', 'начинающие']
    message_subject = None
    for s in subjects:
        if s in user_message_lower:
            message_subject = s
            logger.info(f"Found subject '{s}' in is_related_to_previous_context: {user_message_lower}")
            break
    message_level = next((l for l in levels if l in user_message_lower), None)
    if message_subject or message_level:
        return True, message_subject, message_level
    return False, None, None

# Обработка запросов на репетиторов
async def process_tutor_request(user_id: int, user_message: str, context_subject: Optional[str], context_level: Optional[str], update: Update) -> None:
    subjects = ['физика', 'математика', 'химия', 'биология', 'география', 'история', 'английский', 'русский', 'литература', 'программирование', 'музыка', 'шахматы', 'французский', 'немецкий']
    subject = context_subject or next((s for s in subjects if s in user_message.lower()), None)
    level = context_level or next((l for l in ['егэ', 'огэ', 'старшая школа', 'начинающие'] if l in user_message.lower()), None)
    if not subject:
        logger.warning(f"No subject found in process_tutor_request for message: {user_message}")
    if not level:
        logger.warning(f"No level found in process_tutor_request for message: {user_message}")
    format = None
    filtered_tutors = filter_tutors(subject, level, format)
    bot_reply = display_tutors(filtered_tutors)
    user_history[user_id].append({"role": "user", "content": user_message})
    user_history[user_id].append({"role": "assistant", "content": bot_reply})
    user_menu_context[user_id] = "main"
    await update.message.reply_text(bot_reply, reply_markup=create_tutors_keyboard(filtered_tutors))

# Обработка запросов на корзину
async def process_cart_request(user_id: int, user_message: str, update: Update) -> None:
    bot_reply = "Нажмите 'Меню' и выберите 'Работа с корзиной' для управления корзиной."
    user_history[user_id].append({"role": "user", "content": user_message})
    user_history[user_id].append({"role": "assistant", "content": bot_reply})
    user_menu_context[user_id] = "main"
    await update.message.reply_text(bot_reply, reply_markup=create_menu_button())

# Генерация ответа нейросети
async def generate_neural_response(user_id: int, user_message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=user_history[user_id],
            web_search=False
        )
        bot_reply = response.choices[0].message.content
        if not bot_reply:
            bot_reply = "Извините, я не получил ответа от нейросети. Попробуйте еще раз."
    except Exception as e:
        logger.error(f"Ошибка API: {e}")
        bot_reply = "Извините, я не могу ответить прямо сейчас. Попробуйте переформулировать запрос или написать позже."
    return bot_reply

# Извлечение ссылок
async def extract_links(text: str) -> List[Tuple[str, Optional[str]]]:
    link_pattern = r'(https?://[^\s]+)(?:\s*-\s*([^\n]+))?'
    return re.findall(link_pattern, text)

# Замена ссылок в тексте
async def replace_links_in_text(text: str, links: List[Tuple[str, Optional[str]]]) -> str:
    for i, (link, name) in enumerate(links):
        replacement = f"Вариант {i+1} - {name if name else 'Ссылка'}"
        text = text.replace(f"{link} - {name}" if name else link, replacement)
    return text

# Создание клавиатуры для ссылок
async def create_keyboard(links: List[Tuple[str, Optional[str]]]) -> InlineKeyboardMarkup:
    buttons = [[InlineKeyboardButton(name if name else "Ссылка", url=link)] for link, name in links]
    return InlineKeyboardMarkup(buttons)

# Отправка ответа
async def send_response(user_id: int, bot_reply: str, update: Update) -> None:
    user_menu_context[user_id] = "main"
    links = await extract_links(bot_reply)
    if links:
        bot_reply = await replace_links_in_text(bot_reply, links)
        keyboard = await create_keyboard(links)
        await update.message.reply_text(bot_reply, reply_markup=keyboard)
    else:
        await update.message.reply_text(bot_reply, reply_markup=create_menu_button())

# Функция для команды /start
async def start(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    initialize_user(user_id)
    reply_markup = create_menu_button()
    await update.message.reply_text(
        "Привет! Я бот, который поможет найти репетитора или даст советы по обучению. "
        "Напиши предмет и уровень, чтобы увидеть репетиторов, или нажми 'Меню':",
        reply_markup=reply_markup
    )

# Функция для начала нового диалога
async def start_new_dialog(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    initialize_user(user_id)
    reply_markup = create_menu_button()
    await update.message.reply_text(
        "Новый диалог начат. Напиши предмет и уровень, чтобы увидеть репетиторов, или нажми 'Меню':",
        reply_markup=reply_markup
    )

# Функция для просмотра корзины
async def view_cart(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    user_menu_context[user_id] = "cart"
    if user_id not in user_cart or not user_cart[user_id]:
        await update.message.reply_text("Ваша корзина пуста.", reply_markup=create_menu_button())
        return
    response = "Ваша корзина:\n\n"
    total_price = 0
    for tutor_id in user_cart[user_id]:
        tutor = next((t for t in tutors if t["id"] == tutor_id), None)
        if tutor:
            response += f"Имя: {tutor['name']}\n"
            response += f"Предмет: {tutor['subject']}\n"
            response += f"Уровень: {tutor['level']}\n"
            response += f"Формат: {tutor['format']}\n"
            response += f"Стоимость: {tutor['price_per_hour']} руб./час\n\n"
            total_price += tutor["price_per_hour"]
    response += f"Общая стоимость: {total_price} руб.\n"
    await update.message.reply_text(response, reply_markup=create_menu_button())

# Функция для очистки корзины
async def clear_cart(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    user_menu_context[user_id] = "cart"
    user_cart[user_id] = []
    await update.message.reply_text("Корзина очищена.", reply_markup=create_menu_button())

# Функция для оплаты корзины
async def pay_cart(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    user_menu_context[user_id] = "cart"
    if user_id not in user_cart or not user_cart[user_id]:
        await update.message.reply_text("Ваша корзина пуста.", reply_markup=create_menu_button())
        return
    user_cart[user_id] = []
    await update.message.reply_text("Оплата прошла успешно! Корзина очищена.", reply_markup=create_menu_button())

# Функция для скачивания диалога
async def download_conversation(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    user_menu_context[user_id] = "main"
    if user_id in user_history and len(user_history[user_id]) > 1:
        filename = f"conversation_{user_id}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            for message in user_history[user_id]:
                if message['role'] != "system":
                    role = message['role'].capitalize()
                    content = message['content']
                    file.write(f"{role}: {content}\n\n")
        with open(filename, 'rb') as file:
            await update.message.reply_document(document=file, filename=filename)
        os.remove(filename)
        await update.message.reply_text("Диалог сохранен.", reply_markup=create_menu_button())
    else:
        await update.message.reply_text("История диалога пуста.", reply_markup=create_menu_button())

# Обработчик для кнопок
BUTTON_HANDLERS = {
    "show_menu": handle_menu_buttons,
    "hide_menu": handle_menu_buttons,
    "show_tutors": handle_main_menu_buttons,
    "cart_menu": handle_main_menu_buttons,
    "save_dialog": handle_main_menu_buttons,
    "learning_tips": handle_main_menu_buttons,
    "view_cart": handle_cart_menu_buttons,
    "clear_cart": handle_cart_menu_buttons,
    "pay_cart": handle_cart_menu_buttons,
    "back_to_main": handle_cart_menu_buttons,
    "start_tutor_selection": handle_tutor_selection,
}

async def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data
    if data.startswith("add_tutor_"):
        await handle_tutor_addition(user_id, data, query)
    elif data.startswith("select_subject_") or data.startswith("select_level_"):
        await handle_tutor_selection(user_id, data, query)
    elif data in BUTTON_HANDLERS:
        await BUTTON_HANDLERS[data](user_id, data, query)
    else:
        logger.warning(f"Неизвестная команда кнопки: {data}")

# Основная функция для обработки сообщений
async def handle_message(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    user_message = update.message.text
    if user_id not in user_history:
        initialize_user(user_id)
    if len(user_history[user_id]) > 10:
        user_history[user_id] = [{"role": "system", "content": initial_prompt}] + user_history[user_id][-9:]
    
    # Проверка связи с предыдущим контекстом
    is_related, context_subject, context_level = is_related_to_previous_context(user_message, user_id)
    
    # Определение предмета и уровня из сообщения
    subjects = ['физика', 'математика', 'химия', 'биология', 'география', 'история', 'английский', 'русский', 'литература', 'программирование', 'музыка', 'шахматы', 'французский', 'немецкий']
    levels = ['егэ', 'огэ', 'старшая школа', 'начинающие']
    user_message_lower = user_message.lower()
    message_subject = None
    for s in subjects:
        if s in user_message_lower:
            message_subject = s
            logger.info(f"Found subject '{s}' in handle_message: {user_message_lower}")
            break
    if not message_subject:
        logger.warning(f"No subject found in handle_message: {user_message_lower}, subjects checked: {subjects}")
    message_level = next((l for l in levels if l in user_message_lower), None)
    
    # Отладка
    logger.info(f"Message subject: {message_subject}, Message level: {message_level}")
    
    # Показ репетиторов, если есть предмет и уровень
    if message_subject and message_level:
        update_context(user_id, message_subject, message_level)
        logger.info(f"Showing tutors for user {user_id}: subject={message_subject}, level={message_level}")
        await process_tutor_request(user_id, user_message, message_subject, message_level, update)
        return
    
    # Обновление контекста
    update_context(user_id, message_subject, message_level)
    
    # Классификация намерения
    classified_intent, confidence, matched_intent = classify_intent(user_message)
    logger.info(f"User {user_id} sent: {user_message}, Intent: {classified_intent}, Confidence: {confidence}, Matched Intent: {matched_intent}, Context: subject={context_subject}, level={context_level}, User Context: {user_context[user_id]}")
    
    # Обработка явных запросов на репетиторов
    if classified_intent and any(x in classified_intent.lower() for x in ["показать репетиторов", "карточки репетиторов"]):
        await process_tutor_request(user_id, user_message, context_subject, context_level, update)
        return
    
    # Обработка запросов на корзину
    if classified_intent and any(x in classified_intent.lower() for x in ["показать корзину", "посмотреть корзину", "что в корзине", "очистить корзину", "удалить все из корзины", "оплатить", "оплатить корзину"]):
        await process_cart_request(user_id, user_message, update)
        return
    
    # Обработка классифицированного намерения с высокой уверенностью
    if classified_intent and confidence >= 0.7:
        bot_reply = classified_intent
        user_history[user_id].append({"role": "user", "content": user_message})
        user_history[user_id].append({"role": "assistant", "content": bot_reply})
        await send_response(user_id, bot_reply, update)
        return
    
    # Использование нейросети для неклассифицированных запросов
    user_history[user_id].append({"role": "user", "content": user_message})
    bot_reply = await generate_neural_response(user_id, user_message)
    user_history[user_id].append({"role": "assistant", "content": bot_reply})
    
    if not bot_reply or bot_reply.strip() == "":
        bot_reply = "Извините, я не смог сформировать ответ. Пожалуйста, попробуйте еще раз."
        logger.warning(f"Пустой ответ для сообщения: {user_message}")
    
    await send_response(user_id, bot_reply, update)

# Функция для запуска бота
def main() -> None:
    TOKEN = '7809241077:AAEcHFs_INk1EJK0AD__sRR6wjGn3wJfqfg'
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('new_bot', start_new_dialog))
    app.add_handler(CommandHandler('download_conv', download_conversation))
    app.add_handler(CommandHandler('cart', view_cart))
    app.add_handler(CommandHandler('clear_cart', clear_cart))
    app.add_handler(CommandHandler('pay', pay_cart))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))
    print("Бот запущен...")
    app.run_polling()

if __name__ == '__main__':
    main()