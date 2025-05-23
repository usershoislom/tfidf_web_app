# TF-IDF Analyzer

Это веб-приложение на FastAPI, которое позволяет анализировать текстовые файлы и рассчитывать TF-IDF для слов из загруженного текста. Интерфейс позволяет загружать `.txt` файлы и визуализирует результат в виде таблицы.

## 🚀 Возможности

- Загрузка одного или нескольких `.txt` файлов
- Подсчет TF, IDF и TF-IDF значений
- Отображение таблицы с результатами
- Поддержка русского и английского языков

## 🔧 Запуск проекта

### 📦 Вариант 1: Запуск с помощью Docker

1. Клонируйте себе репозиторий:
   ```bash
    git clone git@github.com:usershoislom/tfidf_web_app.git   
    cd tfidf_web_app

2. Соберите образ:
   ```bash
   docker buildx build -t tfidf-analyzer .

3. Запустите контейнер:
   ```bash
   docker run -p 8081:8000 tfidf-analyzer uvicorn main:app --host 0.0.0.0 --port 8000
   
4. Перейдите в браузере:

   ```bash
   http://localhost:8081
   

### Вариант 2: Запуск без Docker (локально)

1. Клонируйте себе репозиторий:
   ```bash
    git clone git@github.com:usershoislom/tfidf_web_app.git   
    cd tfidf_web_app
   
2. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python3 -m venv venv
   source venv/bin/activate # .\venv\Scripts\activate для Windows
   pip install -r requirements.txt

3. Запустите приложение:
   ```bash
   python main.py

4. Откройте в браузере:
   ```bash
   http://127.0.0.1:8000