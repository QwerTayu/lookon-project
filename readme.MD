# .env に以下を記述
```
SECRET_KEY=
DEBUG=

```

# 以下の手順で環境構築（たぶん）
プロジェクト直下に移動しておくこと。
```
python -m venv env
source env/Scripts/activate
pip install -r requirements.txt
python manage.py makemigrations lookon_app
python manage.py migrate
python manage.py runserver
```

# 指定のURLに移動
http://localhost:8000/
など
