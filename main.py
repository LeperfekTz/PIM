from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pymongo import MongoClient
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from dotenv import load_dotenv
import os
import requests

# Carrega as variáveis do .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Configuração do Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

mail = Mail(app)
s = URLSafeTimedSerializer(app.secret_key)

# Conexão com MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client['PIM']
usuarios_collection = db['PIM']

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        senha = request.form['senha']
        user = usuarios_collection.find_one({'email': email, 'senha': senha})

        if user:
            return redirect('/chat')
        else:
            flash('Usuário ou senha inválidos')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nome = request.form['nome']
        email = request.form['email']
        senha = request.form['senha']

        if usuarios_collection.find_one({'nome': nome}):
            flash('Usuário já existe.')
            return redirect(url_for('register'))

        if usuarios_collection.find_one({'email': email}):
            flash('E-mail já cadastrado.')
            return redirect(url_for('register'))

        usuarios_collection.insert_one({'nome': nome, 'email': email, 'senha': senha})
        flash('Cadastro realizado com sucesso!')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/reset', methods=['GET', 'POST'])
def reset():
    if request.method == 'POST':
        email = request.form.get('email')
        user = usuarios_collection.find_one({'email': email})

        if user:
            token = s.dumps(email, salt='reset-senha')
            link = url_for('redefinir_senha', token=token, _external=True)
            msg = Message("Redefinir sua senha", sender=app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = f"Clique no link para redefinir sua senha: {link}"
            mail.send(msg)
            flash('Um link de redefinição foi enviado para seu e-mail.')
        else:
            flash('E-mail não encontrado.')

        return redirect(url_for('login'))

    return render_template('reset.html')

@app.route('/redefinir-senha/<token>', methods=['GET', 'POST'])
def redefinir_senha(token):
    try:
        email = s.loads(token, salt='reset-senha', max_age=3600)  # expira em 1 hora
    except SignatureExpired:
        flash('O link expirou. Solicite uma nova redefinição.')
        return redirect(url_for('reset'))
    except BadSignature:
        flash('Token inválido.')
        return redirect(url_for('reset'))

    if request.method == 'POST':
        nova_senha = request.form.get('senha')
        usuarios_collection.update_one({'email': email}, {'$set': {'senha': nova_senha}})
        flash('Senha atualizada com sucesso.')
        return redirect(url_for('login'))

    return render_template('reset_pass.html')

@app.route('/executar-api', methods=['POST'])
def executar_api():
    dados_recebidos = request.get_json()

    if not dados_recebidos or "mensagem" not in dados_recebidos:
        return jsonify({"erro": "Requisição inválida. Campo 'mensagem' é obrigatório."}), 400

    url = "http://127.0.0.1:7860/api/v1/run/2f745828-aa27-49d8-afd7-c2c88683aa92"

    payload = {
        "input_value": dados_recebidos.get("mensagem", "Mensagem padrão"),
        "output_type": "chat",
        "input_type": "chat"
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        # Verifica se a resposta está no formato esperado
        resposta_json = response.json()
        return jsonify({"resposta": resposta_json})

    except requests.exceptions.HTTPError as e:
        erro_resposta = e.response.text if e.response else "Sem resposta"
        return jsonify({
            "erro": f"Erro HTTP: {str(e)}",
            "detalhes": erro_resposta
        }), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"erro": f"Erro na requisição: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
