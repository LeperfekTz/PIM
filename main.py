from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from pymongo import MongoClient
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from dotenv import load_dotenv
from datetime import datetime
import os
import requests
import uuid


# Carrega variáveis do .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config['SESSION_TYPE'] = 'filesystem'

# Configuração do Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

mail = Mail(app)
s = URLSafeTimedSerializer(app.secret_key)

# MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client['PIM']
usuarios_collection = db['PIM']
conversas_collection = db['conversas']

# Rotas
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
            session['email'] = email
            return redirect('/chat')
        else:
            flash('Usuário ou senha inválidos')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/chat')
def chat():
    if 'email' not in session:
        return redirect(url_for('login'))

    # Se não tiver um chat_id na sessão, cria novo
    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())
        conversas_collection.insert_one({
            "email": session['email'],
            "chat_id": session['chat_id'],
            "mensagens": [],
            "criado_em": datetime.now()
        })

    # Carrega mensagens da conversa ativa
    conversa = conversas_collection.find_one({
        "email": session['email'],
        "chat_id": session['chat_id']
    })

    mensagens = []
    if conversa and "mensagens" in conversa:
        for item in conversa["mensagens"]:
            mensagens.append({
                "usuario": item.get("pergunta", ""),
                "ia": item.get("resposta", "")
            })

    return render_template('chat.html', mensagens=mensagens)

@app.route('/novo_chat', methods=['POST'])
def novo_chat():
    novo_id = str(uuid.uuid4())
    session['chat_id'] = novo_id
    return redirect(url_for('chat'))


@app.route('/historico')
def historico():
    if 'email' not in session:
        return redirect(url_for('login'))

    # Buscar TODAS as conversas do usu rio
    conversas = conversas_collection.find({"email": session["email"]})

    mensagens = []
    for conversa in conversas:
        grupo = []  # Um grupo de mensagens = uma conversa
        for item in conversa.get("mensagens", []):
            grupo.append({
                "usuario": item.get("pergunta", "Pergunta n o encontrada"),
                "ia": item.get("resposta", "Sem resposta da IA")
            })
        if grupo:
            mensagens.append({
                "chat_id": str(conversa.get("chat_id")),
                "mensagens": grupo
            })

    return render_template("historico.html", mensagens=mensagens)


@app.route('/retomar/<string:chat_id>')
def retomar_conversa(chat_id):
    if 'email' not in session:
        return redirect(url_for('login'))

    conversa = conversas_collection.find_one({
        "chat_id": chat_id,
        "email": session["email"]
    })

    if not conversa:
        flash("Conversa n o encontrada.")
        return redirect(url_for('historico'))

    session['chat_id'] = chat_id
    return redirect(url_for('chat'))

@app.route('/readme')
def readme():
    return render_template('readme.html')


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
        email = s.loads(token, salt='reset-senha', max_age=3600)
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
    if 'email' not in session or 'chat_id' not in session:
        return jsonify({"erro": "Usuário não autenticado ou chat_id ausente."}), 403

    dados_recebidos = request.get_json()
    if not dados_recebidos or "mensagem" not in dados_recebidos:
        return jsonify({"erro": "Campo 'mensagem' é obrigatório."}), 400

    url = os.getenv('url')

    # PROMPT BASE usado só no início da conversa
    prompt_base = (
        "Você é uma IA de suporte técnico especializada em hardware e software.\n"
        "Seu objetivo é ajudar o usuário a identificar e resolver problemas com computadores, periféricos, impressoras, HDs, SSDs, Windows, drivers, instalação de programas, travamentos, lentidão, erros de inicialização, etc.\n\n"
        "Estilo de conversa:\n"
        "Fale como uma pessoa real e atenciosa, sem parecer um robô.\n"
        "Seja simpático, direto e profissional, como um técnico experiente que explica tudo numa boa.\n"
        "Converse com o usuário. Pergunte, ouça (analise a resposta), explique.\n"
        "Evite exageros como emojis em excesso ou frases forçadas. Use só quando for natural.\n"
        "Traga soluções de forma clara, passo a passo, sem jargões técnicos desnecessários.\n"
        "Quando houver mais de uma causa possível, comece pelas mais comuns.\n"
        "Sempre busque entender o que a pessoa já tentou e o contexto antes de sugerir algo.\n"
        "Use palavras fáceis e busque colocar explicação, pessoas que não sabem nada usarão você.\n\n"
    )

    # Montar contexto com histórico da conversa
    historico = conversas_collection.find_one({
        "chat_id": session["chat_id"],
        "email": session["email"]
    })

    contexto = prompt_base

    if historico and "mensagens" in historico:
        for msg in historico["mensagens"][-5:]:  # Limita a 5 interações anteriores
            contexto += f"Usuário: {msg['pergunta']}\nIA: {msg['resposta']}\n"

    # Adiciona a nova pergunta
    contexto += f"Usuário: {dados_recebidos['mensagem']}\nIA:"

    payload = {
        "input_value": contexto,
        "output_type": "chat",
        "input_type": "chat"
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        resposta_json = response.json()

        # Extrai a resposta final do modelo
        mensagem_resposta = resposta_json['outputs'][0]['outputs'][0]['outputs']['message']['message']

        # Atualiza o histórico no banco de dados
        conversas_collection.update_one(
            {"chat_id": session["chat_id"], "email": session["email"]},
            {"$push": {
                "mensagens": {
                    "pergunta": dados_recebidos["mensagem"],
                    "resposta": mensagem_resposta,
                    "timestamp": datetime.now()
                }
            }},
            upsert=True
        )

        return jsonify({"resposta": mensagem_resposta})

    except requests.exceptions.HTTPError as e:
        erro_resposta = e.response.text if e.response else "Sem resposta"
        return jsonify({"erro": f"Erro HTTP: {str(e)}", "detalhes": erro_resposta}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"erro": f"Erro na requisição: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
