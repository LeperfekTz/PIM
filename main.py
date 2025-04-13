from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from pymongo import MongoClient
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from dotenv import load_dotenv
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
import json
import numpy as np
import uuid
from openai import OpenAI
import random




# Carrega modelo treinado: clf (classificador), vectorizer (TF-IDF), e dados
with open('chatbot_model.pkl', 'rb') as f:
    clf, vectorizer, data = pickle.load(f)

# Carrega os intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents_data = json.load(f)

# Carrega variáveis do .env
load_dotenv()

#openai 
client_openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),  # se quiser
    project=os.getenv("OPENAI_PROJECT_ID")     # se for chave de projeto
)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config['SESSION_TYPE'] = 'filesystem'

# Configuração do e-Mail
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
usuarios_collection = db['usuarios']
conversas_collection = db['conversas']
colecao_memoria = db["memorias"]


def salvar_memoria(usuario_id, tag):
    colecao_memoria.update_one(
        {"usuario_id": usuario_id},
        {"$set": {"ultima_tag": tag, "atualizado_em": datetime.utcnow()}},
        upsert=True
    )

def obter_memoria(usuario_id):
    memoria = colecao_memoria.find_one({"usuario_id": usuario_id})
    return memoria.get("ultima_tag") if memoria else None

# Funções
def processar_imagem_com_ia(caminho):
    try:
        resposta = client_openai.image.create(
            prompt="",
            n=1,
            size="1024x1024"
        )
        return resposta["data"][0]["url"]
    except Exception as e:
        print("Erro ao processar imagem com IA:", e)
        return "Tive um problema ao tentar processar a imagem. Pode tentar de novo?"
    
def prever_tag(pergunta, debug=False):
    entrada = vectorizer.transform([pergunta])
    probas = clf.predict_proba(entrada)[0]
    sorted_indices = np.argsort(probas)[::-1]
    tag_prevista = clf.classes_[sorted_indices[0]]
    confianca = probas[sorted_indices[0]]

    if debug:
        top_tags = [(clf.classes_[i], probas[i]) for i in sorted_indices[:3]]
        print("Top 3 predições:", top_tags)

    if confianca < 0.6:
        return "sem_resposta"
    return tag_prevista
    
def obter_respostas_por_tag(tag, dados):
    for intent in dados.get("intents", []):
        if intent["tag"] == tag:
            return intent.get("responses", [])
    return []


# Exemplo básico de memória (em dicionário, por enquanto)
memoria_curta = {}

def gerar_resposta_com_memoria(usuario_id, pergunta, modelo, palavras, dados):
    tag_predita = prever_tag(pergunta)

    # Se não souber a resposta, tenta com OpenAI
    if tag_predita == "sem_resposta":
        contexto = memoria_curta.get(usuario_id, "sem contexto")
        return usar_openai(pergunta, contexto=f"Último assunto: {contexto}")

    # Atualiza o contexto
    memoria_curta[usuario_id] = tag_predita

    # Exemplo: resposta com hora
    if tag_predita == "perguntar_horas":
        hora_atual = datetime.now().strftime("%H:%M")
        respostas = obter_respostas_por_tag(tag_predita, dados)
        return random.choice(respostas).replace("{hora_atual}", hora_atual)

    respostas = obter_respostas_por_tag(tag_predita, dados)
    return random.choice(respostas) if respostas else usar_openai(pergunta)

def usar_openai(pergunta, contexto=""):
    try:
        system_prompt = f"Você é um assistente de tecnologia. Ajude de forma natural. {contexto}"
        resposta = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pergunta}
            ],
            temperature=0.6,
            max_tokens=150
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        print("Erro ao usar OpenAI:", e)
        return "Tive um problema ao tentar responder. Pode tentar de novo?"

    
def salvar_interacao(pergunta, resposta, origem, mongo_collection):
    mongo_collection.insert_one({
        "pergunta": pergunta,
        "resposta": resposta,
        "origem": origem,  # "modelo_local" ou "openai"
        "timestamp": datetime.utcnow()
    })



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

            # Cria novo chat após login
            session['chat_id'] = str(uuid.uuid4())
            conversas_collection.insert_one({
                "email": email,
                "chat_id": session['chat_id'],
                "mensagens": [],
                "criado_em": datetime.now()
            })

            return redirect('/chat')
        else:
            flash('Usuário ou senha inválidos')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/chat')
def chat():
    if 'email' not in session:
        return redirect(url_for('login'))

    conversa = conversas_collection.find_one({
        "email": session['email'],
        "chat_id": session.get('chat_id')
    })

    mensagens = []
    if conversa and "mensagens" in conversa:
        for item in conversa["mensagens"]:
            mensagens.append({
                "usuario": item.get("pergunta", ""),
                "ia": item.get("resposta", "")
            })

    return render_template('chat.html', mensagens=mensagens)

@app.route('/perguntar', methods=['POST'])
def perguntar():
    pergunta = request.form.get('pergunta')
    resposta = gerar_resposta_com_memoria(pergunta, clf, vectorizer.get_feature_names_out(), data)

    # Salva no banco
    if 'chat_id' in session:
        conversas_collection.update_one(
            {"chat_id": session['chat_id']},
            {"$push": {"mensagens": {"pergunta": pergunta, "resposta": resposta, "hora": datetime.now().strftime('%d/%m/%Y %H:%M:%S')}}}
        )

    return jsonify({"resposta": resposta})

@app.route('/novo_chat', methods=['POST'])
def novo_chat():
    novo_id = str(uuid.uuid4())
    session['chat_id'] = novo_id

    # Cria novo documento de conversa no MongoDB
    conversas_collection.insert_one({
        "email": session['email'],
        "chat_id": novo_id,
        "mensagens": [{
        "pergunta": "Chat iniciado.",
        "resposta": "Olá! Como posso te ajudar?",
        "hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        }],
        "criado_em": datetime.now()
    })

    return redirect(url_for('chat'))

@app.route('/upload_imagem', methods=['POST'])
def upload_imagem():

    # print("request.files =>", request.files)
    # print("request.form =>", request.form)

    if 'file' not in request.files:
        return "Nenhum arquivo enviado", 400

    imagem = request.files['file']
    if imagem.filename == '':
        return "Nenhum arquivo selecionado", 400
    
    UPLOAD_FOLDER = 'static/uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    caminho = os.path.join(app.config['UPLOAD_FOLDER'], imagem.filename)
    imagem.save(caminho)

    imagem_url = url_for('static', filename=f'uploads/{imagem.filename}')
    resposta_ia = processar_imagem_com_ia(caminho)

    conversas_collection.update_one(
    {"chat_id": session["chat_id"], "email": session["email"]},
    {"$push": {
        "mensagens": {
            "pergunta": f"[Imagem enviada: {imagem.filename}]",
            "resposta": resposta_ia
        }
    }}
)

    return render_template('chat.html', imagem_url=imagem_url, resposta=resposta_ia)

@app.route('/historico')
def historico():
    if 'email' not in session:
        return redirect(url_for('login'))

    # Buscar TODAS as conversas do usuario
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

    novo_id = str(uuid.uuid4())
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
    #login
    if 'email' not in session or 'chat_id' not in session:
        return jsonify({"resposta": "Sessão expirada. Faça login novamente."}), 401
    # Recebe a mensagem data
    data = request.get_json()
    mensagem = data.get("mensagem", "")

    if not mensagem.strip():
        return jsonify({"resposta": "Por favor, envie uma mensagem válida."})

    # Processa com o modelo
    X_input = vectorizer.transform([mensagem])
    tag_predita = clf.predict(X_input)[0]

    resposta = "Desculpe, não entendi."

    # Busca a resposta correspondente
    for intent in intents_data["intents"]:
        if intent["tag"] == tag_predita:
            if tag_predita == "horas":
                resposta = f"Agora são {datetime.now().strftime('%H:%M')}"
            else:
                resposta = random.choice(intent["responses"])
            break

    pergunta = mensagem

    # Salva no banco de dados
    conversas_collection.update_one(
        {"email": session['email'], "chat_id": session['chat_id']},
        {"$push": {
            "mensagens": {
                "pergunta": pergunta,
                "resposta": resposta,
                "hora": datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            }
        }}
    )

    return jsonify({"resposta": resposta})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  


