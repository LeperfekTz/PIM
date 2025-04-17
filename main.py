import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from pymongo import MongoClient
from flask_mail import Mail, Message
from dotenv import load_dotenv
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import openai
from collections import deque

# Carrega variáveis do .env
load_dotenv()

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
token = URLSafeTimedSerializer(app.secret_key)

# MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client['PIM']
usuarios_collection = db['usuarios']
conversas_collection = db['conversas']
perguntas_respostas_collection = db['perguntas_respostas']

print("Testando conexão com MongoDB...")
print(perguntas_respostas_collection.count_documents({}), "documentos encontrados.")

# ______________________________________________________________________________________________________________________________________________________________


# Exemplo básico de memória (em dicionário, por enquanto)
memoria_curta = {}

# Funçoes 
def atualizar_memoria(usuario_id, pergunta, resposta, limite=5):
    if usuario_id not in memoria_curta:
        memoria_curta[usuario_id] = deque(maxlen=limite)
    memoria_curta[usuario_id].append({"pergunta": pergunta, "resposta": resposta})

# Função para buscar perguntas e respostas no MongoDB
def buscar_resposta_no_banco(mensagem):
    # Busca no banco de dados por perguntas semelhantes à mensagem do usuário
    resultado = perguntas_respostas_collection.find_one({
        "Customer_Issue": {"$regex": f"^{mensagem}$", "$options": "i"}  # Busca exata ou similar
    })
    
    if resultado:
        return resultado.get("Tech_Response")  # Retorna a resposta se encontrada
    return None  # Retorna None se não encontrar nenhuma correspondência

# Função que usa OpenAI para responder com base nos dados do banco
def usar_openai_com_base_no_banco(pergunta_usuario):
    documentos = list(perguntas_respostas_collection.find(
        {"Customer_Issue": {"$exists": True}, "Tech_Response": {"$exists": True}}
    ).limit(20))

    contexto = ""
    for doc in documentos:
        pergunta = doc.get('Customer_Issue') or doc.get('pergunta')
        resposta = doc.get('Tech_Response') or doc.get('resposta')

        if pergunta and resposta:
            contexto += f"Problema: {pergunta}\nIA: {resposta}\n"

    if not contexto:
        return "Não encontrei informações no banco para responder."

    prompt = f"""
    Responda com base apenas nas informações a seguir.

    {contexto}

    Pergunta do usuário: {pergunta_usuario}
    """

    resposta = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    return resposta.choices[0].message["content"]

# Função para obter a resposta (primeiro tenta buscar no banco, senão usa o OpenAI)
def obter_resposta(usuario_id, pergunta_usuario, limite_memoria=5):
    try:
        # Tenta buscar no banco
        resposta = buscar_resposta_no_banco(pergunta_usuario)

        if resposta:
            atualizar_memoria(usuario_id, pergunta_usuario, resposta, limite=limite_memoria)
            return resposta

        # Se não encontrar, usa a OpenAI
        resposta = usar_openai_com_base_no_banco(pergunta_usuario)
        print(f"Resposta gerada pela OpenAI: {resposta}")  # Debug

        # Atualiza a memória curta com a nova resposta
        atualizar_memoria(usuario_id, pergunta_usuario, resposta, limite=limite_memoria)

        # Verificando os valores antes de inserir no banco
        print(f"Inserindo no banco a pergunta: {pergunta_usuario} e a resposta: {resposta}")
        try:
            result = perguntas_respostas_collection.insert_one({
                "Customer_Issue": pergunta_usuario,
                "Tech_Response": resposta
            })
            print(f"Resposta inserida com sucesso! ID: {result.inserted_id}")
        except Exception as e:
            print(f"Erro ao salvar pergunta e resposta no banco: {e}")

        return resposta

    except Exception as e:
        print(f"Erro ao obter ou salvar a resposta: {e}")
        return "Ocorreu um erro ao processar sua solicitação."
    
# ______________________________________________________________________________________________________________________________________________________________

# Rotas
@app.route('/')
def index():
    return redirect(url_for('login'))

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
            pergunta = item.get("pergunta", "").strip()
            resposta = item.get("resposta", "").strip()
            mensagens.append({
                "usuario": pergunta,
                "ia": resposta
            })

    return render_template('chat.html', mensagens=mensagens)

@app.route('/perguntar', methods=['POST'])
def perguntar():
    pergunta = request.form.get('pergunta')

    if not pergunta:
        return jsonify({"resposta": "Por favor, envie uma pergunta válida."}), 400

    usuario_id = session.get('email', 'anonimo')
    resposta = obter_resposta(usuario_id, pergunta)

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
        flash("Conversa não encontrada.")
        return redirect(url_for('historico'))

    # Salva o chat_id atual na sessão
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
async def executar_api():
    if 'email' not in session or 'chat_id' not in session:
        return jsonify({"resposta": "Sessão expirada. Faça login novamente."}), 401

    data = request.get_json()
    mensagem = data.get("mensagem", "").strip()

    if not mensagem:
        return jsonify({"resposta": "Por favor, envie uma mensagem válida."}), 400

    # Gera a resposta usando a base do banco
    resposta_final = usar_openai_com_base_no_banco(mensagem)
    
    # Salva a resposta com IA e pergunta no banco
    perguntas_respostas_collection.insert_one({
        "usuario": "OPENAI",
        "pergunta": mensagem,
        "resposta": resposta_final,
    })

    # Salva a conversa no banco
    conversas_collection.update_one(
        {"email": session['email'], "chat_id": session['chat_id']},
        {"$push": {
            "mensagens": {
                "pergunta": mensagem,
                "resposta": resposta_final,
                "hora": datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            }
        }},
        upsert=True  # cria o documento se não existir
    )

    return jsonify({"resposta": resposta_final})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

