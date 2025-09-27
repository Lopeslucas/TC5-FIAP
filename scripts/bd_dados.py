import json
import psycopg2
import boto3
import os

# Nome do bucket e "prefixo"
bucket_name = "bucket-tc5"
prefix = "bronze/"

# Nomes dos arquivos
caminho_arquivo_applicants = prefix + "applicants.json"
caminho_arquivo_vagas = prefix + "vagas.json"
caminho_arquivo_prospects = prefix + "prospects.json"

# Criar cliente S3
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

def carregar_json_do_s3(bucket, key):
    """Baixa e carrega JSON direto do S3"""
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

# Carregar os JSONs
dados_applicants = carregar_json_do_s3(bucket_name, caminho_arquivo_applicants)
dados_vagas = carregar_json_do_s3(bucket_name, caminho_arquivo_vagas)
dados_prospects = carregar_json_do_s3(bucket_name, caminho_arquivo_prospects)

# Conectar ao PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="Job_Base",
    user="postgres",
    password="123"
)
cursor = conn.cursor()

# ----------------- Helpers -----------------
def normalize(value):
    """Converte strings vazias e valores None para None (NULL no banco)."""
    if value in ("", None):
        return None
    return value

def tratar_data(data):
    """Converte datas inválidas ou vazias para None."""
    if not data or data.strip() in ["", "0000-00-00", "00-00-0000", "0000-00-00 00:00:00", "00-00-0000 00:00:00"]:
        return None
    return data

# ----------------- Inserções Applicants -----------------
def conexao_infos_basicas():
    for item in dados_applicants:
        infos = dados_applicants[item].get('infos_basicas', {})

        telefone_recado = normalize(infos.get('telefone_recado'))
        telefone = normalize(infos.get('telefone'))
        objetivo_profissional = normalize(infos.get('objetivo_profissional'))
        data_criacao = tratar_data(infos.get('data_criacao'))
        inserido_por = normalize(infos.get('inserido_por'))
        email = normalize(infos.get('email'))
        local = normalize(infos.get('local'))
        sabendo_de_nos_por = normalize(infos.get('sabendo_de_nos_por'))
        data_atualizacao = tratar_data(infos.get('data_atualizacao'))
        codigo_profissional = normalize(infos.get('codigo_profissional'))
        nome = normalize(infos.get('nome'))

        cursor.execute(
            """
            INSERT INTO applicants.infos_basicas (
                telefone_recado,
                telefone,
                objetivo_profissional,
                data_criacao,
                inserido_por,
                email,
                local,
                sabendo_de_nos_por,
                data_atualizacao,
                codigo_profissional,
                nome
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                telefone_recado,
                telefone,
                objetivo_profissional,
                data_criacao,
                inserido_por,
                email,
                local,
                sabendo_de_nos_por,
                data_atualizacao,
                codigo_profissional,
                nome
            )
        )

    conn.commit()
    print("Deu bom - infos_basicas!")

def conexao_informacoes_pessoais():
    for item in dados_applicants:
        info_pessoal = dados_applicants[item].get('informacoes_pessoais', {})

        data_aceite = tratar_data(info_pessoal.get('data_aceite'))
        nome = normalize(info_pessoal.get('nome'))
        cpf = normalize(info_pessoal.get('cpf'))
        fonte_indicacao = normalize(info_pessoal.get('fonte_indicacao'))
        email = normalize(info_pessoal.get('email'))
        email_secundario = normalize(info_pessoal.get('email_secundario'))
        data_nascimento = tratar_data(info_pessoal.get('data_nascimento'))
        telefone_celular = normalize(info_pessoal.get('telefone_celular'))
        telefone_recado = normalize(info_pessoal.get('telefone_recado'))
        sexo = normalize(info_pessoal.get('sexo'))
        estado_civil = normalize(info_pessoal.get('estado_civil'))
        pcd = normalize(info_pessoal.get('pcd'))
        endereco = normalize(info_pessoal.get('endereco'))
        skype = normalize(info_pessoal.get('skype'))
        url_linkedin = normalize(info_pessoal.get('url_linkedin'))
        facebook = normalize(info_pessoal.get('facebook'))

        cursor.execute(
            """
            INSERT INTO applicants.informacoes_pessoais (
                data_aceite,
                nome,
                cpf,
                fonte_indicacao,
                email,
                email_secundario,
                data_nascimento,
                telefone_celular,
                telefone_recado,
                sexo,
                estado_civil,
                pcd,
                endereco,
                skype,
                url_linkedin,
                facebook
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                data_aceite,
                nome,
                cpf,
                fonte_indicacao,
                email,
                email_secundario,
                data_nascimento,
                telefone_celular,
                telefone_recado,
                sexo,
                estado_civil,
                pcd,
                endereco,
                skype,
                url_linkedin,
                facebook
            )
        )
    conn.commit()
    print("Deu bom - informacoes_pessoais!")

def conexao_informacoes_profissionais():
    for item in dados_applicants:
        infos_prof = dados_applicants[item].get('informacoes_profissionais', {})

        titulo_profissional = normalize(infos_prof.get('titulo_profissional'))
        area_atuacao = normalize(infos_prof.get('area_atuacao'))
        conhecimentos_tecnicos = normalize(infos_prof.get('conhecimentos_tecnicos'))
        certificacoes = normalize(infos_prof.get('certificacoes'))
        outras_certificacoes = normalize(infos_prof.get('outras_certificacoes'))
        remuneracao = normalize(infos_prof.get('remuneracao'))
        nivel_profissional = normalize(infos_prof.get('nivel_profissional'))

        cursor.execute(
            """
            INSERT INTO applicants.informacoes_profissionais (
                titulo_profissional,
                area_atuacao,
                conhecimentos_tecnicos,
                certificacoes,
                outras_certificacoes,
                remuneracao,
                nivel_profissional
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                titulo_profissional,
                area_atuacao,
                conhecimentos_tecnicos,
                certificacoes,
                outras_certificacoes,
                remuneracao,
                nivel_profissional
            )
        )
    conn.commit()
    print("Deu bom - informacoes_profissionais!")

def conexao_formacao_e_idiomas():
    for item in dados_applicants:
        formacao = dados_applicants[item].get('formacao_e_idiomas', {})

        nivel_academico = normalize(formacao.get('nivel_academico'))
        nivel_ingles = normalize(formacao.get('nivel_ingles'))
        nivel_espanhol = normalize(formacao.get('nivel_espanhol'))
        outro_idioma = normalize(formacao.get('outro_idioma'))

        cursor.execute(
            """
            INSERT INTO applicants.formacao_e_idiomas (
                nivel_academico,
                nivel_ingles,
                nivel_espanhol,
                outro_idioma
            ) VALUES (%s, %s, %s, %s)
            """,
            (
                nivel_academico,
                nivel_ingles,
                nivel_espanhol,
                outro_idioma
            )
        )
    conn.commit()
    print("Deu bom - formacao_e_idiomas!")

def conexao_cargo_atual():
    for item in dados_applicants:
        cargo = dados_applicants[item].get("cargo_atual", {})

        id_ibrati = normalize(cargo.get("id_ibrati"))
        email_corporativo = normalize(cargo.get("email_corporativo"))
        cargo_atual = normalize(cargo.get("cargo_atual"))
        projeto_atual = normalize(cargo.get("projeto_atual"))
        cliente = normalize(cargo.get("cliente"))
        unidade = normalize(cargo.get("unidade"))
        data_admissao = tratar_data(cargo.get("data_admissao"))
        data_ultima_promocao = tratar_data(cargo.get("data_ultima_promocao"))
        nome_superior_imediato = normalize(cargo.get("nome_superior_imediato"))
        email_superior_imediato = normalize(cargo.get("email_superior_imediato"))

        cursor.execute(
            """
            INSERT INTO applicants.cargo_atual (
                id_ibrati,
                email_corporativo,
                cargo_atual,
                projeto_atual,
                cliente,
                unidade,
                data_admissao,
                data_ultima_promocao,
                nome_superior_imediato,
                email_superior_imediato
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                id_ibrati,
                email_corporativo,
                cargo_atual,
                projeto_atual,
                cliente,
                unidade,
                data_admissao,
                data_ultima_promocao,
                nome_superior_imediato,
                email_superior_imediato,
            ),
        )

    conn.commit()
    print("Deu bom - cargo_atual!")

def conexao_curriculos():
    for item in dados_applicants:
        cv_pt = normalize(dados_applicants[item].get("cv_pt"))
        cv_en = normalize(dados_applicants[item].get("cv_en"))

        cursor.execute(
            """
            INSERT INTO applicants.curriculos (cv_pt, cv_en)
            VALUES (%s, %s)
            """,
            (cv_pt, cv_en)
        )
    conn.commit()
    print("Deu bom - curriculos!")

# ----------------- Inserções Vagas -----------------

def conexao_informacoes_basicas():
    for item in dados_vagas:
        info = dados_vagas[item].get("informacoes_basicas", {})

        data_requicisao = tratar_data(info.get("data_requisicao"))
        limite_esperado_para_contratacao = tratar_data(info.get("limite_esperado_para_contratacao"))
        data_inicial = tratar_data(info.get("data_inicial"))  # campo opcional
        data_final = tratar_data(info.get("data_final"))      # campo opcional

        titulo_vaga = normalize(info.get("titulo_vaga"))
        vaga_sap = normalize(info.get("vaga_sap"))
        cliente = normalize(info.get("cliente"))
        solicitante_cliente = normalize(info.get("solicitante_cliente"))
        empresa_divisao = normalize(info.get("empresa_divisao"))
        requisitante = normalize(info.get("requisitante"))
        analista_responsavel = normalize(info.get("analista_responsavel"))
        tipo_contratacao = normalize(info.get("tipo_contratacao"))
        prazo_contratacao = normalize(info.get("prazo_contratacao"))
        objetivo_vaga = normalize(info.get("objetivo_vaga"))
        prioridade_vaga = normalize(info.get("prioridade_vaga"))
        origem_vaga = normalize(info.get("origem_vaga"))
        superior_imediato = normalize(info.get("superior_imediato"))
        nome = normalize(info.get("nome"))
        telefone = normalize(info.get("telefone"))

        cursor.execute(
            """
            INSERT INTO vagas.informacoes_basicas (
                data_requisicao,
                limite_esperado_para_contratacao,
                data_inicial,
                data_final,
                titulo_vaga,
                vaga_sap,
                cliente,
                solicitante_cliente,
                empresa_divisao,
                requisitante,
                analista_responsavel,
                tipo_contratacao,
                prazo_contratacao,
                objetivo_vaga,
                prioridade_vaga,
                origem_vaga,
                superior_imediato,
                nome,
                telefone
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                data_requicisao,
                limite_esperado_para_contratacao,
                data_inicial,
                data_final,
                titulo_vaga,
                vaga_sap,
                cliente,
                solicitante_cliente,
                empresa_divisao,
                requisitante,
                analista_responsavel,
                tipo_contratacao,
                prazo_contratacao,
                objetivo_vaga,
                prioridade_vaga,
                origem_vaga,
                superior_imediato,
                nome,
                telefone,
            ),
        )

    conn.commit()
    print("Deu bom - informacoes_basicas!")

def conexao_perfil_vaga():
    for item in dados_vagas:
        perfil = dados_vagas[item].get("perfil_vaga", {})

        pais = normalize(perfil.get("pais"))
        estado = normalize(perfil.get("estado"))
        cidade = normalize(perfil.get("cidade"))
        bairro = normalize(perfil.get("bairro"))
        regiao = normalize(perfil.get("regiao"))
        local_trabalho = normalize(perfil.get("local_trabalho"))
        vaga_especifica_para_pcd = normalize(perfil.get("vaga_especifica_para_pcd"))
        faixa_etaria = normalize(perfil.get("faixa_etaria"))
        horario_trabalho = normalize(perfil.get("horario_trabalho"))
        nivel_profissional = normalize(perfil.get("nivel profissional"))
        nivel_academico = normalize(perfil.get("nivel_academico"))
        nivel_ingles = normalize(perfil.get("nivel_ingles"))
        nivel_espanhol = normalize(perfil.get("nivel_espanhol"))
        outro_idioma = normalize(perfil.get("outro_idioma"))
        areas_atuacao = normalize(perfil.get("areas_atuacao"))
        principais_atividades = normalize(perfil.get("principais_atividades"))
        competencia_tecnicas_e_comportamentais = normalize(perfil.get("competencia_tecnicas_e_comportamentais"))
        demais_observacoes = normalize(perfil.get("demais_observacoes"))
        viagens_requeridas = normalize(perfil.get("viagens_requeridas"))
        equipamentos_necessarios = normalize(perfil.get("equipamentos_necessarios"))

        cursor.execute(
            """
            INSERT INTO vagas.perfil_vaga (
                pais,
                estado,
                cidade,
                bairro,
                regiao,
                local_trabalho,
                vaga_especifica_para_pcd,
                faixa_etaria,
                horario_trabalho,
                nivel_profissional,
                nivel_academico,
                nivel_ingles,
                nivel_espanhol,
                outro_idioma,
                areas_atuacao,
                principais_atividades,
                competencia_tecnicas_e_comportamentais,
                demais_observacoes,
                viagens_requeridas,
                equipamentos_necessarios
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                pais,
                estado,
                cidade,
                bairro,
                regiao,
                local_trabalho,
                vaga_especifica_para_pcd,
                faixa_etaria,
                horario_trabalho,
                nivel_profissional,
                nivel_academico,
                nivel_ingles,
                nivel_espanhol,
                outro_idioma,
                areas_atuacao,
                principais_atividades,
                competencia_tecnicas_e_comportamentais,
                demais_observacoes,
                viagens_requeridas,
                equipamentos_necessarios,
            ),
        )

    conn.commit()
    print("Deu bom - perfil_vaga!")

def conexao_beneficios():
    for item in dados_vagas:
        beneficios = dados_vagas[item].get("beneficios", {})

        valor_venda = normalize(beneficios.get("valor_venda"))
        valor_compra_1 = normalize(beneficios.get("valor_compra_1"))
        valor_compra_2 = normalize(beneficios.get("valor_compra_2"))

        cursor.execute(
            """
            INSERT INTO vagas.beneficios (
                valor_venda,
                valor_compra_1,
                valor_compra_2
            ) VALUES (%s, %s, %s)
            """,
            (
                valor_venda,
                valor_compra_1,
                valor_compra_2,
            ),
        )

    conn.commit()
    print("Deu bom - beneficios!")

# ----------------- Inserções Prospects -----------------

def conexao_prospects():
    for item in dados_prospects:
        vaga = dados_prospects[item]
        titulo = normalize(vaga.get("titulo"))
        modalidade = normalize(vaga.get("modalidade"))
        prospects = vaga.get("prospects", [])

        for prospect in prospects:
            nome = normalize(prospect.get("nome"))
            codigo = normalize(prospect.get("codigo"))
            situacao_candidado = normalize(prospect.get("situacao_candidado"))
            data_candidatura = tratar_data(prospect.get("data_candidatura"))
            ultima_atualizacao = tratar_data(prospect.get("ultima_atualizacao"))
            comentario = normalize(prospect.get("comentario"))
            recrutador = normalize(prospect.get("recrutador"))

            cursor.execute(
                """
                INSERT INTO prospects.prospects (
                    titulo_vaga,
                    modalidade,
                    nome,
                    codigo,
                    situacao_candidado,
                    data_candidatura,
                    ultima_atualizacao,
                    comentario,
                    recrutador
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    titulo,
                    modalidade,
                    nome,
                    codigo,
                    situacao_candidado,
                    data_candidatura,
                    ultima_atualizacao,
                    comentario,
                    recrutador,
                ),
            )

    conn.commit()
    print("Deu bom - prospects!")

# ----------------- Execução -----------------
conexao_infos_basicas()
conexao_informacoes_pessoais()
conexao_informacoes_profissionais()
conexao_formacao_e_idiomas()
conexao_cargo_atual()
conexao_curriculos()

conexao_informacoes_basicas()
conexao_perfil_vaga()
conexao_beneficios()

conexao_prospects()

# Fechar conexão
cursor.close()
conn.close()

print("Dados inseridos com sucesso!")
