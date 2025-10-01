CREATE TABLE vagas.informacoes_basicas (
    id SERIAL PRIMARY KEY,
    data_requisicao DATE,
    limite_esperado_para_contratacao DATE,
    data_inicial DATE, 
    data_final DATE, 
    titulo_vaga VARCHAR(255),
    vaga_sap VARCHAR(150),
    cliente VARCHAR(150),
    solicitante_cliente VARCHAR(150),
    empresa_divisao VARCHAR(150),
    requisitante VARCHAR(150),
    analista_responsavel VARCHAR(150),
    tipo_contratacao VARCHAR(150),
    prazo_contratacao VARCHAR(150),
    objetivo_vaga TEXT,
    prioridade_vaga VARCHAR(150),
    origem_vaga VARCHAR(100),
    superior_imediato VARCHAR(150),
    nome VARCHAR(150),
    telefone VARCHAR(50)
);

CREATE TABLE vagas.perfil_vaga (
    id SERIAL PRIMARY KEY,
    pais VARCHAR(150),
    estado VARCHAR(150),
    cidade VARCHAR(150),
    bairro VARCHAR(150),
    regiao VARCHAR(150),
    local_trabalho VARCHAR(150),
    vaga_especifica_para_pcd VARCHAR(150),
    faixa_etaria VARCHAR(150),
    horario_trabalho VARCHAR(150),
    nivel_profissional VARCHAR(100),
    nivel_academico VARCHAR(100),
    nivel_ingles VARCHAR(150),
    nivel_espanhol VARCHAR(150),
    outro_idioma VARCHAR(150),
    areas_atuacao TEXT,
    principais_atividades TEXT,
    competencia_tecnicas_e_comportamentais TEXT,
    demais_observacoes TEXT,
    viagens_requeridas VARCHAR(100),
    equipamentos_necessarios VARCHAR(100)
);

CREATE TABLE vagas.beneficios (
    id SERIAL PRIMARY KEY,
    valor_venda VARCHAR(150),
    valor_compra_1 VARCHAR(150),
    valor_compra_2 VARCHAR(150)
);

DROP TABLE IF EXISTS vagas.informacoes_basicas;
DROP TABLE IF EXISTS vagas.perfil_vaga;
DROP TABLE IF EXISTS vagas.beneficios;

SELECT * FROM vagas.informacoes_basicas;
SELECT * FROM vagas.perfil_vaga;
SELECT * FROM vagas.beneficios;

TRUNCATE TABLE vagas.informacoes_basicas RESTART IDENTITY;
TRUNCATE TABLE vagas.perfil_vaga RESTART IDENTITY;
TRUNCATE TABLE vagas.beneficios RESTART IDENTITY;

