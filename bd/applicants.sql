CREATE TABLE applicants.infos_basicas (
	id SERIAL PRIMARY KEY,
	telefone_recado VARCHAR(30),
	telefone VARCHAR(30),
	objetivo_profissional TEXT,
	data_criacao TIMESTAMP,
	inserido_por VARCHAR(100),
	email VARCHAR(255),
	local VARCHAR(100),
	sabendo_de_nos_por VARCHAR(50),
	data_atualizacao TIMESTAMP,
	codigo_profissional VARCHAR(10),
	nome VARCHAR(150)
);

CREATE TABLE applicants.informacoes_pessoais (
	id SERIAL PRIMARY KEY,
	data_aceite VARCHAR(200),
	nome VARCHAR(150),
	cpf VARCHAR(15),
	fonte_indicacao VARCHAR(200),
	email VARCHAR(255),
	email_secundario VARCHAR(255),
	data_nascimento DATE,
	telefone_celular VARCHAR(50),
	telefone_recado VARCHAR(50),
	sexo VARCHAR(100),
	estado_civil VARCHAR(200),
	pcd VARCHAR(100),
	endereco VARCHAR(255),
	skype VARCHAR(200),
	url_linkedin VARCHAR(200),
	facebook VARCHAR(200)
);

CREATE TABLE applicants.informacoes_profissionais (
	id SERIAL PRIMARY KEY,
	titulo_profissional VARCHAR(255),
	area_atuacao TEXT,
	conhecimentos_tecnicos TEXT,
	certificacoes TEXT,
	outras_certificacoes TEXT,
	remuneracao VARCHAR(255),
	nivel_profissional VARCHAR(255)
);

CREATE TABLE applicants.formacao_e_idiomas (
	id SERIAL PRIMARY KEY,
	nivel_academico VARCHAR(50),
	nivel_ingles VARCHAR(20),
	nivel_espanhol VARCHAR(20),
	outro_idioma VARCHAR(50)
);

CREATE TABLE applicants.cargo_atual (
	id SERIAL PRIMARY KEY,
	id_ibrati VARCHAR(10),
	email_corporativo VARCHAR(150),
	cargo_atual VARCHAR(150),
	projeto_atual VARCHAR(255),
	cliente VARCHAR(150),
	unidade VARCHAR(150),
	data_admissao DATE,
	data_ultima_promocao DATE,
	nome_superior_imediato VARCHAR(150),
	email_superior_imediato VARCHAR(150)
);

CREATE TABLE applicants.curriculos (
    id SERIAL PRIMARY KEY,
    cv_pt TEXT,
    cv_en TEXT
);

-- Visualizar as tabelas
SELECT * FROM applicants.infos_basicas ORDER BY id;
SELECT * FROM applicants.informacoes_pessoais ORDER BY id;
SELECT * FROM applicants.informacoes_profissionais ORDER BY id;
SELECT * FROM applicants.formacao_e_idiomas ORDER BY id;
SELECT * FROM applicants.cargo_atual ORDER BY id;
SELECT cv_pt FROM applicants.curriculos ORDER BY id;

SELECT * 
FROM applicants.cargo_atual
WHERE id_ibrati = '52467';

-- Apagar as tabelas
DROP TABLE IF EXISTS applicants.infos_basicas;
DROP TABLE IF EXISTS applicants.informacoes_pessoais;
DROP TABLE IF EXISTS applicants.informacoes_profissionais;
DROP TABLE IF EXISTS applicants.formacao_e_idiomas;
DROP TABLE IF EXISTS applicants.cargo_atual;
DROP TABLE IF EXISTS applicants.curriculos;


-- Limpar dados de uma tabela
TRUNCATE TABLE applicants.infos_basicas RESTART IDENTITY;
TRUNCATE TABLE applicants.informacoes_pessoais RESTART IDENTITY;
TRUNCATE TABLE applicants.informacoes_profissionais RESTART IDENTITY;
TRUNCATE TABLE applicants.formacao_e_idiomas RESTART IDENTITY;
TRUNCATE TABLE applicants.cargo_atual RESTART IDENTITY;
TRUNCATE TABLE applicants.curriculos RESTART IDENTITY;

