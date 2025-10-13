CREATE SCHEMA filtred;

CREATE TABLE filtred.curriculos (
    id SERIAL PRIMARY KEY,
    cv_pt TEXT,
	cv_sugerido TEXT
);

CREATE TABLE filtred.vagas (
	id SERIAL PRIMARY KEY,
	titulo_vaga VARCHAR(255),
	areas_atuacao TEXT,
	principais_atividades TEXT
);

SELECT * FROM filtred.curriculos ORDER BY ID;
SELECT * FROM filtred.vagas ORDER BY ID;

-- DROP TABLE filtred.curriculos;

-- DROP TABLE filtred.vagas;

-- TRUNCATE TABLE filtred.curriculos RESTART IDENTITY;

-- TRUNCATE TABLE filtred.vagas RESTART IDENTITY;

SELECT COUNT(*) AS qtd_gravadas FROM filtred.curriculos;
