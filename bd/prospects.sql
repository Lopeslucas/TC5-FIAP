CREATE TABLE prospects.prospects (
    id SERIAL PRIMARY KEY,
    titulo_vaga VARCHAR(255),
    modalidade VARCHAR(100),
    nome VARCHAR(150),
    codigo VARCHAR(50),
    situacao_candidado VARCHAR(100),
    data_candidatura DATE,
    ultima_atualizacao DATE,
    comentario TEXT,
    recrutador VARCHAR(150)
);

SELECT * FROM prospects.prospects;

DROP TABLE prospects.prospects;

TRUNCATE TABLE prospects.prospects RESTART IDENTITY;