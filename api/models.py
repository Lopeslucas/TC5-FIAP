from django.db import models

class Applicant(models.Model):
    id = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=150)
    email = models.CharField(max_length=255, blank=True, null=True)
    telefone = models.CharField(max_length=30, blank=True, null=True)

    class Meta:
        db_table = "applicants"
        managed = False

class Vaga(models.Model):
    id = models.AutoField(primary_key=True)
    titulo = models.CharField(max_length=255)
    descricao = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "vagas"
        managed = False