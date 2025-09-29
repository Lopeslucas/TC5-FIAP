from django.db import models

class Curriculo(models.Model):
    cv_pt = models.TextField()
    cv_sugerido = models.TextField(blank=True, null=True)

    class Meta:
        db_table = 'curriculos'

    def __str__(self):
        return f"Curriculo {self.id}"


class Vaga(models.Model):
    titulo_vaga = models.CharField(max_length=255)
    areas_atuacao = models.TextField()
    principais_atividades = models.TextField()

    class Meta:
        db_table = 'vagas'

    def __str__(self):
        return self.titulo_vaga
