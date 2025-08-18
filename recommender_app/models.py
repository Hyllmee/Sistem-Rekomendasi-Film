# recommender_app/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.hashers import make_password, check_password 

# Model User kustom yang menggantikan Django's User model
# Menambahkan field 'umur' dan 'jenis_kelamin'
class User(AbstractUser):
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=16, null=True, blank=True)

    def __str__(self):
        return self.username

class Film(models.Model):
    id_film = models.IntegerField(primary_key=True)
    film_name = models.CharField(max_length=255)
    image = models.CharField(max_length=255, blank=True, null=True) # Untuk path gambar poster
    description = models.TextField(blank=True, null=True) # Deskripsi gabungan lama
    sinopsis = models.TextField(blank=True, null=True) # Sinopsis dari OMDb/TMDb
    release_year = models.IntegerField(null=True, blank=True) # <--- FIELD BARU UNTUK TAHUN RILIS
    imdb_url = models.URLField(max_length=200, blank=True, null=True) # <--- FIELD BARU UNTUK IMDB URL

    def __str__(self):
        return self.film_name

class RatingFilm(models.Model):
    id_rating = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    film = models.ForeignKey(Film, on_delete=models.CASCADE)
    film_rating = models.IntegerField()

    class Meta:
        unique_together = ('user', 'film') # Memastikan satu user hanya bisa memberi rating satu film sekali

    def __str__(self):
        return f"User: {self.user.username}, Film: {self.film.film_name}, Rating: {self.film_rating}"

class Admin(models.Model):
    id_admin = models.IntegerField(primary_key=True)
    username = models.CharField(max_length=50, unique=True) # Tambahkan unique=True
    password = models.CharField(max_length=128) # Perbesar ukuran max_length untuk menampung hash password

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)

    def __str__(self):
        return self.username