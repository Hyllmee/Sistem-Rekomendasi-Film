# recommender_app/admin.py
from django.contrib import admin
from .models import User, Film, RatingFilm, Admin

admin.site.register(User)
admin.site.register(Film)
admin.site.register(RatingFilm)
admin.site.register(Admin)