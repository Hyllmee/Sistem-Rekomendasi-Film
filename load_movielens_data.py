# load_movielens_data.py
import os
import django
import pandas as pd
from django.db import IntegrityError, transaction # Tambahkan transaction untuk atomic operations
import re
import requests # Import requests untuk HTTP requests
import time # Import time untuk jeda (rate limit)

# Konfigurasi Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie_recommender.settings')
django.setup()

from recommender_app.models import User, Film, RatingFilm

# --- Konfigurasi OMDb API (Pindahkan dari update_film_synopsis.py) ---
OMDB_API_KEY = '9338faac' # GANTI DENGAN API KEY OMDB ANDA
OMDB_BASE_URL = 'http://www.omdbapi.com/'

def get_synopsis_from_omdb(film_name, release_year=None):
    """Mencari film di OMDb dan mengambil sinopsis serta URL poster."""
    params = {
        'apikey': OMDB_API_KEY,
        't': film_name, # search by title
        'plot': 'full' # to get full plot
    }
    if release_year:
        params['y'] = release_year # Parameter tahun

    try:
        response = requests.get(OMDB_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if data and data.get('Response') == 'True':
            return data.get('Plot'), data.get('Poster')
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from OMDb for '{film_name}': {e}")
        return None, None


# --- Genre Mapping (Sesuai dengan kolom genre di u.item) ---
GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def load_data():
    print("Memulai pemuatan data MovieLens 100K...")
    base_path = 'ml-100k/'

    # ... (Pemuatan data pengguna - u.user) ...
    print("Memuat data pengguna (u.user)...")
    try:
        users_df = pd.read_csv(os.path.join(base_path, 'u.user'), sep='|', header=None, names=['id', 'age', 'gender', 'occupation', 'zip_code'])
        for index, row in users_df.iterrows():
            try:
                user_obj = User.objects.get(id=row['id'])
                user_obj.username = f"user_{row['id']}"
                user_obj.age = row['age']
                user_obj.gender = row['gender']
                user_obj.save()
            except User.DoesNotExist:
                User.objects.create_user(
                    id=row['id'],
                    username=f"user_{row['id']}",
                    age=row['age'],
                    gender=row['gender'],
                    password=f"password{row['id']}"
                )
            except IntegrityError:
                print(f"Peringatan: User dengan ID {row['id']} sudah ada.")
        print(f"{users_df.shape[0]} pengguna telah diproses.")
    except FileNotFoundError:
        print(f"File u.user tidak ditemukan di '{base_path}'. Lewati pemuatan pengguna.")
    except Exception as e:
        print(f"Error saat memuat pengguna: {e}")

    # --- Pemuatan Data Film (u.item) ---
    print("Memuat data film (u.item) dan sinopsis dari OMDb...")
    omdb_updated_count = 0
    omdb_skipped_count = 0
    omdb_error_count = 0

    try:
        film_column_names = ['id_film', 'film_name', 'release_date_str', 'video_release_date', 'imdb_url_original'] + GENRE_NAMES
        
        films_df = pd.read_csv(
            os.path.join(base_path, 'u.item'),
            sep='|',
            header=None,
            encoding='latin-1',
            names=film_column_names
        )
        
        total_films_to_process = films_df.shape[0]

        for i, row in films_df.iterrows():
            film_id = row['id_film']
            film_name_original = row['film_name'].strip()
            
            # Ekstrak nama genre
            active_genres = [GENRE_NAMES[j] for j, genre_name in enumerate(GENRE_NAMES) if row[genre_name] == 1]
            
            # Ekstrak tahun rilis
            release_year = None
            if pd.notna(row['release_date_str']):
                year_match = re.search(r'(\d{4})$', str(row['release_date_str']))
                if year_match:
                    release_year = int(year_match.group(1))

            # Ambil IMDb URL asli dari dataset
            imdb_url_original = row['imdb_url_original'] if pd.notna(row['imdb_url_original']) else None

            # Coba ambil film dari DB, atau buat baru
            try:
                film_obj, created = Film.objects.get_or_create(
                    id_film=film_id,
                    defaults={
                        'film_name': film_name_original,
                        'description': ", ".join(active_genres),
                        'release_year': release_year,
                        'imdb_url': imdb_url_original, # Simpan URL IMDb asli dari MovieLens
                        'image': '', # Akan diisi oleh OMDb
                        'sinopsis': '' # Akan diisi oleh OMDb
                    }
                )
            except IntegrityError:
                print(f"Peringatan: Film dengan ID {film_id} sudah ada. Melanjutkan untuk memperbarui...")
                film_obj = Film.objects.get(id_film=film_id)
                created = False
            except Exception as e:
                omdb_error_count += 1
                print(f"Error saat mendapatkan/membuat Film {film_name_original} (ID: {film_id}): {e}")
                continue # Lanjutkan ke film berikutnya

            # Hanya ambil sinopsis/poster jika belum ada atau film baru dibuat
            if created or not film_obj.sinopsis or not film_obj.image:
                print(f"[{i+1}/{total_films_to_process}] Mengambil sinopsis/poster untuk: {film_name_original} ({release_year or 'N/A'})")
                synopsis, poster_url = get_synopsis_from_omdb(film_name_original, release_year)

                if synopsis and synopsis != "N/A":
                    film_obj.sinopsis = synopsis
                    if poster_url and poster_url != "N/A": # Hanya update image jika ada poster baru yang valid
                        film_obj.image = poster_url
                    try:
                        with transaction.atomic():
                            film_obj.save()
                        omdb_updated_count += 1
                        print(f"  Sinopsis & Poster untuk '{film_name_original}' diperbarui/ditambahkan.")
                    except Exception as e:
                        omdb_error_count += 1
                        print(f"  Error menyimpan sinopsis/poster untuk '{film_name_original}': {e}")
                else:
                    omdb_skipped_count += 1
                    print(f"  Sinopsis/Poster tidak ditemukan di OMDb untuk '{film_name_original}'.")
            else:
                omdb_skipped_count += 1
                print(f"  Sinopsis/Poster untuk '{film_name_original}' sudah ada. Dilewati.")

            time.sleep(0.1) # Jeda untuk menghindari rate limit OMDb

        print(f"\nProses pemuatan film selesai.")
        print(f"Film berhasil diproses dari OMDb: {omdb_updated_count}")
        print(f"Film dilewati dari OMDb (sinopsis/poster sudah ada atau tidak ditemukan): {omdb_skipped_count}")
        print(f"Film error saat mengambil/menyimpan dari OMDb: {omdb_error_count}")

    except FileNotFoundError:
        print(f"File u.item tidak ditemukan di '{base_path}'. Lewati pemuatan film.")
    except Exception as e:
        print(f"Error umum saat memuat film: {e}")

    # ... (Pemuatan data rating - u.data) ...
    print("\nMemuat data rating (u.data)...")
    try:
        ratings_df = pd.read_csv(os.path.join(base_path, 'u.data'), sep='\t', header=None, names=['user_id', 'film_id', 'rating', 'timestamp'])

        for index, row in ratings_df.iterrows():
            try:
                user_obj = User.objects.get(id=row['user_id'])
                film_obj = Film.objects.get(id_film=row['film_id'])

                RatingFilm.objects.update_or_create(
                    user=user_obj,
                    film=film_obj,
                    defaults={'film_rating': row['rating']}
                )
            except (User.DoesNotExist, Film.DoesNotExist):
                print(f"Peringatan: Melewatkan rating untuk user {row['user_id']} atau film {row['film_id']} (tidak ditemukan di DB).")
            except IntegrityError:
                print(f"Peringatan: Rating untuk user {row['user_id']} dan film {row['film_id']} sudah ada.")

        print(f"{ratings_df.shape[0]} rating telah diproses.")
    except FileNotFoundError:
        print(f"File u.data tidak ditemukan di '{base_path}'. Lewati pemuatan rating.")
    except Exception as e:
        print(f"Error saat memuat rating: {e}")

    print("\n--- Pemuatan Data Selesai ---")

if __name__ == '__main__':
    load_data()