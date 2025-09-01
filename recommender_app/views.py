from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
import pandas as pd
from .models import User, Film, RatingFilm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from io import BytesIO
from django.db import transaction
from .forms import RatingForm
from django.contrib.auth.decorators import login_required
from django.core.cache import cache  # Import cache
from scipy.sparse import csr_matrix  # Import sparse matrix
import time  # Import time untuk mengukur waktu eksekusi

# --- Recommendation System Functions (Optimized) ---
def get_user_item_matrix():
    """Mengambil data rating dan membentuk matriks user-item (optimized)."""
    cache_key = 'user_item_matrix'
    user_item_matrix = cache.get(cache_key)

    if user_item_matrix is None:
        start_time = time.time()  # Catat waktu mulai

        # Gunakan values_list untuk mengambil data yang dibutuhkan saja
        ratings_data = RatingFilm.objects.values_list('user__username', 'film__film_name', 'film_rating')
        df_ratings = pd.DataFrame.from_records(ratings_data, columns=['user__username', 'film__film_name', 'film_rating'])

        if df_ratings.empty:
            return pd.DataFrame()

        user_item_matrix = df_ratings.pivot_table(index='user__username', columns='film__film_name', values='film_rating')

        # Simpan ke cache selama 24 jam
        cache.set(cache_key, user_item_matrix, 60 * 60 * 24)

        end_time = time.time()  # Catat waktu selesai
        print(f"Waktu pembuatan matriks user-item: {end_time - start_time:.2f} detik")

    return user_item_matrix

def normalize_ratings(user_item_matrix):
    """Melakukan normalisasi rating menggunakan mean-centering (optimized)."""
    cache_key = 'normalized_ratings'
    normalized_data = cache.get(cache_key)

    if normalized_data is None:
        start_time = time.time()  # Catat waktu mulai

        if user_item_matrix.empty:
            return pd.DataFrame(), pd.Series()

        user_ratings_mean = user_item_matrix.mean(axis=1)
        ratings_normalized = user_item_matrix.sub(user_ratings_mean, axis=0)

        # Simpan ke cache selama 24 jam
        cache.set(cache_key, (ratings_normalized, user_ratings_mean), 60 * 60 * 24)

        end_time = time.time()  # Catat waktu selesai
        print(f"Waktu normalisasi rating: {end_time - start_time:.2f} detik")

    return normalized_data

def calculate_cosine_similarity(ratings_normalized):
    """Menghitung Cosine Similarity antar pengguna (optimized)."""
    cache_key = 'cosine_similarity'
    user_similarity_df = cache.get(cache_key)

    if user_similarity_df is None:
        start_time = time.time()  # Catat waktu mulai

        if ratings_normalized.empty:
            return pd.DataFrame()

        # Ubah matriks menjadi sparse matrix
        ratings_sparse = csr_matrix(ratings_normalized.fillna(0))

        # Hitung cosine similarity
        user_similarity = cosine_similarity(ratings_sparse)
        user_similarity_df = pd.DataFrame(user_similarity, index=ratings_normalized.index, columns=ratings_normalized.index)
        np.fill_diagonal(user_similarity_df.values, 0)

        # Simpan ke cache selama 24 jam
        cache.set(cache_key, user_similarity_df, 60 * 60 * 24)

        end_time = time.time()  # Catat waktu selesai
        print(f"Waktu perhitungan cosine similarity: {end_time - start_time:.2f} detik")

    return user_similarity_df

def predict_rating(user_id, film_id, user_item_matrix, user_similarity_df, user_ratings_mean):
    """Memprediksi rating untuk user_id terhadap film_id (optimized)."""
    try:
        target_user = User.objects.get(id=user_id)
        target_film = Film.objects.get(id_film=film_id)
        target_username = target_user.username
        target_film_name = target_film.film_name
    except (User.DoesNotExist, Film.DoesNotExist):
        return None

    if target_username not in user_item_matrix.index:
        return user_item_matrix.mean().mean() if not user_item_matrix.empty else None

    if target_film_name in user_item_matrix.columns and not pd.isna(user_item_matrix.loc[target_username, target_film_name]):
        return user_item_matrix.loc[target_username, target_film_name]

    similar_users = user_similarity_df[target_username].sort_values(ascending=False)

    if target_film_name not in user_item_matrix.columns:
        return None

    users_who_rated_film = user_item_matrix[target_film_name].dropna().index
    k = 20
    top_similar_users = similar_users[similar_users.index.isin(users_who_rated_film)][:k]
    top_similar_users = top_similar_users[top_similar_users > 0]

    if top_similar_users.empty:
        if target_username in user_ratings_mean.index:
            return user_ratings_mean.loc[target_username]
        elif not user_item_matrix.empty:
            return user_item_matrix.mean().mean()
        else:
            return None

    numerator = 0
    denominator = 0
    for sim_user_username, similarity_score in top_similar_users.items():
        if sim_user_username in user_item_matrix.index and target_film_name in user_item_matrix.columns and not pd.isna(user_item_matrix.loc[sim_user_username, target_film_name]):
            numerator += similarity_score * (user_item_matrix.loc[sim_user_username, target_film_name] - user_ratings_mean.loc[sim_user_username])
            denominator += similarity_score

    if denominator == 0:
        if target_username in user_ratings_mean.index:
            return user_ratings_mean.loc[target_username]
        elif not user_item_matrix.empty:
            return user_item_matrix.mean().mean()
        else:
            return None

    predicted_rating = user_ratings_mean.loc[target_username] + (numerator / denominator)
    return max(1, min(5, predicted_rating))

def get_recommendations(user_id, num_recommendations=10):
    """Memberikan rekomendasi film untuk user_id (optimized)."""
    user_item_matrix = get_user_item_matrix()
    if user_item_matrix.empty:
        return []

    ratings_normalized, user_ratings_mean = normalize_ratings(user_item_matrix)
    user_similarity_df = calculate_cosine_similarity(ratings_normalized)

    try:
        target_user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return []
    target_username = target_user.username

    if target_username not in user_item_matrix.index:
        return []

    rated_films_names = user_item_matrix.columns[user_item_matrix.loc[target_username].notna()].tolist()
    all_films = Film.objects.all()
    unrated_films = [film for film in all_films if film.film_name not in rated_films_names]

    predictions = {}
    for film in unrated_films:
        predicted_rating = predict_rating(user_id, film.id_film, user_item_matrix, user_similarity_df, user_ratings_mean)
        if predicted_rating is not None:
            predictions[film] = predicted_rating

    recommended_films = sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:num_recommendations]
    return [film for film, rating in recommended_films]

# --- Basic Views ---
def landing_page(request):
    return render(request, 'recommender_app/landing_page.html')

def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Akun berhasil dibuat! Silahkan masuk.')
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'recommender_app/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, 'Invalid username or password.')
    else:
        form = LoginForm()
    return render(request, 'recommender_app/login.html', {'form': form})

def home(request):
    if request.user.is_authenticated:
        return render(request, 'recommender_app/home.html')
    else:
        return redirect('login')

def logout_view(request):
    logout(request)
    return redirect('landing_page')

def edit_profile(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            form = EditProfileForm(request.POST, instance=request.user)
            if form.is_valid():
                form.save()
                messages.success(request, 'Profil berhasil diperbarui!')
                return redirect('home')
        else:
            form = EditProfileForm(instance=request.user)
        return render(request, 'recommender_app/edit_profile.html', {'form': form})
    else:
        return redirect('login')

def daftar_film(request):
    films = Film.objects.all()
    return render(request, 'recommender_app/daftar_film.html', {'films': films})

@login_required
def detail_film(request, film_id):
    film = get_object_or_404(Film, id_film=film_id)
    user_rating = None  # Inisialisasi user_rating

    try:
        user_rating_obj = RatingFilm.objects.get(user=request.user, film=film)
        user_rating = user_rating_obj.film_rating
    except RatingFilm.DoesNotExist:
        user_rating = None

    if request.method == 'POST':
        form = RatingForm(request.POST)
        if form.is_valid():
            film_rating = int(form.cleaned_data['film_rating'])

            # Update atau buat rating baru
            RatingFilm.objects.update_or_create(
                user=request.user,
                film=film,
                defaults={'film_rating': film_rating}
            )

            # Setelah menyimpan rating, redirect kembali ke detail film
            return redirect('detail_film', film_id=film_id)
    else:
        # Jika bukan POST, buat instance form kosong
        form = RatingForm(initial={'film_rating': user_rating})

    context = {
        'film': film,
        'form': form,
        'user_rating': user_rating,
    }
    return render(request, 'recommender_app/detail_film.html', context)

def hasil_rekomendasi(request):
    if request.user.is_authenticated:
        user_id = request.user.id
        recommended_films = get_recommendations(user_id)
        return render(request, 'recommender_app/hasil_rekomendasi.html', {'recommended_films': recommended_films})
    else:
        return redirect('login')

# --- Admin Views ---
def admin_login(request):
    if request.method == 'POST':
        form = AdminLoginForm(request.POST) # Gunakan AdminLoginForm
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            admin = authenticate(request, username=username, password=password)
            if admin is not None and admin.is_staff:
                login(request, admin)
                return redirect('admin_dashboard')
            else:
                messages.error(request, 'Invalid admin credentials.')
                return render(request, 'recommender_app/admin_login.html', {'form': form}) # Kirim form ke template
    else:
        form = AdminLoginForm() # Buat instance AdminLoginForm
    return render(request, 'recommender_app/admin_login.html', {'form': form}) # Kirim form ke template

@staff_member_required
def admin_dashboard(request):
    return render(request, 'recommender_app/admin_dashboard.html')

def admin_logout(request):
    logout(request)
    return redirect('admin_login')

@staff_member_required
def admin_daftar_user(request):
    users = User.objects.all()
    return render(request, 'recommender_app/admin_daftar_user.html', {'users': users})

@staff_member_required
def admin_user_add(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('admin_daftar_user')
    else:
        form = UserForm()
    return render(request, 'recommender_app/admin_user_form.html', {'form': form})

@staff_member_required
def admin_edit_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if request.method == 'POST':
        form = UserForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return redirect('admin_daftar_user')
    else:
        form = UserForm(instance=user)
    return render(request, 'recommender_app/admin_user_form.html', {'form': form})

@staff_member_required
def admin_delete_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if request.method == 'POST':
        user.delete()
        return redirect('admin_daftar_user')
    return render(request, 'recommender_app/admin_confirm_delete.html', {'object': user, 'type': 'user'})

@staff_member_required
def admin_daftar_film(request):
    films = Film.objects.all()
    return render(request, 'recommender_app/admin_daftar_film.html', {'films': films})

@staff_member_required
def admin_film_add(request):
    if request.method == 'POST':
        form = FilmForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('admin_daftar_film')
    else:
        form = FilmForm()
    return render(request, 'recommender_app/admin_film_form.html', {'form': form})

@staff_member_required
def admin_film_edit(request, film_id):
    film = get_object_or_404(Film, id_film=film_id)
    if request.method == 'POST':
        form = FilmForm(request.POST, instance=film)
        if form.is_valid():
            form.save()

    else:
        form = FilmForm(instance=film)
    return render(request, 'recommender_app/admin_film_form.html', {'form': form})

@staff_member_required
def admin_film_delete(request, film_id):
    film = get_object_or_404(Film, id_film=film_id)
    if request.method == 'POST':
        film.delete()
        return redirect('admin_daftar_film')
    return render(request, 'recommender_app/admin_confirm_delete.html', {'object': film, 'type': 'film'})

import matplotlib.pyplot as plt
import base64
from io import BytesIO

@staff_member_required
def admin_evaluasi_sistem(request):
    start_time = time.time()  # Catat waktu mulai

    # --- Evaluate System ---
    user_item_matrix = get_user_item_matrix()
    if user_item_matrix.empty:
        mae = 0
        rmse = 0
        scatter_plot_url = None
        messages.warning(request, "Tidak ada data rating yang cukup untuk evaluasi sistem.")
    else:
        ratings_normalized, user_ratings_mean = normalize_ratings(user_item_matrix)
        user_similarity_df = calculate_cosine_similarity(ratings_normalized)

        actual_ratings = []
        predicted_ratings = []

        # Ambil semua rating dari database
        all_ratings = RatingFilm.objects.all()

        for rating in all_ratings:
            user_id = rating.user.id
            film_id = rating.film.id_film

            predicted = predict_rating(user_id, film_id, user_item_matrix, user_similarity_df, user_ratings_mean)
            if predicted is not None:
                actual_ratings.append(rating.film_rating)
                predicted_ratings.append(predicted)

        if not actual_ratings:
            mae = 0
            rmse = 0
            scatter_plot_url = None
            messages.warning(request, "Tidak ada data rating yang cukup untuk evaluasi sistem.")
        else:
            mae = mean_absolute_error(actual_ratings, predicted_ratings)
            rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

            # --- Create Scatter Plot ---
            plt.figure(figsize=(8, 6))
            plt.scatter(actual_ratings, predicted_ratings, alpha=0.5)
            plt.title('Scatter Plot: Actual vs Predicted Ratings')
            plt.xlabel('Actual Ratings')
            plt.ylabel('Predicted Ratings')
            plt.xlim(1, 5)
            plt.ylim(1, 5)
            plt.grid(True)

            # Save the plot to a BytesIO object
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            scatter_plot_url = f'data:image/png;base64,{image_base64}'

    context = {
        'mae': mae,
        'rmse': rmse,
        'scatter_plot_url': scatter_plot_url,
    }

    end_time = time.time()  # Catat waktu selesai
    print(f"Waktu evaluasi sistem: {end_time - start_time:.2f} detik")

    return render(request, 'recommender_app/admin_evaluasi_sistem.html', context)

@staff_member_required
def admin_laporan_rekomendasi(request):
    start_time = time.time()  # Catat waktu mulai

    user_item_matrix = get_user_item_matrix()

    if user_item_matrix.empty:
        report_data = []
        mae = 0
        rmse = 0
        user_similarity_df = pd.DataFrame()  # Empty DataFrame
        messages.warning(request, "Tidak ada data rating untuk menghasilkan laporan.")
    else:
        ratings_normalized, user_ratings_mean = normalize_ratings(user_item_matrix)
        user_similarity_df = calculate_cosine_similarity(ratings_normalized)

        report_data = []
        actual_ratings = []
        predicted_ratings = []

        # Ambil semua rating dari database
        all_ratings = RatingFilm.objects.all()

        for rating_obj in all_ratings:
            user_id = rating_obj.user.id
            film_id = rating_obj.film.id_film
            predicted_rating = predict_rating(user_id, film_id, user_item_matrix, user_similarity_df, user_ratings_mean)

            if predicted_rating is not None:
                report_data.append({
                    'user_name': rating_obj.user.username,
                    'film_name': rating_obj.film.film_name,
                    'predicted_rating': predicted_rating,
                    'actual_rating': rating_obj.film_rating,
                })
                actual_ratings.append(rating_obj.film_rating)
                predicted_ratings.append(predicted_rating)

        if not actual_ratings:
            report_data = []
            mae = 0
            rmse = 0
            user_similarity_df = pd.DataFrame()  # Empty DataFrame
            messages.warning(request, "Tidak ada data rating untuk menghasilkan laporan.")
        else:
            mae = mean_absolute_error(actual_ratings, predicted_ratings)
            rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

    context = {
        'report_data': report_data,
        'mae': mae,
        'rmse': rmse,
        'user_similarity_df': user_similarity_df.to_html(),  # Convert DataFrame to HTML
    }

    end_time = time.time()  # Catat waktu selesai
    print(f"Waktu pembuatan laporan rekomendasi: {end_time - start_time:.2f} detik")

    return render(request, 'recommender_app/admin_laporan_rekomendasi.html', context)

@staff_member_required
def admin_hasil_prediksi(request):
    start_time = time.time()  # Catat waktu mulai

    user_item_matrix = get_user_item_matrix()

    if user_item_matrix.empty:
        messages.warning(request, "Tidak ada data rating untuk menghasilkan prediksi.")
        prediction_data = None
    else:
        ratings_normalized, user_ratings_mean = normalize_ratings(user_item_matrix)
        user_similarity_df = calculate_cosine_similarity(ratings_normalized)

        prediction_data = []
        all_users = User.objects.all()
        all_films = Film.objects.all()

        for user in all_users:
            for film in all_films:
                predicted_rating = predict_rating(user.id, film.id_film, user_item_matrix, user_similarity_df, user_ratings_mean)
                if predicted_rating is not None:
                    prediction_data.append({
                        'user_name': user.username,
                        'film_name': film.film_name,
                        'predicted_rating': predicted_rating
                    })

    context = {
        'prediction_data': prediction_data,
    }

    end_time = time.time()  # Catat waktu selesai
    print(f"Waktu pembuatan hasil prediksi: {end_time - start_time:.2f} detik")

    return render(request, 'recommender_app/admin_hasil_prediksi.html', context)

@staff_member_required
def admin_hasil_similarity(request):
    start_time = time.time()  # Catat waktu mulai

    user_item_matrix = get_user_item_matrix()

    if user_item_matrix.empty:
        messages.warning(request, "Tidak ada data rating untuk menghasilkan similarity.")
        prediction_data = None
        similarity_data = None
    else:
        ratings_normalized, user_ratings_mean = normalize_ratings(user_item_matrix)
        user_similarity_df = calculate_cosine_similarity(ratings_normalized)

        prediction_data = []
        all_users = User.objects.all()
        all_films = Film.objects.all()

        for user in all_users:
            for film in all_films:
                predicted_rating = predict_rating(user.id, film.id_film, user_item_matrix, user_similarity_df, user_ratings_mean)
                if predicted_rating is not None:
                    prediction_data.append({
                        'user_name': user.username,
                        'film_name': film.film_name,
                        'predicted_rating': predicted_rating
                    })

        similarity_data = []
        users = user_similarity_df.index.tolist()
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1 = users[i]
                user2 = users[j]
                similarity = user_similarity_df.loc[user1, user2]
                similarity_data.append({
                    'user1': user1,
                    'user2': user2,
                    'similarity': similarity
                })

    context = {
        'prediction_data': prediction_data,
        'similarity_data': similarity_data
    }

    end_time = time.time()  # Catat waktu selesai
    print(f"Waktu pembuatan hasil similarity: {end_time - start_time:.2f} detik")

    return render(request, 'recommender_app/admin_hasil_prediksi_similarity.html', context)