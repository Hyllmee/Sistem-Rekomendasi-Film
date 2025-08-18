from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('home/', views.home, name='home'),
    path('logout/', views.logout_view, name='logout'),
    path('edit_profile/', views.edit_profile, name='edit_profile'),
    path('daftar_film/', views.daftar_film, name='daftar_film'),
    path('film/<int:film_id>/', views.detail_film, name='detail_film'),
    path('rekomendasi/', views.hasil_rekomendasi, name='hasil_rekomendasi'),

    # Admin URLs
    path('admin/login/', views.admin_login, name='admin_login'),
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin/logout/', views.admin_logout, name='admin_logout'),
    path('admin/daftar_user/', views.admin_daftar_user, name='admin_daftar_user'),
    path('admin/user/add/', views.admin_user_add, name='admin_user_add'),
    path('admin/user/<int:user_id>/edit/', views.admin_edit_user, name='admin_edit_user'),
    path('admin/user/<int:user_id>/delete/', views.admin_delete_user, name='admin_delete_user'),
    path('admin/daftar_film/', views.admin_daftar_film, name='admin_daftar_film'),
    path('admin/film/add/', views.admin_film_add, name='admin_add_film'),
    path('admin/film/<int:film_id>/edit/', views.admin_film_edit, name='admin_edit_film'),
    path('admin/film/<int:film_id>/delete/', views.admin_film_delete, name='admin_delete_film'),
    path('admin/evaluasi_sistem/', views.admin_evaluasi_sistem, name='admin_evaluasi_sistem'),
    path('admin/laporan_rekomendasi/', views.admin_laporan_rekomendasi, name='admin_laporan_rekomendasi'), # Keep this line
    path('admin/hasil_prediksi/', views.admin_hasil_prediksi, name='admin_hasil_prediksi'),
    path('admin/hasil_similarity/', views.admin_hasil_similarity, name='admin_hasil_similarity'),
]