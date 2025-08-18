from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import User, Film

class RegisterForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'email', 'age', 'gender')  # Add other fields as needed

class LoginForm(forms.Form):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)

class EditProfileForm(UserChangeForm):
    class Meta:
        model = User
        fields = ('username', 'email', 'age', 'gender') # Add other fields as needed

class FilmForm(forms.ModelForm):
    class Meta:
        model = Film
        fields = '__all__'  # Or specify the fields you want to include

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('username', 'email', 'is_staff', 'is_active') # Adjust fields as needed

class AdminLoginForm(forms.Form): # Tambahkan form ini
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)


class RatingForm(forms.Form):
    film_rating = forms.ChoiceField(
        choices=[(i, str(i)) for i in range(1, 6)],
        widget=forms.RadioSelect,
        label="Berikan Rating Anda:"
    )