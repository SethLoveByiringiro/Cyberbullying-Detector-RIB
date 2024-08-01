# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.Welcome, name='welcome'),
    path('form', views.form_view, name='form'), 
    path('result', views.formInfo, name='result')
]
