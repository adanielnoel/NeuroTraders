from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.ticker_picker),
    url(r'^$', views.analysis_summary)
]
