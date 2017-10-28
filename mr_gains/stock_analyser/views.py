from django.shortcuts import render

from django.http import HttpResponse


def ticker_picker(request):
    return HttpResponse("Hello, world. You're at the ticker_picker page")

def analysis_summary(request):
    return HttpResponse("Hello, world. You're at the analysis_summary page")