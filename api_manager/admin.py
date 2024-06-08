from django.contrib import admin
from api_manager.models import Candle
from utils.admin import BaseAdminWithActionButtons


# Register your models here.

@admin.register(Candle)
class CandleAdmin(BaseAdminWithActionButtons):
    list_display = [field.name for field in Candle._meta.fields]
