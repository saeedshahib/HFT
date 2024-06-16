from django import forms
from django.utils.dateparse import parse_datetime
from django.utils.dateformat import format as format_datetime


class CustomDateTimeWidget(forms.DateTimeInput):
    def __init__(self, attrs=None, format='%Y-%m-%d %H:%M:%S.%f'):
        super().__init__(attrs, format=format)

    def format_value(self, value):
        if value:
            return format_datetime(value, self.format)
        return super().format_value(value)
