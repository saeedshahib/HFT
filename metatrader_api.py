import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

from metaapi_cloud_sdk import MetaApi

token = '...'
api = MetaApi(token=token)

