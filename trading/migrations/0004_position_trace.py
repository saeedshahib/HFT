# Generated by Django 5.0.4 on 2024-05-09 12:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trading', '0003_currency_precision'),
    ]

    operations = [
        migrations.AddField(
            model_name='position',
            name='trace',
            field=models.TextField(blank=True, null=True),
        ),
    ]
