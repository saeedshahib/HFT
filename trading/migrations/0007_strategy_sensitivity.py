# Generated by Django 5.0.4 on 2024-05-09 17:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trading', '0006_position_break_even_price'),
    ]

    operations = [
        migrations.AddField(
            model_name='strategy',
            name='sensitivity',
            field=models.DecimalField(blank=True, decimal_places=16, max_digits=32, null=True),
        ),
    ]
