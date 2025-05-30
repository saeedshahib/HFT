# Generated by Django 5.0.4 on 2024-05-09 08:26

import django.db.models.deletion
from decimal import Decimal
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Currency',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('symbol', models.CharField(max_length=255)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Market',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('exchange', models.CharField(choices=[('MEXC', 'Mexc'), ('BingX', 'Bingx'), ('Kucoin', 'Kucoin'), ('Bybit', 'Bybit')], max_length=50)),
                ('symbol', models.CharField(max_length=50)),
                ('market_type', models.CharField(choices=[('Spot', 'Spot'), ('Margin', 'Margin'), ('Futures', 'Futures')], max_length=50)),
                ('fee', models.DecimalField(decimal_places=8, default=Decimal('0.001'), max_digits=20)),
                ('first_currency', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='first_currency', to='trading.currency')),
                ('second_currency', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='second_currency', to='trading.currency')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Position',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('symbol', models.CharField(max_length=50)),
                ('amount', models.DecimalField(decimal_places=16, max_digits=32)),
                ('value', models.DecimalField(decimal_places=16, max_digits=32)),
                ('unrealized_usdt_pnl', models.DecimalField(decimal_places=16, max_digits=32)),
                ('realized_usdt_pnl', models.DecimalField(decimal_places=16, max_digits=32)),
                ('average_entry_price', models.DecimalField(decimal_places=16, max_digits=32)),
                ('take_profit_price', models.DecimalField(decimal_places=16, max_digits=32)),
                ('stop_loss_price', models.DecimalField(decimal_places=16, max_digits=32)),
                ('status', models.CharField(choices=[('Initiated', 'Initiated'), ('Open', 'Open'), ('Closed', 'Closed')], default='Initiated', max_length=50)),
                ('market', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='trading.market')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Order',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('symbol', models.CharField(max_length=50)),
                ('order_type', models.CharField(choices=[('Limit', 'Limit'), ('Market', 'Market')], max_length=50)),
                ('side', models.CharField(choices=[('Buy', 'Buy'), ('Sell', 'Sell')], max_length=50)),
                ('amount', models.DecimalField(decimal_places=16, max_digits=32)),
                ('filled_amount', models.DecimalField(blank=True, decimal_places=16, max_digits=32, null=True)),
                ('price', models.DecimalField(blank=True, decimal_places=16, max_digits=32, null=True)),
                ('average_price', models.DecimalField(decimal_places=16, max_digits=32)),
                ('status', models.CharField(choices=[('Pending', 'Pending'), ('Filled', 'Filled'), ('Partially Field', 'Partially Field'), ('Partially Filled Done', 'Partially Filled Done'), ('Cancelled', 'Cancelled')], max_length=50)),
                ('take_profit_price', models.DecimalField(blank=True, decimal_places=16, max_digits=32, null=True)),
                ('stop_loss_price', models.DecimalField(blank=True, decimal_places=16, max_digits=32, null=True)),
                ('market', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='trading.market')),
                ('position', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='trading.position')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Strategy',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('strategy_type', models.CharField(choices=[('Recent Candle', 'Recentcandle')], max_length=50)),
                ('active', models.BooleanField(default=False)),
                ('leverage', models.DecimalField(decimal_places=16, max_digits=32)),
                ('order_size_from_basket', models.DecimalField(decimal_places=16, max_digits=32)),
                ('take_profit', models.DecimalField(decimal_places=16, max_digits=32)),
                ('stop_loss', models.DecimalField(decimal_places=16, max_digits=32)),
                ('market', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='trading.market')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.AddField(
            model_name='position',
            name='strategy',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='trading.strategy'),
        ),
        migrations.CreateModel(
            name='Trade',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('symbol', models.CharField(max_length=50)),
                ('amount', models.DecimalField(decimal_places=16, max_digits=32)),
                ('price', models.DecimalField(decimal_places=16, max_digits=32)),
                ('fee_amount', models.DecimalField(decimal_places=16, max_digits=32)),
                ('side', models.CharField(choices=[('Buy', 'Buy'), ('Sell', 'Sell')], max_length=50)),
                ('market', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='trading.market')),
                ('order', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='trading.order')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
