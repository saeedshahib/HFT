from django.db import models

from utils.models import BaseModel
from trading.models import Market
# Create your models here.


class Candle(BaseModel):
    symbol = models.CharField()
    market = models.ForeignKey(Market, on_delete=models.SET_NULL, null=True)
    date = models.DateTimeField(null=True, blank=True)
    timestamp = models.CharField(db_index=True)
    high = models.DecimalField(decimal_places=16, max_digits=32)
    low = models.DecimalField(decimal_places=16, max_digits=32)
    close = models.DecimalField(decimal_places=16, max_digits=32)

    class Meta:
        unique_together = [
            ['market', 'timestamp'],
        ]
