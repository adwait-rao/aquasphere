import uuid
from django.db import models
# from django.template.defaultfilters import slugify
# Create your models here.


class irrig_sched(models.Model):
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,  # Automatically generates a UUID
        editable=False
    )
    crop_name = models.CharField(max_length=255)  # TEXT in SQL
    date = models.DateField()  # DATE in SQL
    scheduled_irrigation_liters = models.FloatField()  # REAL in SQL

    def __str__(self):
        return f"{self.crop_name} - {self.date}"
    
class water_requi(models.Model):
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,  # Automatically generates a UUID
        editable=False
    )
    crop_name = models.CharField(max_length=255)  # TEXT NOT NULL
    date = models.DateField()  # TEXT -> DateField for proper date handling
    daily_wr_mm = models.FloatField()  # REAL NOT NULL
    daily_wr_liters = models.FloatField()  # REAL NOT NULL
    rainfall_mm = models.FloatField()  # REAL NOT NULL
    area_ha = models.FloatField(null=True) # REAL in SQL

    def __str__(self):
        return f"{self.crop_name} - {self.date}"