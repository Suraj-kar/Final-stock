from django.db import models
from django.utils.timezone import now  # Import now()


class StockLog(models.Model):
    company = models.CharField(
        max_length=50, default="DefaultCompany", verbose_name="Company Name"
    )  # Company field

    open_price = models.DecimalField(
        max_digits=15, decimal_places=2, verbose_name="Open Price"
    )
    high_price = models.DecimalField(
        max_digits=15, decimal_places=2, verbose_name="High Price"
    )
    low_price = models.DecimalField(
        max_digits=15, decimal_places=2, verbose_name="Low Price"
    )
    volume = models.BigIntegerField(verbose_name="Volume")  # Using BigIntegerField for volume

    result = models.DecimalField(
        max_digits=15, decimal_places=2, verbose_name="Predicted Close Price"
    )

    percentage_change = models.DecimalField(
        max_digits=7, decimal_places=2, null=True, blank=True, verbose_name="% Change"
    )
    difference = models.DecimalField(
        max_digits=15, decimal_places=2, null=True, blank=True, verbose_name="Price Difference"
    )

    timestamp = models.DateTimeField(default=now, verbose_name="Recorded At")

    class Meta:
        verbose_name = "Stock Log"
        verbose_name_plural = "Stock Logs"
        ordering = ["-timestamp"]  # Show newest logs first

    def __str__(self):
        return f"{self.company} | Open: {self.open_price} | Close: {self.result} | {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
