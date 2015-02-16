# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('spamfilter', '0002_auto_20150215_1500'),
    ]

    operations = [
        migrations.AddField(
            model_name='distribution',
            name='defaultProbabilities',
            field=models.TextField(default=b''),
            preserve_default=True,
        ),
    ]
