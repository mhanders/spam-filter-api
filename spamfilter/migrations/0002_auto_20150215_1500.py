# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('spamfilter', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='distribution',
            name='logPrior',
            field=models.TextField(default=b'[-0.69314718055994529, -0.69314718055994529]'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='distribution',
            name='logProbabilities',
            field=models.TextField(default=b'{}'),
            preserve_default=True,
        ),
    ]
