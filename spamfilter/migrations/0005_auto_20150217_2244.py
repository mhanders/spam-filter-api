# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('spamfilter', '0004_auto_20150216_1620'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='distribution',
            name='name',
        ),
        migrations.AddField(
            model_name='distribution',
            name='num_ham',
            field=models.IntegerField(default=0),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='distribution',
            name='num_spam',
            field=models.IntegerField(default=0),
            preserve_default=True,
        ),
    ]
