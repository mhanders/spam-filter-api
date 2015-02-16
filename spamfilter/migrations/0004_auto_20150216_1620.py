# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('spamfilter', '0003_distribution_defaultprobabilities'),
    ]

    operations = [
        migrations.RenameField(
            model_name='distribution',
            old_name='defaultProbabilities',
            new_name='default_probabilities',
        ),
        migrations.RenameField(
            model_name='distribution',
            old_name='logPrior',
            new_name='log_priors',
        ),
        migrations.RenameField(
            model_name='distribution',
            old_name='logProbabilities',
            new_name='log_probabilities',
        ),
    ]
