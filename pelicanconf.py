#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os

AUTHOR = 'Garry Chan'
# literally the site name on the tab
SITENAME = 'Garry\'s Blog'
SITEURL = 'https://garrrychan.github.io/blog'
STATIC_PATHS = ['images']
PATH = 'content'
TIMEZONE = 'America/Toronto'
DEFAULT_LANG = 'en'
DEFAULT_PAGINATION = 10
# set to False for Production, True for Development
if os.environ.get('PELICAN_ENV') == 'DEV':
    RELATIVE_URLS = True
else:
    RELATIVE_URLS = False
