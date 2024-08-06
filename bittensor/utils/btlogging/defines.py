# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Btlogging constant definition module."""

BASE_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
TRACE_LOG_FORMAT = (
    f"%(asctime)s | %(levelname)s | %(name)s:%(filename)s:%(lineno)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
BITTENSOR_LOGGER_NAME = "bittensor"
DEFAULT_LOG_FILE_NAME = "bittensor.log"
DEFAULT_MAX_ROTATING_LOG_FILE_SIZE = 25 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 10
