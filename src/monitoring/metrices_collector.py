"""
Prometheus Metrics Definitions
"""

from prometheus_client import Counter

# Counts number of API prediction requests
REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of prediction requests received"
)