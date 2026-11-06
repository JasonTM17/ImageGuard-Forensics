"""Folder ingest for finished still images.

Custody entries are append-only records, not tamper-proof evidence. Camera
labels come from the operator and file modification times are unverified
filesystem values; this package does not authenticate cameras or capture time.
"""
