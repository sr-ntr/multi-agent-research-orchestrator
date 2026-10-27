# CloudSync Pro - Product Specification v4.0

## Overview

CloudSync Pro is the flagship enterprise synchronisation and collaboration platform developed by Arcturus Technologies Inc. Version 4.0, released in Q1 2023, introduces native AI-assisted workflow automation and significantly revised performance benchmarks.

## Supported Platforms

CloudSync Pro v4.0 supports the following platforms:

- Windows 10 and 11 (64-bit)
- macOS 12 (Monterey) and later
- Ubuntu 20.04 LTS and later
- Red Hat Enterprise Linux 8 and later
- Docker containers (official image provided)

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU cores | 4 | 8 |
| RAM | 8 GB | 16 GB |
| Disk space | 50 GB | 200 GB SSD |
| Network | 100 Mbps | 1 Gbps |

## Key Features

### Real-Time Synchronisation

CloudSync Pro maintains a synchronisation latency of under **200 milliseconds** for file changes up to 50 MB. Changes larger than 50 MB are queued and processed asynchronously with a maximum latency of **30 minutes** by default, configurable up to 24 hours.

### AI Workflow Automation

Version 4.0 introduces the AI Automation Engine, which uses pre-trained models (not Arcturus-proprietary) to detect repetitive file-handling tasks and suggest automation rules. The feature requires a minimum of **90 days of usage data** before suggestions are available.

### Security

- All data is encrypted in transit using TLS 1.3.
- Data at rest is encrypted using AES-256.
- Role-based access control supports up to **500 distinct roles** per organisation.
- Single sign-on (SSO) is supported via SAML 2.0 and OAuth 2.0.

### API

The CloudSync Pro REST API supports up to **1,000 requests per minute** per API key. Webhooks are available for 23 event types. The API is versioned; v1 and v2 are currently supported; v2 adds batch operations and cursor-based pagination.

## Storage Limits

| Plan | Storage per user | Max file size | Max users |
|------|-----------------|---------------|-----------|
| Professional | 100 GB | 5 GB | 50 |
| Business | 500 GB | 25 GB | 500 |
| Enterprise | Unlimited | 100 GB | Unlimited |

## Pricing

- **Professional**: 25 dollars per user per month
- **Business**: 45 dollars per user per month
- **Enterprise**: Custom pricing; contact sales

Volume discounts begin at 50 users (5 percent) and scale to a maximum discount of 30 percent at 500 or more users.

## Changelog: v4.0 vs v3.x

| Feature | v3.x | v4.0 |
|---------|------|------|
| Sync latency (small files) | 500 ms | 200 ms |
| Max file size (Enterprise) | 25 GB | 100 GB |
| AI Automation Engine | Not available | Available |
| API rate limit | 500 req/min | 1,000 req/min |
| Max RBAC roles | 100 | 500 |
