# Delivery and Operations

Back to the [main README](../../README.md).

## Walk me through CI/CD for this app on AWS

Goal: every merge to `main` is tested, built into an immutable image, and deployed safely with a quick rollback. Any step can fail the pipeline.

```
PR -> CI checks -> merge to main -> build image -> push to ECR -> deploy to staging
   -> smoke + RAG evaluation -> (approval) -> deploy to production (rolling/blue-green)
```


### Pull request (CI)

1. Lint, type-check, unit tests (frontend and backend).
2. **RAG evaluation suite** on a small golden set to catch prompt/retrieval regressions (see the [evaluation guide](../evaluation-and-performance/README.md)); mock or use a cheap model to control cost.
3. Dependency and secret scanning, container image scan (for example ECR scanning, Trivy), and IaC checks (`terraform validate`, `cdk synth`, `checkov`).

### Build (on merge to main)
1. Authenticate to AWS using **GitHub Actions OIDC** (assume a role; no stored access keys).
2. Build the Docker image; tag with the **git SHA** (immutable), not just `latest`.
3. Push to **Amazon ECR**.
4. Build the React app and upload it to S3, then invalidate CloudFront.

### Deploy
1. Infrastructure is code (Terraform or CDK), applied through the pipeline: `plan` on PR, `apply` on merge, with remote state and locking.
2. Register a new **ECS task definition revision** pointing to the new image tag and update the service.
3. **Rolling deployment** (ECS default) with a minimum healthy percent, or **blue/green with CodeDeploy** for traffic shifting (canary or linear) and automatic rollback.
4. Deployment gates: ALB health checks pass, **CloudWatch alarms** (5xx rate, latency) stay green; the ECS **deployment circuit breaker** rolls back automatically if new tasks fail to become healthy.
5. Promote staging -> production, optionally with a manual approval for production.

### Data and ingestion changes
- Database migrations run as a separate, backward-compatible step (expand, then contract), never in a way that breaks the currently running version.
- If you change the **embedding model or chunking**, build a new index alongside the old one, evaluate it, switch over with a config flag or alias, and keep the old one for rollback.

### Rollback
Redeploy the previous task definition revision (previous image SHA). Because images are immutable and config is separate, this takes minutes.

Tooling options: GitHub Actions (with OIDC) or AWS-native CodePipeline + CodeBuild + CodeDeploy. The stages are the same.

## How do you handle secrets and config?

**Separate three things:**

| Kind | Examples | Where |
|------|----------|-------|
| Non-secret config | Model ID, region, top-k, chunk size, feature flags, log level | Environment variables set in the task definition, sourced from **SSM Parameter Store** (or AppConfig for runtime changes) |
| Secrets | DB password, third-party API keys (for example an OpenAI key, if used) | **AWS Secrets Manager** (rotation supported) or SSM SecureString |
| Identity and credentials for AWS | Access to Bedrock, S3 | **IAM task role**, never keys |

Practices:

- **Prefer not having secrets at all.** Use IAM roles for AWS services, IAM database authentication for RDS where suitable, and OIDC for CI.
- **Never** commit secrets, bake them into images, or put them in frontend code. Anything shipped to the browser is public; the React app talks to *your* backend, which holds the keys.
- **Inject at runtime**: ECS can pull a Secrets Manager or SSM value into an environment variable via the task definition's `secrets` field, using the **execution role**. Scope that role to the specific secret ARNs.
- **Rotate** secrets regularly (Secrets Manager rotation for RDS); the app should handle rotation (re-read credentials or reconnect on auth failure).
- **Encrypt** with KMS, and restrict who can read secrets. CloudTrail records every access.
- **One set per environment** (dev, staging, prod) with separate accounts or at least separate paths (`/mini-rag/prod/...`).
- **Don't log secrets or full prompts containing sensitive data**; redact. Use `git-secrets` or `gitleaks` in CI and pre-commit hooks.
- Local development: a `.env` file that is git-ignored, with an `.env.example` committed that contains only placeholders.

## How would you make this multi-AZ and observable?

### Multi-AZ (high availability)

An Availability Zone is a physically separate data center within a region. Design so that **losing one AZ doesn't cause an outage**.

- **VPC**: at least 2 (ideally 3) AZs, with a public and a private subnet in each.
- **ALB** across all AZs' public subnets, with health checks.
- **ECS service**: `desiredCount >= 2`, spread across AZs (the default spread placement), and **auto scaling** with enough headroom so the remaining AZs can absorb the load.
- **NAT gateway per AZ** (or use VPC endpoints), so a single AZ failure doesn't cut off outbound access for other AZs.
- **Database**: RDS/Aurora **Multi-AZ** with automatic failover (for OpenSearch, use multiple data nodes and replicas across AZs). Enable automated backups and point-in-time recovery; test restores.
- **S3** is already multi-AZ. Use versioning for documents.
- **Bedrock/LLM dependency**: handle throttling and regional errors with retries (exponential backoff and jitter), timeouts, and circuit breakers. Optionally use cross-region inference profiles or a fallback model for resilience.
- **Stateless app**: no local state in containers; sessions and caches in a shared store (for example ElastiCache, or the client).
- **Multi-region** is a further step (Route 53 failover, replicated data) and only needed with strict recovery targets. Define RTO/RPO first.
- Test it: terminate tasks, fail over the database, and do game days.

### Observability

Cover the three pillars (logs, metrics, traces) plus LLM-specific signals.

**Logs**
- Structured JSON logs to **CloudWatch Logs** with a request ID/trace ID on every line; set retention.
- Log the pipeline steps: query, retrieved chunk IDs and scores, model ID, token counts, latency. Redact PII, and treat prompts and answers as sensitive data (consider sampling).

**Metrics** (CloudWatch, with custom metrics via EMF)
- Service: request rate, 4xx/5xx, p50/p95/p99 latency, CPU/memory, task count, ALB healthy hosts.
- LLM: time-to-first-token, total generation time, input/output tokens, **cost per request**, throttling rate, error rate by provider.
- RAG quality: retrieval score distribution, "no answer" rate, citation rate, user thumbs up/down.
- Database: connections, CPU, query latency, replica lag.

**Traces**
- **OpenTelemetry** (with AWS Distro for OpenTelemetry) to **X-Ray** or another backend, with spans for embed -> search -> rerank -> LLM call so you can see which stage is slow.

**Alerts and dashboards**
- Alarms on 5xx rate, p95 latency, unhealthy hosts, Bedrock throttling, DB failover/CPU, and **token spend anomalies**; route to SNS, PagerDuty, or Slack.
- Dashboards for service health, plus a cost and quality view.
- **Synthetic checks** (CloudWatch Synthetics) hitting a health endpoint and a canned question.
- Enable **CloudTrail** and Bedrock model invocation logging (to S3/CloudWatch) for audit, with attention to what is stored.
- Define SLOs (for example 99.9% availability, p95 time-to-first-token under 2s) and alert on burn rate.
