# AWS Infrastructure

Back to the [main README](../../README.md).

Reference architecture for mini-rag:

```
Browser (React, S3 + CloudFront)
        |
   Application Load Balancer  (public subnets, 2+ AZs)
        |
   API container (ECS Fargate, private subnets, 2+ AZs)
        |-- Amazon Bedrock (LLM + embeddings)       via VPC endpoint
        |-- Vector store: RDS PostgreSQL + pgvector  (private/isolated subnets, Multi-AZ)
        |-- S3 (source documents)                    via gateway endpoint
        |-- Secrets Manager / SSM Parameter Store    via VPC endpoint
```

## Bedrock vs calling OpenAI directly: pros and cons

| | Amazon Bedrock | OpenAI API directly |
|---|---|---|
| **Auth** | IAM roles, no API keys to store or rotate | API key you must store, rotate and protect |
| **Network and data** | Stays inside AWS; can use VPC endpoints (PrivateLink); data is not used to train the base models | Traffic goes over the public internet; check OpenAI's data-use and retention terms (enterprise options exist) |
| **Models** | Several providers behind one API (Anthropic, Meta, Amazon, Mistral, and others) | OpenAI models only, often first to get new features |
| **Compliance and governance** | Fits existing AWS compliance, CloudTrail audit, billing, Guardrails, Knowledge Bases | Separate vendor to assess; separate billing |
| **Observability** | CloudWatch metrics and invocation logging out of the box | Build or buy it (usage dashboard, your own logging) |
| **Features and speed of access** | New models and features can arrive later, and availability varies by region | Usually the newest models and the most complete feature set first |
| **Quotas** | Per-region, per-model quotas; may need quota increase requests; provisioned throughput available | Rate-limit tiers by usage; Azure OpenAI is another route |
| **Lock-in** | Tied to AWS, but easy to swap models (Converse API); the code differs from the OpenAI SDK | Tied to OpenAI's API; many tools assume the OpenAI format |
| **Cost** | Per-token pricing is similar; no extra infrastructure to run | Per-token pricing; no AWS integration costs |

**Rule of thumb:** If you already run on AWS and care about IAM, private networking, and audit, choose Bedrock. Choose OpenAI directly when you need a specific OpenAI-only model or feature. Put the LLM behind a thin interface in your code so you can switch or fall back.

## ECS Fargate vs EKS vs Lambda for this app

| | ECS Fargate | EKS | Lambda |
|---|---|---|---|
| Model | Containers, no servers to manage | Managed Kubernetes | Functions, event-driven |
| Ops effort | Low | High (cluster upgrades, add-ons, networking, RBAC) | Lowest |
| Long-lived connections and streaming | Good (ALB, SSE and WebSockets) | Good | Possible with response streaming via a function URL, but with limits; API Gateway is awkward for it |
| Run time limits | None | None | Max 15 minutes per invocation |
| Cold starts | None for running tasks | None | Yes (mitigate with provisioned concurrency) |
| Scaling | Task auto scaling on CPU/requests, takes tens of seconds | Very flexible | Instant and to zero |
| Cost pattern | Pay for running tasks, even when idle | Cluster fee plus nodes | Pay per request and duration |
| Best for | Steady web APIs and workers | Large platform teams, multi-service, existing Kubernetes skills | Spiky and low-traffic APIs, glue code, and event-driven jobs |

**Recommendation for this app: ECS Fargate** for the API. It is a long-running HTTP service that streams responses and holds a connection pool to the database. It needs no cluster management, and it keeps the cost and operational overhead of the small team low.

- Use **Lambda** for event-driven pieces: ingestion triggered by S3 uploads (or Step Functions for heavy pipelines), and scheduled jobs.
- Choose **EKS** only if the organization already standardizes on Kubernetes, or you need its ecosystem (service mesh, custom operators).
- Lambda for the API is reasonable when traffic is low and spiky and you accept cold starts and need to verify streaming works with your setup.

## What's an IAM role and why is it better than access keys?

An **IAM role** is an AWS identity with permissions (policies) that is **assumed** by something, such as an ECS task, a Lambda function, an EC2 instance, or a CI job. AWS issues **temporary credentials** (via STS) that expire automatically, usually within hours.

**Access keys** are long-lived credentials (`AKIA...` plus a secret) tied to an IAM user.

Why roles are better:

- **No secret to store.** Credentials are injected into the workload by the platform; nothing sits in code, env files, or images.
- **Short-lived**: a leaked credential expires quickly. Leaked long-lived keys, for example in a public Git repo, are a very common cause of breaches.
- **Automatic rotation** by AWS.
- **Least privilege per workload**: each service gets its own role scoped to exactly what it needs.
- **Auditable**: CloudTrail shows which role was assumed and by whom.
- **Works across accounts** with trust policies.

Two roles matter for ECS:

- **Task execution role**: used by ECS itself to pull the image from ECR, write logs, and fetch secrets for injection.
- **Task role**: used by *your application code* (for example `bedrock:InvokeModel`, `s3:GetObject` on one bucket).

Example task role policy (least privilege):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": "arn:aws:bedrock:eu-central-1::foundation-model/anthropic.claude-*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::mini-rag-docs/*"
    }
  ]
}
```

For CI/CD, use GitHub Actions **OIDC** to assume a role rather than storing access keys as repository secrets.

## Public vs private subnet, NAT gateway, security group

**VPC** is your private network in AWS, divided into **subnets**, each in one Availability Zone.

- **Public subnet**: its route table has a route to an **Internet Gateway** (`0.0.0.0/0 -> igw-...`). Resources there can receive traffic from the internet if they have a public IP and the security group allows it. Put the **ALB** and NAT gateways here.
- **Private subnet**: no direct route from the internet. Resources can't be reached from outside, but may reach out through a NAT gateway. Put **Fargate tasks** here. Put the **database** in private (or isolated, with no outbound route at all) subnets.
- **NAT gateway**: sits in a *public* subnet and lets resources in *private* subnets make **outbound** connections (to pull packages, call an external API such as OpenAI) without allowing inbound connections from the internet. It is billed per hour plus per GB processed, so it is often a major cost line. Deploy one per AZ for resilience (or one total for dev to save money).
  - **VPC endpoints** reduce NAT traffic and improve security: gateway endpoints for S3 (free) and interface endpoints (PrivateLink) for ECR, CloudWatch Logs, Secrets Manager, and Bedrock. With these, tasks may not need internet access at all.
- **Security group**: a **stateful** virtual firewall attached to a resource (ENI). Allow rules only; return traffic is automatically allowed. Reference other security groups instead of IP ranges.
  - ALB SG: allow 443 from `0.0.0.0/0`.
  - App SG: allow 8080 **only from the ALB SG**.
  - DB SG: allow 5432 **only from the App SG**.
- **Network ACL** (for contrast): stateless, subnet-level, with allow and deny rules. Usually left at defaults.
