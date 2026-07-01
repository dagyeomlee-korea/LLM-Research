# AIS팀 도구 및 기술 스택 현황

AIS팀 내에서 사용 중인 도구와 기술 스택 현황을 소개합니다.

## 1. 개발 도구

### A. 커뮤니케이션

#### Wiki

- Microsoft Teams: Teams Channel 내 Shared Files
- github share-document repo: 

제품 기획은 팀 내에서만 공유되는 비공개 문서입니다. 이외의 폴더는 공개 문서입니다.

팀 내에서 사용하는 공유 계정 및 내부 문서는 `Teams > AI Security 채널 > 공유`에서 확인할 수 있습니다.

#### Issue Tracking

- 각 git repository의 Issue 기능 사용

#### Chat & Schedule

- [Microsoft Teams](https://teams.cloud.microsoft/l/channel/19%3AnV1XFtvV0G0pYid_Pq02G9xyy9t5i3THkyEh3L9jjRI1%40thread.tacv2/%EC%9D%BC%EB%B0%98?groupId=698636e5-928d-4431-8881-57e467b5b68a&tenantId=50898671-e85d-44b2-b9bb-bc1f3e3e1ecd)

### B. 개발

#### 개발 IDE

- 자유 툴 사용
- 별도 지원 없음

#### AI 구독

- Claude Max: 개인 지원
- Codex Pro: 공통 계정

#### Git

- [Atlassian Bitbucket Cloud](https://bitbucket.org/coontec/workspace/projects/AIS)
- [GitHub](https://github.com/orgs/coontec-ai-labs/repositories)

### C. 오피스

- Microsoft Office
- 한글

### D. 인프라

- Terraform Cloud

## 2. 기술 스택

| 구분 | 사용 기술 |
| --- | --- |
| AWS Infra 운영 | Terraform |
| Container Registry | Harbor |
| CI/CD | Jenkins, Ansible |
| LLM Serving | vLLM, SGLang |
| RDB | PostgreSQL 또는 Aurora RDS |
| Kubernetes | K3s 예정 |
| Message Queue | 미정 |
| Cache | 미정 |
| OS | Amazon Linux, Ubuntu 24, Proxmox VE |
| VM | Proxmox, Docker |

## 3. Cloud Infra Spec

### AWS

- ALB
- NAT Gateway
- Internet Gateway
- EC2
- ECS 예정
- Route 53
- WAF
- RDS 예정
- ElastiCache 예정
- ECR 예정
